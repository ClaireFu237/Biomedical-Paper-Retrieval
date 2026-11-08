import os
import json
import math
import pickle
import argparse
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import torch
from sentence_transformers import SentenceTransformer, util


# Evaluate dense retrieval models on BioASQ-style query-document pairs.
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_json(path: str, obj: Any) -> None:
    """
    Save an object to a JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(text: Any) -> str:
    """
    Convert a value to a normalized string.
    """
    if text is None:
        return ""
    return str(text).strip()


def tokenize_for_overlap(text: str) -> List[str]:
    """
    A simple tokenizer for lexical overlap scoring.
    """
    text = text.lower()
    cleaned = []
    current = []

    for ch in text:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                cleaned.append("".join(current))
                current = []

    if current:
        cleaned.append("".join(current))

    return cleaned


def lexical_overlap_score(query: str, document: str) -> float:
    """
    Compute a simple lexical overlap score between query and document.

    Score = |intersection(query_tokens, doc_tokens)| / |query_tokens|
    """
    q_tokens = set(tokenize_for_overlap(query))
    d_tokens = set(tokenize_for_overlap(document))

    if not q_tokens:
        return 0.0

    return len(q_tokens & d_tokens) / len(q_tokens)


def build_doc_text(doc: Dict[str, Any], doc_mode: str) -> str:
    """
    Build the text used for document embedding.

    Supported modes:
    - abstract
    - title
    - title_abstract
    """
    title = normalize_text(doc.get("title"))
    abstract = normalize_text(doc.get("abstract"))
    text = normalize_text(doc.get("text"))

    if doc_mode == "abstract":
        return abstract

    if doc_mode == "title":
        return title

    if doc_mode == "title_abstract":
        if text:
            return text
        if title and abstract:
            return f"Title: {title} [SEP] Abstract: {abstract}"
        return title or abstract

    raise ValueError(f"Unsupported doc_mode: {doc_mode}")


def build_query_text(query: Dict[str, Any], query_prefix: bool) -> str:
    """
    Build the text used for query embedding.
    """
    q = normalize_text(query.get("query_text"))
    if query_prefix:
        return f"Question: {q}"
    return q


def build_eval_queries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize supported evaluation files into:
    query_id, query_text, question_type, relevant_doc_ids.

    Supported inputs:
    - eval_queries.jsonl from prepare_eval_data.py
    - retrieval_dev.jsonl from scriptsv2/get_finetuning_data.py
    """
    # Merge duplicate query rows so each query can have multiple relevant docs.
    query_map: Dict[str, Dict[str, Any]] = {}

    for i, row in enumerate(rows):
        if "relevant_doc_ids" in row:
            query_id = normalize_text(row.get("query_id") or row.get("question_id") or f"q_{i}")
            query_text = normalize_text(row.get("query_text") or row.get("query"))
            question_type = normalize_text(row.get("question_type"))
            relevant_doc_ids = [
                normalize_text(x)
                for x in row.get("relevant_doc_ids", [])
                if normalize_text(x)
            ]
        else:
            query_text = normalize_text(row.get("query") or row.get("query_text"))
            query_id = normalize_text(row.get("question_id") or query_text or f"q_{i}")
            question_type = normalize_text(row.get("question_type"))
            doc_id = normalize_text(row.get("doc_id") or row.get("pmid"))
            relevant_doc_ids = [doc_id] if doc_id else []

        if not query_id or not query_text or not relevant_doc_ids:
            continue

        if query_id not in query_map:
            query_map[query_id] = {
                "query_id": query_id,
                "query_text": query_text,
                "question_type": question_type,
                "relevant_doc_ids": set(),
            }
        else:
            if query_map[query_id]["query_text"] != query_text:
                print(f"Warning: query_id={query_id} has inconsistent query text. Keeping first version.")
            if query_map[query_id]["question_type"] != question_type:
                print(f"Warning: query_id={query_id} has inconsistent question type. Keeping first version.")

        query_map[query_id]["relevant_doc_ids"].update(relevant_doc_ids)

    eval_queries = []
    for value in query_map.values():
        eval_queries.append(
            {
                "query_id": value["query_id"],
                "query_text": value["query_text"],
                "question_type": value["question_type"],
                "relevant_doc_ids": sorted(value["relevant_doc_ids"]),
            }
        )

    return eval_queries


def resolve_model_name(model_name: str, model_dir: str) -> str:
    """
    Prefer a local model path. Fall back to known Hugging Face aliases.
    """
    if os.path.isdir(model_name):
        return model_name

    local_path = os.path.join(model_dir, model_name)
    if os.path.isdir(local_path):
        return local_path

    model_aliases = {
        "scibert-nli": "gsarti/scibert-nli",
        "biobert-nli": "gsarti/biobert-nli",
    }
    return model_aliases.get(model_name, model_name)


def get_cache_path(cache_dir: str, model_name: str, doc_mode: str) -> str:
    """
    Build the cache file path for corpus embeddings.
    """
    safe_model_name = model_name.replace("/", "_")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{safe_model_name}-eval-embeddings-{doc_mode}.pkl")


def encode_corpus(
    model: SentenceTransformer,
    corpus: List[Dict[str, Any]],
    doc_mode: str,
    cache_dir: str,
    model_name: str,
    batch_size: int = 64,
    device: str = "cpu",
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Encode the corpus texts and cache the embeddings.

    Returns:
    - corpus_embeddings
    - corpus_doc_ids
    - corpus_texts
    """
    cache_path = get_cache_path(cache_dir, model_name, doc_mode)

    corpus_doc_ids = [normalize_text(doc["doc_id"]) for doc in corpus]
    corpus_texts = [build_doc_text(doc, doc_mode) for doc in corpus]

    # Reuse cached embeddings only when both IDs and texts match the corpus.
    if os.path.exists(cache_path):
        print(f"Loading cached corpus embeddings from: {cache_path}")
        with open(cache_path, "rb") as f:
            cache_obj = pickle.load(f)

        cached_doc_ids = cache_obj["doc_ids"]
        cached_embeddings = cache_obj["embeddings"]
        cached_texts = cache_obj["texts"]

        if cached_doc_ids == corpus_doc_ids and cached_texts == corpus_texts:
            return cached_embeddings.to(device), corpus_doc_ids, corpus_texts
        else:
            print("Cache content does not match current corpus. Recomputing embeddings.")

    print("Computing corpus embeddings...")
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "doc_ids": corpus_doc_ids,
                "texts": corpus_texts,
                "embeddings": corpus_embeddings.cpu(),
            },
            f,
        )

    print(f"Saved corpus embeddings to: {cache_path}")
    return corpus_embeddings.to(device), corpus_doc_ids, corpus_texts


def reciprocal_rank(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Compute reciprocal rank at k.
    """
    relevant_set = set(relevant_doc_ids)
    for idx, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / idx
    return 0.0


def hit_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Compute Hit@k.
    """
    relevant_set = set(relevant_doc_ids)
    return 1.0 if any(doc_id in relevant_set for doc_id in retrieved_doc_ids[:k]) else 0.0


def recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Compute Recall@k.
    """
    relevant_set = set(relevant_doc_ids)
    if not relevant_set:
        return 0.0

    retrieved_set = set(retrieved_doc_ids[:k])
    hits = len(retrieved_set & relevant_set)
    return hits / len(relevant_set)


def dcg_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Compute DCG@k for binary relevance.
    """
    relevant_set = set(relevant_doc_ids)
    dcg = 0.0

    for i, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        rel = 1.0 if doc_id in relevant_set else 0.0
        if rel > 0:
            dcg += rel / math.log2(i + 1)

    return dcg


def ndcg_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Compute nDCG@k for binary relevance.
    """
    actual_dcg = dcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)

    ideal_hits = min(len(relevant_doc_ids), k)
    if ideal_hits == 0:
        return 0.0

    ideal_dcg = 0.0
    for i in range(1, ideal_hits + 1):
        ideal_dcg += 1.0 / math.log2(i + 1)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def rerank_candidates(
    query_text: str,
    candidate_indices: List[int],
    dense_scores: torch.Tensor,
    corpus_texts: List[str],
    rerank_alpha: float,
) -> List[Tuple[int, float]]:
    """
    Re-rank dense candidates using a weighted combination of:
    - normalized dense score
    - lexical overlap score

    final_score = alpha * dense_score + (1 - alpha) * overlap_score
    """
    reranked = []

    # Combine dense similarity with a lightweight lexical overlap signal.
    for idx in candidate_indices:
        dense_score = float(dense_scores[idx].item())
        overlap_score = lexical_overlap_score(query_text, corpus_texts[idx])
        final_score = rerank_alpha * dense_score + (1.0 - rerank_alpha) * overlap_score
        reranked.append((idx, final_score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def evaluate(
    model: SentenceTransformer,
    corpus: List[Dict[str, Any]],
    eval_queries: List[Dict[str, Any]],
    model_name: str,
    doc_mode: str,
    ks: List[int],
    cache_dir: str,
    batch_size: int,
    device: str,
    query_prefix: bool,
    rerank: bool,
    rerank_pool_size: int,
    rerank_alpha: float,
) -> Dict[str, Any]:
    """
    Run retrieval evaluation and return detailed results.
    """
    max_k = max(ks)
    max_k = min(max_k, len(corpus))
    if max_k <= 0:
        raise ValueError("Corpus is empty; cannot run retrieval evaluation.")

    corpus_embeddings, corpus_doc_ids, corpus_texts = encode_corpus(
        model=model,
        corpus=corpus,
        doc_mode=doc_mode,
        cache_dir=cache_dir,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )

    # Encode all queries once before the per-query ranking loop.
    query_texts = [build_query_text(q, query_prefix=query_prefix) for q in eval_queries]
    query_embeddings = model.encode(
        query_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).to(device)

    per_query_results = []

    metric_sums = {k: {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0} for k in ks}
    per_type_metric_sums = defaultdict(lambda: {k: {"hit": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0} for k in ks})
    per_type_counts = defaultdict(int)

    print("Running retrieval evaluation...")
    for i, query in enumerate(eval_queries):
        query_embedding = query_embeddings[i]
        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

        if rerank:
            # Rerank only a small dense candidate pool for speed.
            dense_top_n = min(rerank_pool_size, len(corpus_doc_ids))
            dense_top_indices = torch.topk(scores, k=dense_top_n).indices.tolist()
            reranked = rerank_candidates(
                query_text=normalize_text(query["query_text"]),
                candidate_indices=dense_top_indices,
                dense_scores=scores,
                corpus_texts=corpus_texts,
                rerank_alpha=rerank_alpha,
            )
            final_top_indices = [idx for idx, _ in reranked[:max_k]]
            final_top_scores = [score for _, score in reranked[:max_k]]
        else:
            top_values, top_indices = torch.topk(scores, k=max_k)
            final_top_indices = top_indices.tolist()
            final_top_scores = top_values.tolist()

        retrieved_doc_ids = [corpus_doc_ids[idx] for idx in final_top_indices]
        relevant_doc_ids = query["relevant_doc_ids"]
        qtype = normalize_text(query.get("question_type")) or "unknown"

        top_k_results = []
        for rank, (doc_idx, score) in enumerate(zip(final_top_indices, final_top_scores), start=1):
            doc = corpus[doc_idx]
            top_k_results.append(
                {
                    "rank": rank,
                    "doc_id": normalize_text(doc["doc_id"]),
                    "score": float(score),
                    "title": normalize_text(doc.get("title")),
                    "abstract": normalize_text(doc.get("abstract")),
                }
            )

        # Compute every requested metric cutoff for this query.
        metric_record = {}
        for k in ks:
            hit = hit_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            recall = recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
            mrr = reciprocal_rank(retrieved_doc_ids, relevant_doc_ids, k)
            ndcg = ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)

            metric_record[f"Hit@{k}"] = hit
            metric_record[f"Recall@{k}"] = recall
            metric_record[f"MRR@{k}"] = mrr
            metric_record[f"nDCG@{k}"] = ndcg

            metric_sums[k]["hit"] += hit
            metric_sums[k]["recall"] += recall
            metric_sums[k]["mrr"] += mrr
            metric_sums[k]["ndcg"] += ndcg

            per_type_metric_sums[qtype][k]["hit"] += hit
            per_type_metric_sums[qtype][k]["recall"] += recall
            per_type_metric_sums[qtype][k]["mrr"] += mrr
            per_type_metric_sums[qtype][k]["ndcg"] += ndcg

        per_type_counts[qtype] += 1

        per_query_results.append(
            {
                "query_id": query["query_id"],
                "query_text": normalize_text(query["query_text"]),
                "question_type": qtype,
                "relevant_doc_ids": relevant_doc_ids,
                "retrieved_doc_ids": retrieved_doc_ids,
                "metrics": metric_record,
                "top_k_results": top_k_results,
            }
        )

    num_queries = len(eval_queries)
    summary = {"num_queries": num_queries}

    for k in ks:
        summary[f"avg_Hit@{k}"] = metric_sums[k]["hit"] / num_queries if num_queries else 0.0
        summary[f"avg_Recall@{k}"] = metric_sums[k]["recall"] / num_queries if num_queries else 0.0
        summary[f"avg_MRR@{k}"] = metric_sums[k]["mrr"] / num_queries if num_queries else 0.0
        summary[f"avg_nDCG@{k}"] = metric_sums[k]["ndcg"] / num_queries if num_queries else 0.0

    # Aggregate the same metrics by BioASQ question type.
    per_question_type = {}
    for qtype, count in per_type_counts.items():
        per_question_type[qtype] = {"count": count}
        for k in ks:
            per_question_type[qtype][f"avg_Hit@{k}"] = per_type_metric_sums[qtype][k]["hit"] / count if count else 0.0
            per_question_type[qtype][f"avg_Recall@{k}"] = per_type_metric_sums[qtype][k]["recall"] / count if count else 0.0
            per_question_type[qtype][f"avg_MRR@{k}"] = per_type_metric_sums[qtype][k]["mrr"] / count if count else 0.0
            per_question_type[qtype][f"avg_nDCG@{k}"] = per_type_metric_sums[qtype][k]["ndcg"] / count if count else 0.0

    return {
        "config": {
            "model_name": model_name,
            "doc_mode": doc_mode,
            "ks": ks,
            "query_prefix": query_prefix,
            "rerank": rerank,
            "rerank_pool_size": rerank_pool_size,
            "rerank_alpha": rerank_alpha,
        },
        "summary": summary,
        "per_question_type": per_question_type,
        "per_query_results": per_query_results,
    }


def print_summary(results: Dict[str, Any]) -> None:
    """
    Print the evaluation summary to stdout.
    """
    print("\n===== Evaluation Summary =====")
    for k, v in results["summary"].items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n===== Per Question Type =====")
    for qtype, stats in sorted(results["per_question_type"].items()):
        print(f"\n[{qtype}]")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True, help="SentenceTransformer model name or local path")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory for local model folders")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus_with_ids.jsonl")
    parser.add_argument(
        "--eval_queries_path",
        type=str,
        required=True,
        help="Path to eval_queries.jsonl or retrieval_dev.jsonl"
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save results JSON")

    parser.add_argument(
        "--doc_mode",
        type=str,
        default="title_abstract",
        choices=["abstract", "title", "title_abstract"],
        help="Document text mode for embedding"
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="1,3,5,10,20",
        help="Comma-separated k values, e.g. 1,3,5,10,20"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data",
        help="Directory to store cached corpus embeddings"
    )
    parser.add_argument(
        "--query_prefix",
        action="store_true",
        help='Prefix queries with "Question: "'
    )

    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable lightweight lexical reranking on top dense candidates"
    )
    parser.add_argument(
        "--rerank_pool_size",
        type=int,
        default=20,
        help="Number of top dense candidates to rerank"
    )
    parser.add_argument(
        "--rerank_alpha",
        type=float,
        default=0.8,
        help="Weight for dense score in reranking; lexical overlap gets (1-alpha)"
    )

    args = parser.parse_args()

    # Parse cutoffs once so every metric uses the same ordered k values.
    ks = sorted({int(x.strip()) for x in args.ks.split(",") if x.strip()})
    if not ks:
        raise ValueError("At least one k must be provided in --ks")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model...")
    model_name = resolve_model_name(args.model_name, args.model_dir)
    print(f"Resolved model path/name: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    print("Loading corpus...")
    corpus = load_jsonl(args.corpus_path)
    print(f"Loaded {len(corpus)} corpus documents")

    print("Loading evaluation queries...")
    eval_rows = load_jsonl(args.eval_queries_path)
    # Normalize either grouped eval files or row-per-positive retrieval files.
    eval_queries = build_eval_queries(eval_rows)
    print(f"Loaded {len(eval_queries)} evaluation queries")
    print(f"Loaded {sum(len(x['relevant_doc_ids']) for x in eval_queries)} relevant query-paper pairs")

    results = evaluate(
        model=model,
        corpus=corpus,
        eval_queries=eval_queries,
        model_name=args.model_name,
        doc_mode=args.doc_mode,
        ks=ks,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        device=device,
        query_prefix=args.query_prefix,
        rerank=args.rerank,
        rerank_pool_size=args.rerank_pool_size,
        rerank_alpha=args.rerank_alpha,
    )

    save_json(args.output_path, results)
    print_summary(results)
    print(f"\nSaved evaluation results to: {args.output_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Set

import torch
from sentence_transformers import SentenceTransformer, util


# Mine high-scoring non-gold documents for each training query.
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_doc_text(row: Dict[str, Any], doc_mode: str) -> str:
    # Keep corpus text formatting aligned with retrieval evaluation.
    title = normalize_text(row.get("title"))
    abstract = normalize_text(row.get("abstract"))
    text = normalize_text(row.get("text"))

    if doc_mode == "title":
        return title
    if doc_mode == "abstract":
        return abstract or text
    if doc_mode == "title_abstract":
        if text:
            return text
        if title and abstract:
            return f"Title: {title} [SEP] Abstract: {abstract}"
        return title or abstract
    raise ValueError(f"Unsupported doc_mode: {doc_mode}")


def resolve_model_name(model_name: str, model_dir: str) -> str:
    # Prefer local fine-tuned models, then fall back to known Hugging Face aliases.
    if os.path.isdir(model_name):
        return model_name
    local_path = os.path.join(model_dir, model_name)
    if os.path.isdir(local_path):
        return local_path
    aliases = {
        "scibert-nli": "gsarti/scibert-nli",
        "biobert-nli": "gsarti/biobert-nli",
    }
    return aliases.get(model_name, model_name)


def build_relevant_by_query(rows: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    # Gold docs for the same query are forbidden as hard negatives.
    relevant = defaultdict(set)
    for i, row in enumerate(rows):
        query = normalize_text(row.get("query") or row.get("query_text"))
        qid = normalize_text(row.get("question_id") or query or f"q_{i}")
        doc_id = normalize_text(row.get("doc_id") or row.get("pmid"))
        if qid and doc_id:
            relevant[qid].add(doc_id)
    return relevant


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mine hard negatives by retrieving high-scoring non-gold documents for each training query."
        )
    )
    parser.add_argument("--model_name", type=str, default="scibert-nli_retrieval")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--train_path", type=str, default="data/retrieval_train.jsonl")
    parser.add_argument("--corpus_path", type=str, default="result/resultv4/augmented_corpus_noise1000_seed42.jsonl")
    parser.add_argument("--output_path", type=str, default="data/retrieval_train_hardneg.jsonl")
    parser.add_argument("--doc_mode", type=str, default="title_abstract", choices=["title", "abstract", "title_abstract"])
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--hard_negatives_per_query", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=256)
    args = parser.parse_args()

    if args.hard_negatives_per_query < 1:
        raise ValueError("--hard_negatives_per_query must be at least 1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_path = resolve_model_name(args.model_name, args.model_dir)
    print(f"Loading mining model: {model_path}")
    model = SentenceTransformer(model_path, device=device)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_seq_length

    train_rows = load_jsonl(args.train_path)
    corpus_rows = load_jsonl(args.corpus_path)

    # Build parallel lists so embedding indices can map back to document IDs.
    corpus_doc_ids = []
    corpus_texts = []
    for row in corpus_rows:
        doc_id = normalize_text(row.get("doc_id") or row.get("pmid"))
        text = get_doc_text(row, args.doc_mode)
        if doc_id and text:
            corpus_doc_ids.append(doc_id)
            corpus_texts.append(text)

    print(f"Loaded train rows: {len(train_rows)}")
    print(f"Loaded candidate corpus docs: {len(corpus_doc_ids)}")

    print("Encoding candidate corpus...")
    # Encode the full candidate corpus once for efficient cosine search.
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).to(device)

    relevant_by_query = build_relevant_by_query(train_rows)
    mined_rows = []
    total_hard_negatives = 0
    skipped_no_negative = 0

    query_texts = [normalize_text(row.get("query") or row.get("query_text")) for row in train_rows]
    print("Encoding training queries...")
    query_embeddings = model.encode(
        query_texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).to(device)

    search_k = min(max(args.top_k, args.hard_negatives_per_query + 5), len(corpus_doc_ids))
    print("Mining hard negatives...")
    for i, row in enumerate(train_rows):
        query = query_texts[i]
        positive = normalize_text(row.get("positive") or row.get("text"))
        qid = normalize_text(row.get("question_id") or query or f"q_{i}")
        forbidden = set(relevant_by_query.get(qid, set()))

        if not query or not positive:
            continue

        scores = util.cos_sim(query_embeddings[i], corpus_embeddings)[0]
        top_indices = torch.topk(scores, k=search_k).indices.tolist()

        # Keep top-ranked documents that are not labeled relevant for this query.
        hard_negatives = []
        for idx in top_indices:
            doc_id = corpus_doc_ids[idx]
            text = corpus_texts[idx]
            if doc_id in forbidden:
                continue
            hard_negatives.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "score": float(scores[idx].item()),
                }
            )
            if len(hard_negatives) >= args.hard_negatives_per_query:
                break

        if not hard_negatives:
            skipped_no_negative += 1
            continue

        out = dict(row)
        out["hard_negatives"] = hard_negatives
        mined_rows.append(out)
        total_hard_negatives += len(hard_negatives)

    save_jsonl(mined_rows, args.output_path)
    print(f"Saved mined training rows: {args.output_path}")
    print(f"Rows with hard negatives: {len(mined_rows)}")
    print(f"Total hard negatives: {total_hard_negatives}")
    print(f"Skipped rows with no hard negative: {skipped_no_negative}")


if __name__ == "__main__":
    main()

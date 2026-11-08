import argparse
import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Set

import torch
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# Fine-tune a retriever with random negative documents.
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_doc_text(row: Dict[str, Any]) -> str:
    return normalize_text(
        row.get("text")
        or row.get("positive")
        or row.get("abstract")
        or row.get("title")
    )


def build_relevant_by_question(rows: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
    # Track gold document IDs so sampled negatives do not include positives.
    relevant = defaultdict(set)
    for i, row in enumerate(rows):
        qid = normalize_text(row.get("question_id") or row.get("query") or f"q_{i}")
        doc_id = normalize_text(row.get("doc_id") or row.get("pmid"))
        if qid and doc_id:
            relevant[qid].add(doc_id)
    return relevant


def build_negative_pool(noise_rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    # Deduplicate noise documents before sampling negatives.
    pool = []
    seen = set()
    for row in noise_rows:
        doc_id = normalize_text(row.get("doc_id") or row.get("pmid"))
        text = get_doc_text(row)
        if not doc_id or not text or doc_id in seen:
            continue
        pool.append({"doc_id": doc_id, "text": text})
        seen.add(doc_id)
    return pool


def sample_negative(
    negative_pool: List[Dict[str, str]],
    forbidden_doc_ids: Set[str],
    rng: random.Random,
    max_tries: int = 100,
) -> str:
    # Try random sampling first, then fall back to a linear scan.
    for _ in range(max_tries):
        candidate = rng.choice(negative_pool)
        if candidate["doc_id"] not in forbidden_doc_ids:
            return candidate["text"]
    for candidate in negative_pool:
        if candidate["doc_id"] not in forbidden_doc_ids:
            return candidate["text"]
    raise ValueError("Could not sample a negative document outside the forbidden set.")


def build_train_examples(
    train_rows: List[Dict[str, Any]],
    negative_pool: List[Dict[str, str]],
    negatives_per_positive: int,
    loss_type: str,
    seed: int,
) -> List[InputExample]:
    # Add sampled negatives to each query-positive pair.
    rng = random.Random(seed)
    relevant_by_question = build_relevant_by_question(train_rows)
    examples = []

    if not negative_pool:
        raise ValueError("Negative pool is empty. Build a noise corpus first.")

    for i, row in enumerate(train_rows):
        query = normalize_text(row.get("query") or row.get("query_text"))
        positive = normalize_text(row.get("positive") or row.get("text"))
        qid = normalize_text(row.get("question_id") or query or f"q_{i}")

        if not query or not positive:
            continue

        forbidden = set(relevant_by_question.get(qid, set()))
        negatives = [
            sample_negative(negative_pool, forbidden_doc_ids=forbidden, rng=rng)
            for _ in range(negatives_per_positive)
        ]

        if loss_type == "triplet":
            for negative in negatives:
                examples.append(InputExample(texts=[query, positive, negative]))
        else:
            examples.append(InputExample(texts=[query, positive] + negatives))

    return examples


def build_ir_evaluator(dev_rows: List[Dict[str, Any]], corpus_rows: List[Dict[str, Any]]) -> InformationRetrievalEvaluator:
    # Evaluate against the original corpus, not the random-negative pool.
    queries = {}
    relevant_docs = defaultdict(set)
    corpus = {}

    for row in corpus_rows:
        doc_id = normalize_text(row.get("doc_id") or row.get("pmid"))
        text = get_doc_text(row)
        if doc_id and text:
            corpus[doc_id] = text

    for i, row in enumerate(dev_rows):
        query = normalize_text(row.get("query") or row.get("query_text"))
        qid = normalize_text(row.get("question_id") or query or f"q_{i}")
        doc_id = normalize_text(row.get("doc_id") or row.get("pmid"))

        if not query or not qid or not doc_id or doc_id not in corpus:
            continue
        queries.setdefault(qid, query)
        relevant_docs[qid].add(doc_id)

    logging.info(f"Dev queries used for evaluator: {len(queries)}")
    logging.info(f"Dev corpus size: {len(corpus)}")

    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=dict(relevant_docs),
        show_progress_bar=True,
        precision_recall_at_k=[1, 3, 5, 10],
        map_at_k=[10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        name="retrieval-dev-random-negatives",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment C: retrieval fine-tuning with sampled random negatives from training13b snippets."
    )
    parser.add_argument("--model_name", type=str, default="scibert-nli")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="retrieval_train.jsonl")
    parser.add_argument("--dev_file", type=str, default="retrieval_dev.jsonl")
    parser.add_argument("--corpus_file", type=str, default="corpus_with_ids.jsonl")
    parser.add_argument("--noise_corpus_path", type=str, default="data/noise_corpus_1000.jsonl")
    parser.add_argument("--output_suffix", type=str, default="_retrieval_randomneg")
    parser.add_argument("--loss_type", type=str, default="mnrl", choices=["mnrl", "triplet"])
    parser.add_argument("--negatives_per_positive", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.negatives_per_positive < 1:
        raise ValueError("--negatives_per_positive must be at least 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_model_path = os.path.join(args.model_dir, args.model_name)
    output_path = os.path.join(args.model_dir, args.model_name + args.output_suffix)

    train_path = os.path.join(args.data_dir, args.train_file)
    dev_path = os.path.join(args.data_dir, args.dev_file)
    corpus_path = os.path.join(args.data_dir, args.corpus_file)

    logging.info(f"Loading base model from: {base_model_path}")
    model = SentenceTransformer(base_model_path)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_seq_length

    train_rows = load_jsonl(train_path)
    dev_rows = load_jsonl(dev_path)
    corpus_rows = load_jsonl(corpus_path)
    noise_rows = load_jsonl(args.noise_corpus_path)

    logging.info(f"Loaded train rows: {len(train_rows)}")
    logging.info(f"Loaded dev rows: {len(dev_rows)}")
    logging.info(f"Loaded corpus rows: {len(corpus_rows)}")
    logging.info(f"Loaded noise rows: {len(noise_rows)}")

    negative_pool = build_negative_pool(noise_rows)
    train_examples = build_train_examples(
        train_rows=train_rows,
        negative_pool=negative_pool,
        negatives_per_positive=args.negatives_per_positive,
        loss_type=args.loss_type,
        seed=args.seed,
    )
    logging.info(f"Built train examples: {len(train_examples)}")
    logging.info(f"Loss type: {args.loss_type}")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )

    # MNRL consumes query-positive-negative groups; triplet loss consumes triples.
    if args.loss_type == "triplet":
        train_loss = losses.TripletLoss(model=model)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

    evaluator = build_ir_evaluator(dev_rows=dev_rows, corpus_rows=corpus_rows)
    warmup_steps = math.ceil(len(train_dataloader) * args.epochs * 0.1)
    logging.info(f"Warmup steps: {warmup_steps}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        output_path=output_path,
        evaluation_steps=max(100, len(train_dataloader) // 2),
        use_amp=torch.cuda.is_available(),
    )

    logging.info(f"Saved random-negative retrieval model to: {output_path}")


if __name__ == "__main__":
    main()

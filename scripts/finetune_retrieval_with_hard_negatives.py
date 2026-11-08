import argparse
import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator


logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# Fine-tune a retriever with mined hard negatives.
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
    return normalize_text(row.get("text") or row.get("positive") or row.get("abstract") or row.get("title"))


def build_train_examples(
    rows: List[Dict[str, Any]],
    loss_type: str,
    hard_negatives_per_positive: int,
) -> List[InputExample]:
    # Convert each mined row into pair or triplet examples for the selected loss.
    examples = []

    for row in rows:
        query = normalize_text(row.get("query") or row.get("query_text"))
        positive = normalize_text(row.get("positive") or row.get("text"))
        hard_negatives = row.get("hard_negatives") or []

        if not query or not positive or not hard_negatives:
            continue

        negatives = []
        for neg in hard_negatives[:hard_negatives_per_positive]:
            if isinstance(neg, dict):
                neg_text = normalize_text(neg.get("text"))
            else:
                neg_text = normalize_text(neg)
            if neg_text:
                negatives.append(neg_text)

        if not negatives:
            continue

        if loss_type == "triplet":
            for negative in negatives:
                examples.append(InputExample(texts=[query, positive, negative]))
        else:
            examples.append(InputExample(texts=[query, positive] + negatives))

    return examples


def build_ir_evaluator(dev_rows: List[Dict[str, Any]], corpus_rows: List[Dict[str, Any]]) -> InformationRetrievalEvaluator:
    # Reuse the clean dev split to monitor retrieval quality while training.
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
        if query and qid and doc_id and doc_id in corpus:
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
        name="retrieval-dev-hard-negatives",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment D: retrieval fine-tuning with mined hard negatives.")
    parser.add_argument("--model_name", type=str, default="scibert-nli")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_hardneg_file", type=str, default="retrieval_train_hardneg.jsonl")
    parser.add_argument("--dev_file", type=str, default="retrieval_dev.jsonl")
    parser.add_argument("--corpus_file", type=str, default="corpus_with_ids.jsonl")
    parser.add_argument("--output_suffix", type=str, default="_retrieval_hardneg")
    parser.add_argument("--loss_type", type=str, default="mnrl", choices=["mnrl", "triplet"])
    parser.add_argument("--hard_negatives_per_positive", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_model_path = os.path.join(args.model_dir, args.model_name)
    output_path = os.path.join(args.model_dir, args.model_name + args.output_suffix)
    train_path = os.path.join(args.data_dir, args.train_hardneg_file)
    dev_path = os.path.join(args.data_dir, args.dev_file)
    corpus_path = os.path.join(args.data_dir, args.corpus_file)

    logging.info(f"Loading base model from: {base_model_path}")
    model = SentenceTransformer(base_model_path)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_seq_length

    train_rows = load_jsonl(train_path)
    dev_rows = load_jsonl(dev_path)
    corpus_rows = load_jsonl(corpus_path)

    logging.info(f"Loaded hard-negative train rows: {len(train_rows)}")
    logging.info(f"Loaded dev rows: {len(dev_rows)}")
    logging.info(f"Loaded corpus rows: {len(corpus_rows)}")

    train_examples = build_train_examples(
        rows=train_rows,
        loss_type=args.loss_type,
        hard_negatives_per_positive=args.hard_negatives_per_positive,
    )
    logging.info(f"Built train examples: {len(train_examples)}")
    logging.info(f"Loss type: {args.loss_type}")

    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True,
    )

    # Select the loss that matches the example shape built above.
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

    logging.info(f"Saved hard-negative retrieval model to: {output_path}")


if __name__ == "__main__":
    main()

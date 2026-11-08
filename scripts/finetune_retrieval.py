import os
import json
import math
import argparse
import logging
from collections import defaultdict
from typing import List, Dict

import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO
)


# Fine-tune a base NLI encoder as a dense retrieval model.
def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="scibert-nli")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="retrieval_train.jsonl")
    parser.add_argument("--dev_file", type=str, default="retrieval_dev.jsonl")
    parser.add_argument("--corpus_file", type=str, default="corpus_with_ids.jsonl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_suffix", type=str, default="_retrieval")
    return parser.parse_args()

def build_train_examples(train_rows: List[Dict]) -> List[InputExample]:
    # MultipleNegativesRankingLoss treats other positives in the batch as negatives.
    examples = []
    for row in train_rows:
        query = row["query"].strip()
        positive = row["positive"].strip()
        if not query or not positive:
            continue
        examples.append(InputExample(texts=[query, positive]))
    return examples

def build_ir_evaluator(dev_rows: List[Dict], corpus_rows: List[Dict]) -> InformationRetrievalEvaluator:
    # Build the SentenceTransformers IR evaluator from dev queries and corpus docs.
    queries = {}
    relevant_docs = defaultdict(set)
    corpus = {}

    for row in corpus_rows:
        doc_id = str(row["doc_id"]).strip()
        text = (row.get("text") or "").strip()
        if doc_id and text:
            corpus[doc_id] = text

    for i, row in enumerate(dev_rows):
        query = (row.get("query") or "").strip()
        qid = str(row.get("question_id") or query or f"q_{i}").strip()
        doc_id = str(row.get("doc_id") or "").strip()

        if not query or not doc_id:
            continue
        if doc_id not in corpus:
            continue

        if qid in queries and queries[qid] != query:
            logging.warning(
                f"Question id {qid} has inconsistent query text. Keeping the first version."
            )
        else:
            queries[qid] = query

        relevant_docs[qid].add(doc_id)
    logging.info(f"Dev queries used for evaluator: {len(queries)}")
    logging.info(f"Dev query-paper pairs used for evaluator: {sum(len(v) for v in relevant_docs.values())}")
    logging.info(f"Corpus size used for evaluator: {len(corpus)}")
    if relevant_docs:
        avg_relevant = sum(len(v) for v in relevant_docs.values()) / len(relevant_docs)
        logging.info(f"Average relevant docs per dev query: {avg_relevant:.2f}")

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=dict(relevant_docs),
        show_progress_bar=True,
        precision_recall_at_k=[1, 3, 5, 10],
        map_at_k=[10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        name="retrieval-dev"
    )
    return evaluator

def main():
    args = get_args()

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

    logging.info(f"Loaded train rows: {len(train_rows)}")
    logging.info(f"Loaded dev rows: {len(dev_rows)}")
    logging.info(f"Loaded corpus rows: {len(corpus_rows)}")

    train_examples = build_train_examples(train_rows)
    logging.info(f"Built train examples: {len(train_examples)}")

    # Drop the last partial batch so in-batch negatives stay well formed.
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size,
        drop_last=True
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Evaluate periodically on held-out retrieval queries during fine-tuning.
    evaluator = build_ir_evaluator(dev_rows, corpus_rows)

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
        use_amp=torch.cuda.is_available()
    )

    logging.info(f"Saved retrieval-tuned model to: {output_path}")

if __name__ == "__main__":
    main()

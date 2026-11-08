import os
import json
import random
import argparse
from collections import defaultdict
from typing import Dict, List


# Convert paired question-paper rows into retrieval training data.
def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def save_jsonl(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_doc_text(item: Dict, doc_mode: str) -> str:
    # Keep document text formatting consistent with evaluation scripts.
    title = (item.get("title") or "").strip()
    abstract = (item.get("abstract") or "").strip()

    if doc_mode == "abstract":
        return abstract
    if doc_mode == "title":
        return title
    if doc_mode == "title_abstract":
        if title and abstract:
            return f"Title: {title} [SEP] Abstract: {abstract}"
        return title or abstract

    raise ValueError(f"Unsupported doc_mode: {doc_mode}")

def get_query_group_id(item: Dict) -> str:
    question_id = str(item.get("question_id") or "").strip()
    if question_id:
        return question_id
    return item["question_text"]

def split_by_question(rows: List[Dict], dev_ratio: float) -> tuple[List[Dict], List[Dict]]:
    # Split whole questions together to avoid train/dev leakage.
    grouped = defaultdict(list)
    for item in rows:
        grouped[get_query_group_id(item)].append(item)

    group_ids = list(grouped.keys())
    random.shuffle(group_ids)

    dev_group_count = int(len(group_ids) * dev_ratio)
    if dev_ratio > 0 and dev_group_count == 0 and len(group_ids) > 1:
        dev_group_count = 1

    dev_group_ids = set(group_ids[:dev_group_count])

    train_rows = []
    dev_rows = []
    for group_id in group_ids:
        if group_id in dev_group_ids:
            dev_rows.extend(grouped[group_id])
        else:
            train_rows.extend(grouped[group_id])

    return train_rows, dev_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--doc_mode", type=str, default="title_abstract",
                        choices=["abstract", "title", "title_abstract"])
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 <= args.dev_ratio < 1:
        raise ValueError("--dev_ratio must be greater than or equal to 0 and less than 1")

    random.seed(args.seed)

    rows = load_jsonl(args.input_path)
    print(f"Loaded {len(rows)} raw rows")

    # Drop rows that cannot form a complete query-positive pair.
    filtered = []
    for item in rows:
        question = (item.get("question_text") or "").strip()
        pmid = str(item.get("pmid") or "").strip()
        doc_text = build_doc_text(item, args.doc_mode).strip()

        if not question or not pmid or not doc_text:
            continue

        filtered.append({
            "question_id": item.get("question_id"),
            "question_text": question,
            "question_type": item.get("question_type"),
            "pmid": pmid,
            "doc_text": doc_text,
            "title": item.get("title", ""),
            "abstract": item.get("abstract", "")
        })

    print(f"Kept {len(filtered)} valid rows")

    # Build a unique retrieval corpus keyed by PMID.
    corpus_map = {}
    for item in filtered:
        pmid = item["pmid"]
        if pmid not in corpus_map:
            corpus_map[pmid] = {
                "doc_id": pmid,
                "pmid": pmid,
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "text": item["doc_text"]
            }

    corpus_rows = list(corpus_map.values())
    print(f"Unique corpus docs: {len(corpus_rows)}")

    # Split by question so the same query never appears in both train and dev.
    train_rows, dev_rows = split_by_question(filtered, args.dev_ratio)

    print(f"Train rows: {len(train_rows)}")
    print(f"Dev rows: {len(dev_rows)}")
    print(f"Train questions: {len({get_query_group_id(x) for x in train_rows})}")
    print(f"Dev questions: {len({get_query_group_id(x) for x in dev_rows})}")

    train_out = []
    for item in train_rows:
        train_out.append({
            "query": item["question_text"],
            "doc_id": item["pmid"],
            "positive": item["doc_text"],
            "question_type": item.get("question_type"),
            "question_id": item.get("question_id")
        })

    dev_out = []
    for item in dev_rows:
        dev_out.append({
            "query": item["question_text"],
            "doc_id": item["pmid"],
            "positive": item["doc_text"],
            "question_type": item.get("question_type"),
            "question_id": item.get("question_id")
        })

    os.makedirs(args.output_dir, exist_ok=True)

    save_jsonl(train_out, os.path.join(args.output_dir, "retrieval_train.jsonl"))
    save_jsonl(dev_out, os.path.join(args.output_dir, "retrieval_dev.jsonl"))
    save_jsonl(corpus_rows, os.path.join(args.output_dir, "corpus_with_ids.jsonl"))

    print("Saved:")
    print(f"- {os.path.join(args.output_dir, 'retrieval_train.jsonl')}")
    print(f"- {os.path.join(args.output_dir, 'retrieval_dev.jsonl')}")
    print(f"- {os.path.join(args.output_dir, 'corpus_with_ids.jsonl')}")

if __name__ == "__main__":
    main()

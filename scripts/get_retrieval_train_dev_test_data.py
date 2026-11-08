import os
import json
import random
import argparse
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


# Build train/dev/test retrieval splits without question leakage.
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
    # Match document text construction across train, dev, and test.
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


def build_retrieval_row(item: Dict) -> Dict:
    return {
        "query": item["question_text"],
        "doc_id": item["pmid"],
        "positive": item["doc_text"],
        "question_type": item.get("question_type"),
        "question_id": item.get("question_id"),
    }


def split_by_question(
    rows: List[Dict],
    dev_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    # Assign question groups, not individual rows, to each split.
    grouped = defaultdict(list)
    for item in rows:
        grouped[get_query_group_id(item)].append(item)

    group_ids = list(grouped.keys())
    random.shuffle(group_ids)

    total_groups = len(group_ids)
    test_group_count = int(total_groups * test_ratio)
    dev_group_count = int(total_groups * dev_ratio)

    if test_ratio > 0 and test_group_count == 0 and total_groups > 2:
        test_group_count = 1
    if dev_ratio > 0 and dev_group_count == 0 and total_groups - test_group_count > 1:
        dev_group_count = 1

    test_group_ids = set(group_ids[:test_group_count])
    dev_group_ids = set(group_ids[test_group_count:test_group_count + dev_group_count])

    train_rows = []
    dev_rows = []
    test_rows = []

    for group_id in group_ids:
        if group_id in test_group_ids:
            test_rows.extend(grouped[group_id])
        elif group_id in dev_group_ids:
            dev_rows.extend(grouped[group_id])
        else:
            train_rows.extend(grouped[group_id])

    return train_rows, dev_rows, test_rows


def print_split_stats(name: str, rows: List[Dict]) -> None:
    question_ids = {get_query_group_id(x) for x in rows}
    doc_ids = {x["pmid"] for x in rows}
    type_counts = Counter((x.get("question_type") or "unknown") for x in rows)

    print(f"{name} rows: {len(rows)}")
    print(f"{name} questions: {len(question_ids)}")
    print(f"{name} positive docs: {len(doc_ids)}")
    print(f"{name} question types: {dict(type_counts)}")


def assert_no_question_overlap(
    train_rows: List[Dict],
    dev_rows: List[Dict],
    test_rows: List[Dict],
) -> None:
    # Fail fast if the split accidentally leaks a query across partitions.
    train_questions = {get_query_group_id(x) for x in train_rows}
    dev_questions = {get_query_group_id(x) for x in dev_rows}
    test_questions = {get_query_group_id(x) for x in test_rows}

    if train_questions & dev_questions:
        raise ValueError("Question leakage found between train and dev.")
    if train_questions & test_questions:
        raise ValueError("Question leakage found between train and test.")
    if dev_questions & test_questions:
        raise ValueError("Question leakage found between dev and test.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument(
        "--doc_mode",
        type=str,
        default="title_abstract",
        choices=["abstract", "title", "title_abstract"],
    )
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 <= args.dev_ratio < 1:
        raise ValueError("--dev_ratio must be greater than or equal to 0 and less than 1")
    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test_ratio must be greater than or equal to 0 and less than 1")
    if args.dev_ratio + args.test_ratio >= 1:
        raise ValueError("--dev_ratio + --test_ratio must be less than 1")

    random.seed(args.seed)

    rows = load_jsonl(args.input_path)
    print(f"Loaded {len(rows)} raw rows")

    # Keep only rows with the fields needed for retrieval training/evaluation.
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
            "abstract": item.get("abstract", ""),
        })

    print(f"Kept {len(filtered)} valid rows")

    # Store one corpus entry per PMID while preserving all query-positive pairs.
    corpus_map = {}
    for item in filtered:
        pmid = item["pmid"]
        if pmid not in corpus_map:
            corpus_map[pmid] = {
                "doc_id": pmid,
                "pmid": pmid,
                "title": item.get("title", ""),
                "abstract": item.get("abstract", ""),
                "text": item["doc_text"],
            }

    corpus_rows = list(corpus_map.values())
    print(f"Unique corpus docs: {len(corpus_rows)}")

    train_rows, dev_rows, test_rows = split_by_question(
        filtered,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
    )
    assert_no_question_overlap(train_rows, dev_rows, test_rows)

    print_split_stats("Train", train_rows)
    print_split_stats("Dev", dev_rows)
    print_split_stats("Test", test_rows)

    train_out = [build_retrieval_row(item) for item in train_rows]
    dev_out = [build_retrieval_row(item) for item in dev_rows]
    test_out = [build_retrieval_row(item) for item in test_rows]

    os.makedirs(args.output_dir, exist_ok=True)

    train_path = os.path.join(args.output_dir, "retrieval_train.jsonl")
    dev_path = os.path.join(args.output_dir, "retrieval_dev.jsonl")
    test_path = os.path.join(args.output_dir, "retrieval_test.jsonl")
    corpus_path = os.path.join(args.output_dir, "corpus_with_ids.jsonl")

    save_jsonl(train_out, train_path)
    save_jsonl(dev_out, dev_path)
    save_jsonl(test_out, test_path)
    save_jsonl(corpus_rows, corpus_path)

    print("Saved:")
    print(f"- {train_path}")
    print(f"- {dev_path}")
    print(f"- {test_path}")
    print(f"- {corpus_path}")


if __name__ == "__main__":
    main()

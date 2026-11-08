import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Set


# Build random distractor documents from BioASQ snippet text.
PUBMED_RE = re.compile(r"pubmed/(\d+)")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_pmid(value: Any) -> str:
    # Accept both PubMed URLs and plain numeric PMID strings.
    text = normalize_text(value)
    if not text:
        return ""
    match = PUBMED_RE.search(text)
    if match:
        return match.group(1)
    if text.isdigit():
        return text
    return ""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not path:
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_base_doc_ids(corpus_path: str) -> Set[str]:
    # Existing corpus documents should not be reused as noise.
    doc_ids = set()
    for row in load_jsonl(corpus_path):
        doc_id = normalize_text(row.get("doc_id") or row.get("pmid"))
        if doc_id:
            doc_ids.add(doc_id)
    return doc_ids


def collect_relevant_doc_ids(*paths: str) -> Set[str]:
    # Gold documents are excluded so distractors do not become false negatives.
    relevant = set()
    for path in paths:
        for row in load_jsonl(path):
            if "relevant_doc_ids" in row:
                for doc_id in row.get("relevant_doc_ids") or []:
                    doc_id = normalize_text(doc_id)
                    if doc_id:
                        relevant.add(doc_id)
            else:
                doc_id = normalize_text(row.get("doc_id") or row.get("pmid"))
                if doc_id:
                    relevant.add(doc_id)
    return relevant


def load_training13b_noise_docs(path: str, min_chars: int, max_snippets_per_doc: int) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group snippets by PMID and merge a few snippets into each candidate document.
    snippets_by_pmid: Dict[str, List[str]] = defaultdict(list)

    for question in data.get("questions", []):
        for snippet in question.get("snippets") or []:
            pmid = extract_pmid(snippet.get("document"))
            text = normalize_text(snippet.get("text"))
            if not pmid or len(text) < min_chars:
                continue
            if len(snippets_by_pmid[pmid]) < max_snippets_per_doc:
                snippets_by_pmid[pmid].append(text)

    docs = {}
    for pmid, snippets in snippets_by_pmid.items():
        merged = " ".join(snippets).strip()
        if len(merged) < min_chars:
            continue
        docs[pmid] = {
            "doc_id": pmid,
            "pmid": pmid,
            "title": "",
            "abstract": merged,
            "text": merged,
            "source": "training13b_snippets",
        }
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small noise/distractor corpus from BioASQ training13b snippets. "
            "This does not fetch PubMed; it only uses snippet text already present in training13b.json."
        )
    )
    parser.add_argument("--training13b_path", type=str, default="data/training13b.json")
    parser.add_argument("--base_corpus_path", type=str, default="data/corpus_with_ids.jsonl")
    parser.add_argument("--exclude_eval_path", type=str, default="data/retrieval_test.jsonl")
    parser.add_argument("--exclude_train_path", type=str, default="")
    parser.add_argument("--output_noise_path", type=str, default="data/noise_corpus_1000.jsonl")
    parser.add_argument("--output_augmented_corpus_path", type=str, default="")
    parser.add_argument("--num_noise_docs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--max_snippets_per_doc", type=int, default=3)
    args = parser.parse_args()

    random.seed(args.seed)

    base_rows = load_jsonl(args.base_corpus_path) if args.base_corpus_path else []
    base_doc_ids = collect_base_doc_ids(args.base_corpus_path) if args.base_corpus_path else set()
    relevant_doc_ids = collect_relevant_doc_ids(args.exclude_eval_path, args.exclude_train_path)
    excluded = base_doc_ids | relevant_doc_ids

    noise_docs = load_training13b_noise_docs(
        path=args.training13b_path,
        min_chars=args.min_chars,
        max_snippets_per_doc=args.max_snippets_per_doc,
    )

    # Sample only documents that are outside the base and relevant-document sets.
    candidates = [doc for pmid, doc in noise_docs.items() if pmid not in excluded]
    random.shuffle(candidates)
    sampled = candidates[: args.num_noise_docs]

    save_jsonl(sampled, args.output_noise_path)

    print(f"Loaded base corpus docs: {len(base_rows)}")
    print(f"Excluded doc ids: {len(excluded)}")
    print(f"Candidate noise docs from training13b snippets: {len(candidates)}")
    print(f"Sampled noise docs: {len(sampled)}")
    print(f"Saved noise corpus: {args.output_noise_path}")

    if args.output_augmented_corpus_path:
        augmented = base_rows + sampled
        save_jsonl(augmented, args.output_augmented_corpus_path)
        print(f"Saved augmented corpus: {args.output_augmented_corpus_path}")
        print(f"Augmented corpus docs: {len(augmented)}")


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import sys


# Orchestrate Experiment D from noise generation through final evaluation.
def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment D: mine hard negatives, train with them, then evaluate on base corpus + noise docs."
    )
    parser.add_argument("--base_model_name", type=str, default="scibert-nli")
    parser.add_argument("--mining_model_name", type=str, default="scibert-nli_retrieval")
    parser.add_argument("--hardneg_model_name", type=str, default="scibert-nli_retrieval_hardneg")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="retrieval_train.jsonl")
    parser.add_argument("--dev_file", type=str, default="retrieval_dev.jsonl")
    parser.add_argument("--test_file", type=str, default="retrieval_test.jsonl")
    parser.add_argument("--corpus_file", type=str, default="corpus_with_ids.jsonl")
    parser.add_argument("--training13b_path", type=str, default="data/training13b.json")
    parser.add_argument("--noise_docs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--hard_negatives_per_query", type=int, default=3)
    parser.add_argument("--hard_negatives_per_positive", type=int, default=1)
    parser.add_argument("--loss_type", type=str, default="mnrl", choices=["mnrl", "triplet"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output_dir", type=str, default="result/resultv4")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache_d")
    os.makedirs(cache_dir, exist_ok=True)

    base_corpus_path = os.path.join(args.data_dir, args.corpus_file)
    train_path = os.path.join(args.data_dir, args.train_file)
    dev_path = os.path.join(args.data_dir, args.dev_file)
    test_path = os.path.join(args.data_dir, args.test_file)
    noise_path = os.path.join(args.output_dir, f"noise_corpus_{args.noise_docs}_seed{args.seed}.jsonl")
    augmented_path = os.path.join(args.output_dir, f"augmented_corpus_noise{args.noise_docs}_seed{args.seed}.jsonl")
    hardneg_train_path = os.path.join(args.data_dir, f"retrieval_train_hardneg_noise{args.noise_docs}.jsonl")

    # Build the noisy corpus used both for mining and evaluation.
    run([
        sys.executable,
        "scriptsv4/build_noise_corpus.py",
        "--training13b_path",
        args.training13b_path,
        "--base_corpus_path",
        base_corpus_path,
        "--exclude_eval_path",
        test_path,
        "--exclude_train_path",
        train_path,
        "--output_noise_path",
        noise_path,
        "--output_augmented_corpus_path",
        augmented_path,
        "--num_noise_docs",
        str(args.noise_docs),
        "--seed",
        str(args.seed),
    ])

    # Mine difficult non-gold documents with the current retrieval model.
    run([
        sys.executable,
        "scriptsv4/mine_hard_negatives.py",
        "--model_name",
        args.mining_model_name,
        "--model_dir",
        args.model_dir,
        "--train_path",
        train_path,
        "--corpus_path",
        augmented_path,
        "--output_path",
        hardneg_train_path,
        "--top_k",
        str(args.top_k),
        "--hard_negatives_per_query",
        str(args.hard_negatives_per_query),
        "--batch_size",
        str(args.eval_batch_size),
    ])

    # Train a new retriever using the mined hard negatives.
    run([
        sys.executable,
        "scriptsv4/finetune_retrieval_with_hard_negatives.py",
        "--model_name",
        args.base_model_name,
        "--model_dir",
        args.model_dir,
        "--data_dir",
        args.data_dir,
        "--train_hardneg_file",
        os.path.basename(hardneg_train_path),
        "--dev_file",
        args.dev_file,
        "--corpus_file",
        args.corpus_file,
        "--output_suffix",
        "_retrieval_hardneg",
        "--loss_type",
        args.loss_type,
        "--hard_negatives_per_positive",
        str(args.hard_negatives_per_positive),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
    ])

    # Evaluate the hard-negative model on the same noisy corpus.
    run([
        sys.executable,
        "scriptsv3/evaluate_retrival.py",
        "--model_name",
        args.hardneg_model_name,
        "--model_dir",
        args.model_dir,
        "--corpus_path",
        augmented_path,
        "--eval_queries_path",
        test_path,
        "--output_path",
        os.path.join(args.output_dir, f"{args.hardneg_model_name}-experiment-d-noise{args.noise_docs}.json"),
        "--doc_mode",
        "title_abstract",
        "--batch_size",
        str(args.eval_batch_size),
        "--cache_dir",
        cache_dir,
    ])


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import sys


# Orchestrate Experiment E: hard-negative retrieval plus lexical reranking.
def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment E: evaluate the hard-negative model with second-stage lexical reranking."
    )
    parser.add_argument("--model_name", type=str, default="scibert-nli_retrieval_hardneg")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--training13b_path", type=str, default="data/training13b.json")
    parser.add_argument("--base_corpus_path", type=str, default="data/corpus_with_ids.jsonl")
    parser.add_argument("--eval_queries_path", type=str, default="data/retrieval_test.jsonl")
    parser.add_argument("--noise_docs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rerank_pool_size", type=int, default=50)
    parser.add_argument("--rerank_alpha", type=float, default=0.8)
    parser.add_argument("--output_dir", type=str, default="result/resultv4")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "cache_e")
    os.makedirs(cache_dir, exist_ok=True)

    noise_path = os.path.join(args.output_dir, f"noise_corpus_{args.noise_docs}_seed{args.seed}.jsonl")
    augmented_path = os.path.join(args.output_dir, f"augmented_corpus_noise{args.noise_docs}_seed{args.seed}.jsonl")
    output_path = os.path.join(args.output_dir, f"{args.model_name}-experiment-e-rerank-noise{args.noise_docs}.json")

    # Reuse the noisy corpus if Experiment B/D already created it.
    if not os.path.exists(augmented_path):
        run([
            sys.executable,
            "scriptsv4/build_noise_corpus.py",
            "--training13b_path",
            args.training13b_path,
            "--base_corpus_path",
            args.base_corpus_path,
            "--exclude_eval_path",
            args.eval_queries_path,
            "--output_noise_path",
            noise_path,
            "--output_augmented_corpus_path",
            augmented_path,
            "--num_noise_docs",
            str(args.noise_docs),
            "--seed",
            str(args.seed),
        ])

    # Evaluate with the reranking options enabled for a second-stage ranking pass.
    run([
        sys.executable,
        "scriptsv3/evaluate_retrival.py",
        "--model_name",
        args.model_name,
        "--model_dir",
        args.model_dir,
        "--corpus_path",
        augmented_path,
        "--eval_queries_path",
        args.eval_queries_path,
        "--output_path",
        output_path,
        "--doc_mode",
        "title_abstract",
        "--batch_size",
        str(args.batch_size),
        "--cache_dir",
        cache_dir,
        "--rerank",
        "--rerank_pool_size",
        str(args.rerank_pool_size),
        "--rerank_alpha",
        str(args.rerank_alpha),
    ])

    print(f"\nExperiment E result saved to: {output_path}")


if __name__ == "__main__":
    main()

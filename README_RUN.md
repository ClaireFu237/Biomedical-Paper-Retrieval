# Biomedical Retrieval Experiments: Run Guide

This project implements a biomedical question-to-paper retrieval pipeline.
Given a biomedical question, the system ranks candidate PubMed-style papers and
evaluates whether the relevant papers appear near the top of the retrieved list.

The code uses SentenceTransformers with SciBERT/BioBERT checkpoints, retrieval
fine-tuning, noise-document robustness tests, random negatives, hard negatives,
and an optional lexical reranking step.

## 1. Project Layout

After unzipping the package, run all commands from the `final_version` directory:

```bash
cd final_version
```

Expected layout:

```text
final_version/
  data/
    question_abstract_pairs.jsonl
    training13b.json
    abstracts_only.jsonl
    questions_only.jsonl
  models/              # created after downloading/training models
  result/
  scriptsv/
    README_RUN.md
    get_retrieval_train_dev_test_data.py
    download_model.py
    finetune_retrieval.py
    evaluate_retrival.py
    build_noise_corpus.py
    finetune_retrieval_with_random_negatives.py
    mine_hard_negatives.py
    finetune_retrieval_with_hard_negatives.py
    run_experiment_b.py
    run_experiment_d.py
    run_experiment_e.py
```

Important: the actual script folder is `scriptsv`, not `scriptsv4`.

## 2. Environment Setup

Install the required Python packages:

```bash
pip install -U sentence-transformers transformers torch tqdm prettytable
```

`prettytable` is only needed for `interactive_search.py`. Training and
evaluation will run on CPU, but CUDA is used automatically when available.

## 3. Prepare Train / Dev / Test Retrieval Data

Input file:

```text
data/question_abstract_pairs.jsonl
```

Run:

```bash
python scriptsv/get_retrieval_train_dev_test_data.py \
  --input_path data/question_abstract_pairs.jsonl \
  --output_dir data \
  --doc_mode title_abstract \
  --dev_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 42
```

This creates:

```text
data/retrieval_train.jsonl
data/retrieval_dev.jsonl
data/retrieval_test.jsonl
data/corpus_with_ids.jsonl
```

The split is grouped by `question_id`, so the same question does not appear in
train, dev, and test at the same time.

## 4. Download the Base Model

Download SciBERT-NLI:

```bash
python scriptsv/download_model.py \
  --model scibert-nli \
  --output_dir models
```

This saves the model to:

```text
models/scibert-nli
```

The code also supports:

```bash
python scriptsv/download_model.py --model biobert-nli --output_dir models
```

## 5. Train the Baseline Retriever

Train a dense retriever using positive question-paper pairs:

```bash
python scriptsv/finetune_retrieval.py \
  --model_name scibert-nli \
  --model_dir models \
  --data_dir data \
  --train_file retrieval_train.jsonl \
  --dev_file retrieval_dev.jsonl \
  --corpus_file corpus_with_ids.jsonl \
  --batch_size 32 \
  --epochs 3 \
  --max_seq_length 256 \
  --lr 2e-5
```

This saves:

```text
models/scibert-nli_retrieval
```

The baseline uses `MultipleNegativesRankingLoss`, where other positives in the
same batch act as in-batch negatives.

## 6. Evaluate a Retriever

Evaluate the baseline on the clean test corpus:

```bash
python scriptsv/evaluate_retrival.py \
  --model_name scibert-nli_retrieval \
  --model_dir models \
  --corpus_path data/corpus_with_ids.jsonl \
  --eval_queries_path data/retrieval_test.jsonl \
  --output_path result/scibert-nli_retrieval-baseline-test.json \
  --doc_mode title_abstract \
  --ks 1,3,5,10,20 \
  --batch_size 64 \
  --cache_dir data/cache_eval
```

The evaluator reports `Hit@k`, `Recall@k`, `MRR@k`, and `nDCG@k`. It also saves
per-query retrieval results and per-question-type metrics.

Note: the file name `evaluate_retrival.py` contains a spelling typo. Use the
filename exactly as it appears in the folder.

## 7. Experiment B: Baseline With Noise Documents

Experiment B tests whether the trained baseline remains robust when irrelevant
biomedical snippets are added to the candidate corpus.

First build the noisy corpus:

```bash
python scriptsv/build_noise_corpus.py \
  --training13b_path data/training13b.json \
  --base_corpus_path data/corpus_with_ids.jsonl \
  --exclude_eval_path data/retrieval_test.jsonl \
  --output_noise_path result/noise_corpus_1000_seed42.jsonl \
  --output_augmented_corpus_path result/augmented_corpus_noise1000_seed42.jsonl \
  --num_noise_docs 1000 \
  --seed 42
```

Then evaluate the baseline on the augmented corpus:

```bash
python scriptsv/evaluate_retrival.py \
  --model_name scibert-nli_retrieval \
  --model_dir models \
  --corpus_path result/augmented_corpus_noise1000_seed42.jsonl \
  --eval_queries_path data/retrieval_test.jsonl \
  --output_path result/scibert-nli_retrieval-experiment-b-noise1000.json \
  --doc_mode title_abstract \
  --ks 1,3,5,10,20 \
  --batch_size 64 \
  --cache_dir data/cache_b
```

The noise documents are sampled from snippets in `training13b.json`. Evaluation
gold documents are excluded from the noise pool.

## 8. Experiment C: Train With Random Negatives

Build a reusable random-noise corpus:

```bash
python scriptsv/build_noise_corpus.py \
  --training13b_path data/training13b.json \
  --base_corpus_path data/corpus_with_ids.jsonl \
  --exclude_eval_path data/retrieval_test.jsonl \
  --exclude_train_path data/retrieval_train.jsonl \
  --output_noise_path data/noise_corpus_1000.jsonl \
  --num_noise_docs 1000 \
  --seed 42
```

Fine-tune with random negative documents:

```bash
python scriptsv/finetune_retrieval_with_random_negatives.py \
  --model_name scibert-nli \
  --model_dir models \
  --data_dir data \
  --train_file retrieval_train.jsonl \
  --dev_file retrieval_dev.jsonl \
  --corpus_file corpus_with_ids.jsonl \
  --noise_corpus_path data/noise_corpus_1000.jsonl \
  --output_suffix _retrieval_randomneg \
  --loss_type mnrl \
  --negatives_per_positive 1 \
  --batch_size 16 \
  --epochs 1 \
  --max_seq_length 256 \
  --lr 2e-5
```

This saves:

```text
models/scibert-nli_retrieval_randomneg
```

Evaluate the random-negative model on the same augmented corpus used in
Experiment B:

```bash
python scriptsv/evaluate_retrival.py \
  --model_name scibert-nli_retrieval_randomneg \
  --model_dir models \
  --corpus_path result/augmented_corpus_noise1000_seed42.jsonl \
  --eval_queries_path data/retrieval_test.jsonl \
  --output_path result/scibert-nli_retrieval_randomneg-experiment-c-noise1000.json \
  --doc_mode title_abstract \
  --ks 1,3,5,10,20 \
  --batch_size 64 \
  --cache_dir data/cache_c
```

## 9. Experiment D: Train With Hard Negatives

Experiment D mines high-scoring non-gold documents, trains a hard-negative
retriever, and evaluates it on the augmented noisy corpus.

Build the augmented corpus if it does not already exist:

```bash
python scriptsv/build_noise_corpus.py \
  --training13b_path data/training13b.json \
  --base_corpus_path data/corpus_with_ids.jsonl \
  --exclude_eval_path data/retrieval_test.jsonl \
  --exclude_train_path data/retrieval_train.jsonl \
  --output_noise_path result/noise_corpus_1000_seed42.jsonl \
  --output_augmented_corpus_path result/augmented_corpus_noise1000_seed42.jsonl \
  --num_noise_docs 1000 \
  --seed 42
```

Mine hard negatives with the baseline retriever:

```bash
python scriptsv/mine_hard_negatives.py \
  --model_name scibert-nli_retrieval \
  --model_dir models \
  --train_path data/retrieval_train.jsonl \
  --corpus_path result/augmented_corpus_noise1000_seed42.jsonl \
  --output_path data/retrieval_train_hardneg_noise1000.jsonl \
  --top_k 50 \
  --hard_negatives_per_query 3 \
  --batch_size 64
```

Train the hard-negative retriever:

```bash
python scriptsv/finetune_retrieval_with_hard_negatives.py \
  --model_name scibert-nli \
  --model_dir models \
  --data_dir data \
  --train_hardneg_file retrieval_train_hardneg_noise1000.jsonl \
  --dev_file retrieval_dev.jsonl \
  --corpus_file corpus_with_ids.jsonl \
  --output_suffix _retrieval_hardneg \
  --loss_type mnrl \
  --hard_negatives_per_positive 1 \
  --batch_size 16 \
  --epochs 1 \
  --max_seq_length 256 \
  --lr 2e-5 \
  --seed 42
```

This saves:

```text
models/scibert-nli_retrieval_hardneg
```

Evaluate the hard-negative retriever:

```bash
python scriptsv/evaluate_retrival.py \
  --model_name scibert-nli_retrieval_hardneg \
  --model_dir models \
  --corpus_path result/augmented_corpus_noise1000_seed42.jsonl \
  --eval_queries_path data/retrieval_test.jsonl \
  --output_path result/scibert-nli_retrieval_hardneg-experiment-d-noise1000.json \
  --doc_mode title_abstract \
  --ks 1,3,5,10,20 \
  --batch_size 64 \
  --cache_dir data/cache_d
```

## 10. Experiment E: Hard Negatives + Reranking

Experiment E evaluates the hard-negative model with a lightweight lexical
reranking step over the top dense candidates.

```bash
python scriptsv/evaluate_retrival.py \
  --model_name scibert-nli_retrieval_hardneg \
  --model_dir models \
  --corpus_path result/augmented_corpus_noise1000_seed42.jsonl \
  --eval_queries_path data/retrieval_test.jsonl \
  --output_path result/scibert-nli_retrieval_hardneg-experiment-e-rerank-noise1000.json \
  --doc_mode title_abstract \
  --ks 1,3,5,10,20 \
  --batch_size 64 \
  --cache_dir data/cache_e \
  --rerank \
  --rerank_pool_size 50 \
  --rerank_alpha 0.8
```

The reranker computes:

```text
final_score = alpha * dense_score + (1 - alpha) * lexical_overlap_score
```

The default setting uses `alpha = 0.8`, so dense retrieval remains the main
signal while lexical overlap adjusts close candidates.

## 11. Optional Interactive Search

After creating `data/corpus_with_ids.jsonl` and downloading/training a model,
you can run:

```bash
python scriptsv/interactive_search.py
```

By default, this script loads:

```text
models/scibert-nli
data/corpus_with_ids.jsonl
```

To search with a fine-tuned model, edit `MODEL_NAME` in `interactive_search.py`
from `scibert-nli` to the desired local model folder, such as
`scibert-nli_retrieval`.

## 12. Recommended Full Run Order

For a clean full run:

```bash
python scriptsv/get_retrieval_train_dev_test_data.py --input_path data/question_abstract_pairs.jsonl --output_dir data
python scriptsv/download_model.py --model scibert-nli --output_dir models
python scriptsv/finetune_retrieval.py --model_name scibert-nli --model_dir models --data_dir data
python scriptsv/build_noise_corpus.py --training13b_path data/training13b.json --base_corpus_path data/corpus_with_ids.jsonl --exclude_eval_path data/retrieval_test.jsonl --exclude_train_path data/retrieval_train.jsonl --output_noise_path result/noise_corpus_1000_seed42.jsonl --output_augmented_corpus_path result/augmented_corpus_noise1000_seed42.jsonl --num_noise_docs 1000
python scriptsv/evaluate_retrival.py --model_name scibert-nli_retrieval --model_dir models --corpus_path result/augmented_corpus_noise1000_seed42.jsonl --eval_queries_path data/retrieval_test.jsonl --output_path result/scibert-nli_retrieval-experiment-b-noise1000.json --cache_dir data/cache_b
python scriptsv/build_noise_corpus.py --training13b_path data/training13b.json --base_corpus_path data/corpus_with_ids.jsonl --exclude_eval_path data/retrieval_test.jsonl --exclude_train_path data/retrieval_train.jsonl --output_noise_path data/noise_corpus_1000.jsonl --num_noise_docs 1000
python scriptsv/finetune_retrieval_with_random_negatives.py --model_name scibert-nli --model_dir models --data_dir data --noise_corpus_path data/noise_corpus_1000.jsonl
python scriptsv/evaluate_retrival.py --model_name scibert-nli_retrieval_randomneg --model_dir models --corpus_path result/augmented_corpus_noise1000_seed42.jsonl --eval_queries_path data/retrieval_test.jsonl --output_path result/scibert-nli_retrieval_randomneg-experiment-c-noise1000.json --cache_dir data/cache_c
python scriptsv/mine_hard_negatives.py --model_name scibert-nli_retrieval --model_dir models --train_path data/retrieval_train.jsonl --corpus_path result/augmented_corpus_noise1000_seed42.jsonl --output_path data/retrieval_train_hardneg_noise1000.jsonl
python scriptsv/finetune_retrieval_with_hard_negatives.py --model_name scibert-nli --model_dir models --data_dir data --train_hardneg_file retrieval_train_hardneg_noise1000.jsonl
python scriptsv/evaluate_retrival.py --model_name scibert-nli_retrieval_hardneg --model_dir models --corpus_path result/augmented_corpus_noise1000_seed42.jsonl --eval_queries_path data/retrieval_test.jsonl --output_path result/scibert-nli_retrieval_hardneg-experiment-d-noise1000.json --cache_dir data/cache_d
python scriptsv/evaluate_retrival.py --model_name scibert-nli_retrieval_hardneg --model_dir models --corpus_path result/augmented_corpus_noise1000_seed42.jsonl --eval_queries_path data/retrieval_test.jsonl --output_path result/scibert-nli_retrieval_hardneg-experiment-e-rerank-noise1000.json --cache_dir data/cache_e --rerank --rerank_pool_size 50 --rerank_alpha 0.8
```

## 13. Notes From Code Review

- Use `scibert-nli_retrieval` as the main baseline model.
- Use `title_abstract` as the default document representation.
- Keep `seed=42` for reproducible splits and noise sampling.
- If CUDA is available, training scripts automatically use mixed precision.
- If `mnrl` training with explicit negatives is unstable in your local
  SentenceTransformers version, try `--loss_type triplet`.
- The wrapper scripts `run_experiment_b.py`, `run_experiment_d.py`, and
  `run_experiment_e.py` still contain legacy hard-coded paths such as
  `scriptsv4/...` and `scriptsv3/...`. Unless those scripts are patched, use the
  explicit commands in this README instead.

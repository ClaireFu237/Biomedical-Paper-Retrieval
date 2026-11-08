import os
import json
import pickle
import textwrap
import prettytable

import torch
from sentence_transformers import SentenceTransformer, util

# Interactive dense retrieval search over the prepared corpus.
DATA_PATH = 'data'
MODEL_NAME = 'scibert-nli'
DOC_MODE = 'title_abstract'

SOURCE_CORPUS_PATH = os.path.join(DATA_PATH, 'corpus_with_ids.jsonl')
CACHE_PREFIX = f'{MODEL_NAME}-{DOC_MODE}'
CORPUS_PATH = os.path.join(DATA_PATH, f'{CACHE_PREFIX}-corpus.pkl')
EMBEDDINGS_PATH = os.path.join(DATA_PATH, f'{CACHE_PREFIX}-embeddings.pkl')
MODEL_PATH = os.path.join('models', MODEL_NAME)

def load_bioasq_jsonl(path):
    # Load corpus text and fall back to title/abstract when text is absent.
    corpus = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            title = (item.get("title") or "").strip()
            abstract = (item.get("abstract") or "").strip()
            text = (item.get("text") or "").strip()

            if not text and title and abstract:
                text = f"Title: {title} [SEP] Abstract: {abstract}"
            elif not text:
                text = title or abstract

            if text:
                corpus.append(text)

    print("Corpus size:", len(corpus))
    return corpus

def ask_question(query, model, corpus, corpus_embeddings, top_k=5):
    # Encode the user query and retrieve the nearest corpus embeddings.
    if not corpus:
        return []

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    corpus_embeddings = corpus_embeddings.to(query_embedding.device)

    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    top_k = min(top_k, len(corpus))
    top_results = torch.topk(cos_scores, k=top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
        results.append([
            rank + 1,
            corpus[idx],
            float(score)
        ])

    return results

def show_answers(results):
    # Print retrieved documents in a compact terminal table.
    table = prettytable.PrettyTable(['Rank', 'Document', 'Score'])

    for rank, text, score in results:
        text = textwrap.fill(text, width=75)
        table.add_row([rank, text, round(score, 4)])

    print("\n", table, "\n")

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    # Cache parsed corpus text to avoid rebuilding it each session.
    if not os.path.exists(CORPUS_PATH):
        print("Loading corpus...")
        corpus = load_bioasq_jsonl(SOURCE_CORPUS_PATH)

        with open(CORPUS_PATH, 'wb') as f:
            pickle.dump(corpus, f)
    else:
        print(" Loading cached corpus...")
        with open(CORPUS_PATH, 'rb') as f:
            corpus = pickle.load(f)

    if not corpus:
        raise ValueError(f"No documents loaded from {SOURCE_CORPUS_PATH}")

    # Use GPU when available for faster embedding search.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = SentenceTransformer(MODEL_PATH, device=device)
    
    # Cache corpus embeddings because they are reused for every question.
    if not os.path.exists(EMBEDDINGS_PATH):
        print("Computing embeddings...")

        corpus_embeddings = model.encode(
            corpus,
            batch_size=64,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump(corpus_embeddings, f)
    else:
        print(" Loading cached embeddings...")
        with open(EMBEDDINGS_PATH, 'rb') as f:
            corpus_embeddings = pickle.load(f)

    # Simple REPL for ad hoc retrieval checks.
    while True:
        query = input("\n Ask your question (type 'exit' to quit): ")

        query = query.strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = ask_question(query, model, corpus, corpus_embeddings)

        show_answers(results)

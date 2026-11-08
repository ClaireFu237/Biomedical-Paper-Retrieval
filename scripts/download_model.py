import os
import argparse
from shutil import rmtree
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import models, SentenceTransformer

import os
import argparse
from sentence_transformers import SentenceTransformer

# Map short local names to Hugging Face model IDs.
MODELS = {

    "scibert-nli": "gsarti/scibert-nli",
    "biobert-nli": "gsarti/biobert-nli",
}

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="biobert-nli",
        help=f"Available models: {list(MODELS.keys())}"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="Where to save the model"
    )

    args = parser.parse_args()

    if args.model not in MODELS:
        raise ValueError(f"Model must be one of: {list(MODELS.keys())}")

    save_path = os.path.join(args.output_dir, args.model)
    os.makedirs(save_path, exist_ok=True)

    # Download the selected model once and save it for local scripts.
    print(f" Loading model: {args.model}")

    model = SentenceTransformer(MODELS[args.model])

    print("💾 Saving model...")
    model.save(save_path)

    print(f" Model saved at: {save_path}")

    print("\n Testing embeddings...")

    sentences = [
        "COVID-19 affects the respiratory system.",
        "The virus impacts lungs and breathing."
    ]

    # Run a tiny encode pass to catch model loading issues early.
    embeddings = model.encode(sentences)

if __name__ == "__main__":
    main()

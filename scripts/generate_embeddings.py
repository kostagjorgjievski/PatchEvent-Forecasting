# scripts/generate_embeddings.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

INPUT_FILE = "data/processed/patch_dataset.parquet"
OUTPUT_FILE = "data/processed/patch_dataset_embedded.parquet"

def main():
    print("Loading dataset...")
    df = pd.read_parquet(INPUT_FILE)
    df["event_text"] = df["event_text"].fillna("")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding event_text fields...")
    embeddings = model.encode(df["event_text"].tolist(), show_progress_bar=True)

    print("Adding embeddings to DataFrame...")
    df["event_embedding"] = [e.tolist() for e in embeddings]  # convert from np.array to JSON-safe list

    print(f"Saving to: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)

if __name__ == "__main__":
    main()

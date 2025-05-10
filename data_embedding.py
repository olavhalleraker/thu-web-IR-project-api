from sentence_transformers import SentenceTransformer 
import numpy as np
import json
import torch
import time
from typing import List, Dict

# Configuration
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 128
INPUT_FILE = 'test.json'
OUTPUT_JSON = 'test_metadata.json'
OUTPUT_EMBEDDINGS = 'test_embeddings.npy'
TITLE_WEIGHT = 0.7
SUMMARY_WEIGHT = 0.3

# Load model
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = SentenceTransformer(MODEL_NAME, device=device)

# Load articles
def load_articles(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Extract titles and summaries
def prepare_text_pairs(articles: List[Dict]) -> List[Dict[str, str]]:
    return [
        {
            "title": article.get("title", "") or "",
            "summary": article.get("summary", "") or ""
        }
        for article in articles if article is not None
    ]

# Encode in batches, with weighting
def batch_encode_weighted(pairs: List[Dict[str, str]], batch_size: int) -> np.ndarray:
    embeddings = []

    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i + batch_size]

        titles = [p['title'] if p['title'] else "." for p in batch_pairs]
        summaries = [p['summary'] if p['summary'] else "." for p in batch_pairs]

        title_embs = model.encode(titles, convert_to_numpy=True, normalize_embeddings=True)
        summary_embs = model.encode(summaries, convert_to_numpy=True, normalize_embeddings=True)

        combined = TITLE_WEIGHT * title_embs + SUMMARY_WEIGHT * summary_embs
        embeddings.append(combined)

    return np.vstack(embeddings)

# Save cleaned metadata (without embeddings)
def save_metadata(articles: List[Dict], path: str):
    for article in articles:
        article.pop("embedding", None)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

# Main
if __name__ == "__main__":
    start_time = time.time()

    articles = load_articles(INPUT_FILE)
    pairs = prepare_text_pairs(articles)
    embeddings = batch_encode_weighted(pairs, BATCH_SIZE)

    save_metadata(articles, OUTPUT_JSON)
    np.save(OUTPUT_EMBEDDINGS, embeddings)

    print(f"{len(articles)} articles processed in {time.time() - start_time:.2f} seconds")
    print(f"Metadata saved in '{OUTPUT_JSON}' and embeddings in '{OUTPUT_EMBEDDINGS}'")

from sentence_transformers import SentenceTransformer
import numpy as np
import json
import torch
import time
from typing import List, Dict

# Configuration parameters
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 128
INPUT_FILE = 'test.json'
OUTPUT_JSON = 'test_metadata.json'
OUTPUT_EMBEDDINGS = 'test_embeddings.npy'

# Load Model
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = SentenceTransformer(MODEL_NAME, device=device)

# Functions
def load_articles(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_texts(articles: List[Dict]) -> List[str]:
    texts = []
    for article in articles:
        if article is None:
            continue
            
        title = article.get("title", "") or ""
        summary = article.get("summary", "") or ""
        text = f"{title}. {summary}".strip()
        texts.append(text)
    return texts

def batch_encode(texts: List[str], batch_size: int) -> np.ndarray:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Substitute empty texts to avoid errors
        batch = [text if text else "." for text in batch]
        embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

def save_metadata(articles: List[Dict], path: str):
    for article in articles:
        article.pop("embedding", None)  # Make sure there is no embedding in he json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

# Create the embeddings
if __name__ == "__main__":
    start_time = time.time()
    
    articles = load_articles(INPUT_FILE)
    texts = prepare_texts(articles)
    embeddings = batch_encode(texts, BATCH_SIZE)

    save_metadata(articles, OUTPUT_JSON)
    np.save(OUTPUT_EMBEDDINGS, embeddings)

    print(f"{len(articles)} articles processed in {time.time() - start_time:.2f} seconds")
    print(f"Metadata saved in '{OUTPUT_JSON}' and embeddings in '{OUTPUT_EMBEDDINGS}'")
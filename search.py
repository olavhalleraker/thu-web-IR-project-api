import pandas as pd
import numpy as np
import json
from data_embedding import model
from sklearn.metrics.pairwise import cosine_similarity

# Configuration parameters
METADATA_PATH = 'test_metadata.json'
EMBEDDINGS_PATH = 'test_embeddings.npy'
NUMBER_OF_RESULTS = 10000

# Load metadata and embeddings
metadata_path = METADATA_PATH 
embeddings_path = EMBEDDINGS_PATH

with open(metadata_path, 'r', encoding='utf-8') as f:
    articles = json.load(f)
embeddings = np.load(embeddings_path)  # shape: (N, D)

def search_func(query, n_results=NUMBER_OF_RESULTS):
    """
    Search for the query using cosine vector search.
    """

    # Create a DataFrame from metadata
    df = pd.DataFrame(articles)
    df['lastmod'] = pd.to_datetime(df['lastmod'], errors='coerce')
    df['image_url'] = df.get('image_url', pd.Series([''] * len(df))).fillna('')

    # Compute query embedding
    query = query.strip() if query.strip() else "."
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)  # shape: (1, D)

    # Compute cosine similarity
    similarities = cosine_similarity(embeddings, query_vector).flatten()  # shape: (N,)F

    # Sort by similarity and return the top n_results
    df['similarity'] = similarities
    results = df[df['similarity'] > 0.3].sort_values(by='similarity', ascending=False).head(n_results) # Limits to 300 results

    # Convert results to a list of dictionaries
    results_list = []
    for _, row in results.iterrows():
        result = {
            "url": row['url'],
            "lastmod": row['lastmod'].isoformat(),
            "image_url": row['image_url'],
            "title": row['title'],
            "summary": row['summary'],
            "score": row['similarity']
        }
        results_list.append(result)
    # Print a list of scores:
    print([result['score'] for result in results_list])
    return results_list

# def cosine_similarity(a, b):
#     """
#     Calculate the cosine similarity between two vectors.
#     """
#     dot_product = np.dot(a, b)
#     norm_a = np.linalg.norm(a)
#     norm_b = np.linalg.norm(b)
#     return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

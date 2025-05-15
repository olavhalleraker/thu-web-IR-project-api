import pandas as pd
import numpy as np
import json
from data_embedding import model
from sklearn.metrics.pairwise import cosine_similarity
from config import config

with open(config.METADATA_PATH, 'r', encoding='utf-8') as f:
    articles = json.load(f)
embeddings = np.load(config.EMBEDDINGS_PATH)

df = pd.DataFrame(articles)
df['lastmod'] = pd.to_datetime(df['lastmod'], errors='coerce')
df['image_url'] = df.get('image_url', pd.Series([''] * len(df))).fillna('')

def search_func(query):
    """
    Search for the query using cosine vector search.
    """

    # Create a DataFrame from metadata


    # Compute query embedding
    query = query.strip() if query.strip() else "."
    query_vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)  # shape: (1, D)

    # Compute cosine similarity
    similarities = cosine_similarity(embeddings, query_vector).flatten()  # shape: (N,)F

    # Sort by similarity and return the top n_results
    df['similarity'] = similarities
    results = df[df['similarity'] > 0.3].sort_values(by='similarity', ascending=False).head(config.NUMBER_OF_RESULTS)

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

    print([result['score'] for result in results_list])
    return results_list

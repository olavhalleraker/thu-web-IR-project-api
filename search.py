import pandas as pd
import numpy as np

def search_func(query, n_results=10):
    """
    Search for the query using cosine vector search.
    """
    
    # Load the index
    index_path = "output.json"
    df = pd.read_json(index_path)
    df['lastmod'] = pd.to_datetime(df['lastmod'])
    df['image_url'] = df['image_url'].fillna('')

    # NOTE: Create a dummy vector for the query
    # TODO: Replace this with the actual embedding of the query
    query_vector = np.random.rand(1, 1536)  # Assuming the embedding size is 1536

    # Calculate cosine similarity
    df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(np.array(x), query_vector.flatten()))

    # Sort by similarity and return the top n_results
    results = df.sort_values(by='similarity', ascending=False).head(n_results)

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
    return results_list

def cosine_similarity(a, b):
    """
    Calculate the cosine similarity between two vectors.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

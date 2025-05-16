# Configuration variables for the project

class Config:
    DEBUG = False
    TESTING = False
    METADATA_PATH = 'articles/articles_metadata.json'
    EMBEDDINGS_PATH = 'articles/articles_embeddings.npy'
    NUMBER_OF_RESULTS = 10000 # Top-k Number of results retrieved
    SIMILARITY_THRESHOLD = 0.3 # Filter for similarity of the results related to the query
    SCORE_THRESHOLD = 0.5 # Condifence below which articles are automatically labeled as 'neutral'


class DevelopmentConfig(Config):
    METADATA_PATH = 'articles/test_metadata.json'
    EMBEDDINGS_PATH = 'articles/test_embeddings.npy'
    NUMBER_OF_RESULTS = 1000


# Export the desired configuration
config = Config() 
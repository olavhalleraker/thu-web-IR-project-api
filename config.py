# Configuration variables for the project

class Config:
    DEBUG = False
    TESTING = False
    METADATA_PATH = 'articles/articles_metadata.json'
    EMBEDDINGS_PATH = 'articles/articles_embeddings.npy'
    NUMBER_OF_RESULTS = 10000
    SCORE_THRESHOLD = 0.5
    SIMILARITY_THRESHOLD = 0.2


class DevelopmentConfig(Config):
    METADATA_PATH = 'articles/test_metadata.json'
    EMBEDDINGS_PATH = 'articles/test_embeddings.npy'
    NUMBER_OF_RESULTS = 1000


# Export the desired configuration
config = Config() 
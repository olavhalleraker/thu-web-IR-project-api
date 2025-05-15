# Configuration variables for the project

class Config:
    DEBUG = False
    TESTING = False
    METADATA_PATH = 'articles_metadata.json'
    EMBEDDINGS_PATH = 'articles_embeddings.npy'
    NUMBER_OF_RESULTS = 10000
    SCORE_THRESHOLD = 0.4


class DevelopmentConfig(Config):
    METADATA_PATH = 'test_metadata.json'
    EMBEDDINGS_PATH = 'test_embeddings.npy'
    NUMBER_OF_RESULTS = 1000


# Export the desired configuration
config = Config() 
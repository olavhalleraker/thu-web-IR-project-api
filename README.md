# 🔍 BiasSearch API - Tsinghua Web Information Retrieval Project

This repository (`thu-web-IR-project-api`) provides the backend API for the **BiasSearch** project, an AI-powered news article search engine that retrieves articles relevant to a user's query and classifies them as **in favor**, **against**, or **neutral** with respect to that query. It is developed for the **Web Information Retrieval** course at **Tsinghua University**.

This frontend connects with the frontend (hosted in the `thu-web-IR-project-webapp repository`) and includes the underlying logic for the system, it handles:

- **News article search** using semantic embeddings and cosine similarity.
- **Stance classification** relative to the search query using zero-shot language models.
- **API endpoints** that connect to the frontend (`webapp`) interface.

The frontend web application (in a separate repository) connects to this API to retrieve and display search results with their classifications


## System Overview

The system consists of the following main components:

- **Search Engine**: Retrieves semantically similar articles using embeddings. 🔎
- **Stance Classifier**: Uses multiple large language models (RoBERTa, BART, DeBERTa) in a zero-shot setup to determine if the article is in favor, against, or neutral to a user's query. 🧠
- **Flask API**: Serves as the bridge between the frontend UI and backend logic.🌐
- **Embedding Geneartion**: Run once during initial setup or when the database (DB) is updated to create the embeddings needed for the search. ⚡

Refer to the Key Components sention for a more detailed explanation.

---

## 📁 Repository Structure


```
thu-web-IR-project-api
│
├── articles/
│   ├── articles_database.json      # Full set of crawled articles
│   ├── articles_metadata.json      # Article metadata (titles, summaries, etc.)
│   └── articles_embeddings.npy     # Precomputed article embeddings
│
├── app.py                         # Flask API server
├── classifier.py                  # Classification logic using multiple NLP models
├── search.py                      # Semantic search logic
├── config.py                      # Configuration constants
├── data_embedding.py              # Script for generating metadata and embeddings
├── requirements.txt               # Python dependencies
├── setup.py                       # Python package configuration
└── README.md                      # You're reading this!
```
---

## 🔧 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Article Embeddings (Optional)
If using your own dataset, use this script to generate embeddings:
```bash
python data_embedding.py
```

### 3. Run the API Locally (Development)
```bash
flask --app app run
```

### 4. Production Deployment (Waitress)
```bash
pip install build waitress
python -m build --wheel

waitress-serve --host=0.0.0.0 --port=8000 app:app

For macOS, run the following command to find you IP-adress:
```ipconfig getifaddr en0```
http://{IP-address}:8000 is the address others can access.
```
---

## 🔑 Key Components


### 1. Search Logic 🔎 
Implemented in `search.py`, articles are searched by:

1. Encoding the query using `sentence-transformers/all-MiniLM-L6-v2`.
2. Computing cosine similarity with precomputed article embeddings.
3. Returning the most relevant articles above a similarity threshold.

### 2. Classification Logic 🧠
Implemented in `classifier.py`, the classifier uses zero-shot learning with:

- `roberta-large-mnli`
- `facebook/bart-large-mnli`
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`

These models return label confidence scores (`in favor`, `against`, `neutral to`). The final prediction is a majority agreement across models, averaged and thresholded.


### 3. API Endpoints 🌐

Implemented in `app.py`,

#### `GET /`
Check API connection.

**Returns**: `"connected"`

#### `GET /search/<query>`
Searches for articles semantically similar to the query.
**Returns**: List of top articles with metadata and similarity score.

#### `GET /classify?query=...&url=...`
Classifies a single article’s stance toward the user query.

**Returns**: Agreement score and classification (`in favor of`, `against`, or `neutral to`).

#### `POST /classify/bundle?query=...`
Batch classify multiple articles.

**Body**: JSON array with `url` keys.

**Returns**: List of results with scores and stance.

### 4. Embedding Generation (One-Time Setup)⚡  
Implemented in `data_embedding.py`, it is run only during initial DB load or updates.   
It generates dense vector embeddings which are stored in a numpy matrix and enable semantic search through cosine similarity:

1. Loading articles from articles_database.json
2. Generating weighted embeddings (title 70%, summary 30%) using `sentence-transformers/all-MiniLM-L6-v2` model, batch processing for efficiency and automatic GPU acceleraiton if available.
3. Outputting cleaned articles data `articles_metadata.json` and embeddings matrix `articles_embedidngs.npy` 

### 5. Configuration ⚙️
See `config.py` for paths, thresholds, and settings. Key parameters:

- `SCORE_THRESHOLD = 0.4`
- `NUMBER_OF_RESULTS = 10000`
- File paths for embeddings and metadata

---
Olav Larsen Halleraker  
Guillermo Rodrigo Pérez  
Project for Web Information Retrieval — Tsinghua University 2024/2025
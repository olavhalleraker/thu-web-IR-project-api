import time
from flask import Flask, json, request
import pandas as pd

from classifier import classify_text, classify_texts
from search import search_func

from config import config

app = Flask(__name__)

try:
    with open(config.METADATA_PATH, 'r', encoding='utf-8') as f:
        articles = json.load(f)
except FileNotFoundError:
    print(f"Not connected to data fi: {config.METADATA_PATH}")
    articles = []

df = pd.DataFrame(articles)


@app.route("/")
def index():
    return "connected"


@app.route("/search/<q>")
def search(q):

    docs = search_func(q)
    return json.dumps(docs)

@app.route("/classify")
def classify():
    # start_time = time.time()
    query = request.args.get('query')
    url = request.args.get('url')

    doc = df.loc[df['url'] == url].to_dict('records')

    if not doc:
        return json.dumps({"error": "Document not found"}), 404
    doc = doc[0]
    
    classification = classify_text(query=query, document=(doc["title"] + '\n' + doc['summary']))

    # print("Time to classify one document: ", time.time() - start_time)
    return json.dumps({"query": query, "url": url, "agreementscore": classification[0], "classification": classification[1]})

@app.route("/classify/bundle", methods=["POST"])
def classify_bundle():
    start_time = time.time()

    data = request.get_json()
    query = request.args.get('query')

    print("Time to load metadata:", time.time() - start_time)

    url_to_doc = {article['url']: ((article["title"] or '') + '\n' + (article['summary'] or '')) for article in articles}

    documents = []
    results = []

    for item in data:
        url = item.get('url')
        if url in url_to_doc:
            documents.append(url_to_doc[url])
        else:
            results.append({"query": query, "url": url, "error": "Document not found"})

    if documents:
        classifications = classify_texts(query=query, documents=documents)
        # Map classifications back to URLs
        doc_index = 0
        for item in data:
            url = item.get('url')
            if url in url_to_doc:
                classification = classifications[doc_index]
                results.append({
                    "query": query,
                    "url": url,
                    "agreementscore": classification[0],
                    "classification": classification[1]
                })
                doc_index += 1

    print("Time to process ", len(results), " results: ", time.time() - start_time)

    return json.dumps(results)

import time
from flask import Flask, json, request

from classifier import classify_text, classify_texts
from search import search_func

app = Flask(__name__)


@app.route("/")
def index():
    return "connected"


@app.route("/search/<q>")
def search(q):

    docs = search_func(q)
    return json.dumps(docs)

@app.route("/classify")
def classify():
    start_time = time.time()
    query = request.args.get('query')
    url = request.args.get('url')
    
    # retrieve document from database
    with open('test_metadata.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    for article in articles:
        if article['url'] == url:
            doc = article
            break
    else:
        return json.dumps({"error": "Document not found"}), 404
    
    classification = classify_text(query=query, document=(doc["title"] + '\n' + doc['summary']))

    print("Time to classify one document: ", time.time() - start_time)
    return json.dumps({"query": query, "url": url, "agreementscore": classification[0], "classification": classification[1]})

@app.route("/classify/bundle", methods=["POST"])
def classify_bundle():
    start_time = time.time()

    data = request.get_json()
    query = request.args.get('query')
    with open('test_metadata.json', 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print("Time to load metadata:", time.time() - start_time)
    # Create a mapping of URLs to documents
    url_to_doc = {article['url']: ((article["title"] or '') + '\n' + (article['summary'] or '')) for article in articles}

    # Prepare documents for classification
    documents = []
    results = []

    for item in data:
        url = item.get('url')
        if url in url_to_doc:
            documents.append(url_to_doc[url])
        else:
            results.append({"query": query, "url": url, "error": "Document not found"})

    # Run classify_texts for all documents at once
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

from flask import Flask, json

from search import search_func

app = Flask(__name__)


@app.route("/")
def index():
    return "Index Page"


@app.route("/hello")
def hello():
    return "Hello, World"


@app.route("/search/<q>")
def search(q):

    docs = search_func(q, 100)
    return json.dumps(docs)
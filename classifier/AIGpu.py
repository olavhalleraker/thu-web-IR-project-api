from transformers import pipeline
import json
import torch
import time

# ---------- Setup ----------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}")

query = "Trump interpretation of the constitution"

MODEL_NAME = 'roberta-large-mnli'
INPUT_FILE = 'testCnnAI.json'
OUTPUT_COMPACT = 'AIGPU/testStanceCnnCompact.json'
OUTPUT_VERBOSE = 'AIGPU/testStanceCnnVerbose.json'
labels = ["in favor of", "against", "neutral to"]

# ---------- Load Data ----------

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    articles = json.load(f)

# ---------- Prepare Input Texts ----------
input_texts = [f"{article['title']} {article['summary']}" for article in articles]

# ---------- Initialize Classifier ----------
classifier = pipeline("zero-shot-classification", model=MODEL_NAME, device=0 if device == 'cuda' else -1)

# ---------- Run Classification in Batch ----------

start_time = time.time()

results = classifier(
    input_texts,
    candidate_labels=labels,
    hypothesis_template="This text is {} the statement: '" + query 
)

# ---------- Process Results ----------


compact_results = []
verbose_results = []

for article, result in zip(articles, results):
    top_index = result["scores"].index(max(result["scores"]))
    stance = result["labels"][top_index]
    score = result["scores"][top_index]

    # Compact
    compact_results.append({
        "stance": stance,
        "title": article["title"],
        "summary": article["summary"],
    })

    # Verbose
    verbose_article = article.copy()
    verbose_article.update({
        "stance": stance,
        "stance_score": round(score, 4)
    })
    verbose_results.append(verbose_article)

print(f"{len(articles)} articles processed in {time.time() - start_time:.2f} seconds")

# ---------- Save Outputs ----------
with open(OUTPUT_COMPACT, 'w', encoding='utf-8') as f:
    json.dump(compact_results, f, indent=2, ensure_ascii=False)

with open(OUTPUT_VERBOSE, 'w', encoding='utf-8') as f:
    json.dump(verbose_results, f, indent=2, ensure_ascii=False)

print("Done. Results saved.")

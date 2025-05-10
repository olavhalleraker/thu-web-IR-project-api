## MOdel Inference Optimization



from transformers import pipeline
import json
import torch
import time
import numpy as np
import os

# ---------- Setup ----------

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device} {torch.cuda.get_device_name(0) if device == 'cuda' else ''}")

query = "Trump interpretation of the constitution"
MODELS = [
    'roberta-large-mnli',
    'facebook/bart-large-mnli',
    'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
]

INPUT_FILE = 'testCnnAI.json'
OUTPUT_COMPACT = 'AIGPU/testStanceCnnCompact.json'
OUTPUT_VERBOSE = 'AIGPU/testStanceCnnVerbose.json'
labels = ["in favor of", "against", "neutral to"]
batch_size = 64

# ---------- Load Data ----------

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    articles = json.load(f)

input_texts = [f"{article['title']} {article['summary']}" for article in articles]

# ---------- Load Models (Once) ----------

print("Loading models...")
loaded_classifiers = {}

for model_name in MODELS:
    print(f"  🔄 Loading {model_name}")
    loaded_classifiers[model_name] = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if device == 'cuda' else -1
    )
print("✅ All models loaded.\n")

# ---------- Inference Function ----------

def run_model_on_texts(classifier, texts, labels, query, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"  Processing batch {i}–{i + len(batch)}")
        try:
            batch_results = classifier(
                batch,
                candidate_labels=labels,
                hypothesis_template="This text is {} the statement: '" + query
            )
            if isinstance(batch_results, dict):
                results.append(batch_results)
            else:
                results.extend(batch_results)
        except Exception as e:
            print(f"  ⚠️ Batch {i}-{i+len(batch)} failed: {e}")
    return results

# ---------- Run Inference ----------

start_time = time.time()
all_model_outputs = []

print("Running inference for all models...\n")
for model_name, classifier in loaded_classifiers.items():
    print(f"▶️ Inference with: {model_name}")
    model_results = run_model_on_texts(classifier, input_texts, labels, query, batch_size)
    print(f"  ✔️ Finished model {model_name} — {len(model_results)} results\n")
    all_model_outputs.append(model_results)

# ---------- Combine & Save Results ----------

print("Combining results...")
compact_results = []
verbose_results = []

for idx, article in enumerate(articles):
    label_scores = {label: [] for label in labels}

    for model_results in all_model_outputs:
        result = model_results[idx]
        for label, score in zip(result["labels"], result["scores"]):
            label_scores[label].append(score)

    avg_scores = {label: np.mean(scores) for label, scores in label_scores.items()}
    stance = max(avg_scores, key=avg_scores.get)
    stance_score = avg_scores[stance]

    compact_results.append({
        "stance": stance,
        "title": article["title"],
        "summary": article["summary"],
    })

    verbose_article = article.copy()
    verbose_article.update({
        "stance": stance,
        "stance_score": round(stance_score, 4),
        "avg_scores": {k: round(v, 4) for k, v in avg_scores.items()}
    })
    verbose_results.append(verbose_article)

print(f"\n✅ {len(articles)} articles processed in {time.time() - start_time:.2f} seconds")

# ---------- Save to Disk ----------

with open(OUTPUT_COMPACT, 'w', encoding='utf-8') as f:
    json.dump(compact_results, f, indent=2, ensure_ascii=False)

with open(OUTPUT_VERBOSE, 'w', encoding='utf-8') as f:
    json.dump(verbose_results, f, indent=2, ensure_ascii=False)

print("📁 Done. Results saved to:")
print(f"  • {OUTPUT_COMPACT}")
print(f"  • {OUTPUT_VERBOSE}")


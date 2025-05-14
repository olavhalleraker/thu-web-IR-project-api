from transformers import pipeline
import torch
import numpy as np

# ---------- Setup ----------
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

SCORE_THRESHOLD = 0.4

# device_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'Metal Performance Shaders' if device == 'mps' else ''
# print(f"Using device: {device} {device_name}")

MODELS = [
    'roberta-large-mnli',
    'facebook/bart-large-mnli',
    'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
]

LABELS = ["in favor of", "against", "neutral to"]

# ---------- Load Models ----------
print("Loading models...")
loaded_classifiers = {
    model_name: pipeline(
        "zero-shot-classification",
        model=model_name,
        device=0 if device in ['cuda', 'mps'] else -1
    )
    for model_name in MODELS
}
print("All models loaded.\n")


# ---------- Classification Function ----------
def classify_text(query, document):
    """
    Returns the average score for the most likely label across models.
    Only one document and one query are expected.
    """
    label_scores = {label: [] for label in LABELS}

    hypothesis_template = "This text is {} the statement: '" + query + "'"

    for model_name, classifier in loaded_classifiers.items():
        result = classifier(
            sequences=document,
            candidate_labels=LABELS,
            hypothesis_template=hypothesis_template
        )
        for label, score in zip(result["labels"], result["scores"]):
            label_scores[label].append(score)
    print(label_scores)

    avg_scores = {label: np.mean(scores) for label, scores in label_scores.items()}
    stance = max(avg_scores, key=avg_scores.get)
    stance_score = avg_scores[stance]
    if stance_score < SCORE_THRESHOLD:
        stance = "neutral to"


    return (round(stance_score, 3), -1 if stance == "against" else 1 if stance == "in favor of" else 0)

def classify_texts(query, documents):
    """
    Classifies each document in relation to the query using multiple classifiers.
    Returns a list of (score, stance) for each document.
    """
    # hypothesis_template = "This text is {} the statement: '" + query + "'"
    hypothesis_template = "The opinion in this text is {} \n the claim: '" + query + "'"

    document_scores = [[] for _ in documents]

    for model_name, classifier in loaded_classifiers.items():
        shuffled_labels = LABELS.copy()[:]
        results = classifier(
            sequences=documents,
            candidate_labels=shuffled_labels,
            hypothesis_template=hypothesis_template
        )
        for i, res in enumerate(results):
            label_score = {label: 0.0 for label in LABELS}
            for label, score in zip(res["labels"], res["scores"]):
                label_score[label] = score
            document_scores[i].append(label_score)

    final_results = []
    for scores_list in document_scores:

        avg_scores = {
            label: np.mean([scores[label] for scores in scores_list])
            for label in LABELS
        }
        stance = max(avg_scores, key=avg_scores.get)
        stance_score = avg_scores[stance]
        
        stance_value = -1 if stance == "against" else 1 if stance == "in favor of" else 0
        if stance_score < SCORE_THRESHOLD:
            stance_value = 0
        final_results.append((round(stance_score, 3), stance_value))

    return final_results

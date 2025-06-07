from transformers import pipeline
import torch
import numpy as np
from config import config

# ---------- Setup ----------
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

MODELS = [
    'roberta-large-mnli',
    'facebook/bart-large-mnli',
    # 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
]

LABELS = ["in favor of", "against", "neutral to"]
# LABELS = ["entailing", "neutral to",  "contradicting"]

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
def classify_text(query, document, score_threshold=config.SCORE_THRESHOLD):
    """
    Returns the average score for the most likely label across models.
    Only one document and one query are expected.
    """
    label_scores = {label: [] for label in LABELS}

    # hypothesis_template = "This text is {} the statement: '" + query + "'"
    hypothesis_template = "This text is {} the statement: '" + query + "'"


    for model_name, classifier in loaded_classifiers.items():
        result = classifier(
            sequences=document,
            candidate_labels=LABELS,
            hypothesis_template=hypothesis_template
        )
        for label, score in zip(result["labels"], result["scores"]):
            label_scores[label].append(score)

    avg_scores = {label: np.mean(scores) for label, scores in label_scores.items()}
    stance = max(avg_scores, key=avg_scores.get)
    stance_score = avg_scores[stance]

    # if (stance == LABELS[2]

    if stance_score < score_threshold:
        stance = LABELS[1]


    return (round(stance_score, 3), -1 if stance == LABELS[2] else 1 if stance == LABELS[0] else 0)


def classify_text_binary(query, document, score_threshold=config.SCORE_THRESHOLD):
    """
    Classifies stance as Agree, Disagree, or Neutral based on threshold.

    Returns:
        (stance_score, stance_label)
            stance_score: float (confidence score of the selected stance)
            stance_label: int (1 for Agree, -1 for Disagree, 0 for Neutral)
    """
    # Only consider two stance labels
    stance_labels = [LABELS[0], LABELS[-1]]  # Assuming LABELS[0] = "Agree", LABELS[-1] = "Disagree"
    label_scores = {label: [] for label in stance_labels}

    hypothesis_template = "The stance of the document towards the statement '{}' is: {}".format(query, "{}")

    for model_name, classifier in loaded_classifiers.items():
        result = classifier(
            sequences=document,
            candidate_labels=stance_labels,
            hypothesis_template=hypothesis_template
        )
        for label, score in zip(result["labels"], result["scores"]):
            label_scores[label].append(score)

    # Average the scores across models
    avg_scores = {label: np.mean(scores) for label, scores in label_scores.items()}

    print(label_scores)

    stance = max(avg_scores, key=avg_scores.get)
    stance_score = avg_scores[stance]

    # Apply threshold: force Neutral if below threshold
    if stance_score < score_threshold:
        return (round(stance_score, 3), 0)  # Neutral

    # Map stance to label
    stance_label = 1 if stance == stance_labels[0] else -1

    return (round(stance_score, 3), stance_label)
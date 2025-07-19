from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer


MODEL_DIR = "models/text"

try:
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
except Exception:  # pragma: no cover - fallback when local model is missing
    _tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    _model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )


def _predict_proba(texts: List[str]) -> List[List[float]]:
    inputs = _tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def explain(sentence: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Generate word importances for a sentence using LIME."""
    explainer = LimeTextExplainer(class_names=["benign", "harass", "hate"])
    exp = explainer.explain_instance(sentence, _predict_proba, num_features=top_k)
    return exp.as_list()

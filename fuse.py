import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np



MODEL_TEXT_DIR = "models/text"
MODEL_VISION_CKPT = "best.ckpt"
DATA_DIR = Path("data/video")
REPORTS_DIR = Path("reports")


def concat_probs(vision_probs: np.ndarray, text_probs: np.ndarray) -> np.ndarray:
    """Concatenate vision and text probabilities."""
    return np.concatenate([vision_probs, text_probs])


def train_logreg(X: np.ndarray, y: np.ndarray):
    """Train a logistic regression classifier."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    return clf


def _load_vision_model(ckpt_path: str):
    from train_vision import DeepFakeModel
    model = DeepFakeModel.load_from_checkpoint(ckpt_path)
    model.eval()
    return model


def _load_text_model(model_dir: str):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


def _vision_prob(model, path: str) -> np.ndarray:
    import torch
    import torchvision.transforms as T
    from torchvision.io import read_video
    from torchvision.models import EfficientNet_V2_S_Weights

    video, _, _ = read_video(path, pts_unit="sec")
    frame = video[0].permute(2, 0, 1).float() / 255.0
    weights = EfficientNet_V2_S_Weights.DEFAULT
    transform = T.Compose([
        T.Resize(weights.transforms().crop_size),
        T.CenterCrop(weights.transforms().crop_size),
        T.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
    ])
    frame = transform(frame)
    with torch.no_grad():
        logits = model(frame.unsqueeze(0))
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def _text_prob(whisper_model, tokenizer, model, path: str) -> np.ndarray:
    import torch

    result = whisper_model.transcribe(path)
    text = result.get("text", "")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def _gather(split: str) -> List[Tuple[str, int]]:
    samples = []
    for label, cls in enumerate(["real", "fake"]):
        for p in (DATA_DIR / cls / split).glob("*.mp4"):
            samples.append((str(p), label))
    return samples


def predict(video_path: str,
            vision_ckpt: str = MODEL_VISION_CKPT,
            text_model_dir: str = MODEL_TEXT_DIR,
            clf_path: Path = REPORTS_DIR / "logreg.pkl") -> float:
    """Predict the fake probability for a video using fused models."""
    import joblib

    vision_model = _load_vision_model(vision_ckpt)
    tokenizer, text_model = _load_text_model(text_model_dir)
    import whisper
    whisper_model = whisper.load_model("tiny")
    clf = joblib.load(clf_path)
    v_prob = _vision_prob(vision_model, video_path)
    t_prob = _text_prob(whisper_model, tokenizer, text_model, video_path)
    feats = concat_probs(v_prob, t_prob).reshape(1, -1)
    return float(clf.predict_proba(feats)[0, 1])


def main() -> None:
    import joblib
    from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_ckpt", default=MODEL_VISION_CKPT)
    parser.add_argument("--text_model_dir", default=MODEL_TEXT_DIR)
    args = parser.parse_args()

    vision_model = _load_vision_model(args.vision_ckpt)
    tokenizer, text_model = _load_text_model(args.text_model_dir)
    import whisper
    whisper_model = whisper.load_model("tiny")

    val_samples = _gather("val")
    test_samples = _gather("test")

    def build_features(samples: List[Tuple[str, int]]):
        X, y = [], []
        for path, label in samples:
            v_prob = _vision_prob(vision_model, path)
            t_prob = _text_prob(whisper_model, tokenizer, text_model, path)
            X.append(concat_probs(v_prob, t_prob))
            y.append(label)
        return np.array(X), np.array(y)

    X_val, y_val = build_features(val_samples)
    X_test, y_test = build_features(test_samples)

    clf = train_logreg(X_val, y_val)

    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    ConfusionMatrixDisplay.from_predictions(y_test, clf.predict(X_test))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    model_path = REPORTS_DIR / "logreg.pkl"
    joblib.dump(clf, model_path)
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Confusion matrix saved to {cm_path}")
    print(f"Classifier saved to {model_path}")


if __name__ == "__main__":
    main()


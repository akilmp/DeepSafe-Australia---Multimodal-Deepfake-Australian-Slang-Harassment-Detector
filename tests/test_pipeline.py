import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
from moviepy import ColorClip

sys.path.insert(0, os.path.abspath("src"))
import fuse


def make_video(path: Path, color):
    ColorClip(size=(64, 64), color=color, duration=1).write_videofile(
        str(path), codec="libx264", fps=24, logger=None
    )


def test_full_pipeline(monkeypatch):
    assets = Path("data/test_assets")
    assets.mkdir(parents=True, exist_ok=True)
    real_path = assets / "real.mp4"
    fake_path = assets / "fake.mp4"
    if not real_path.exists():
        make_video(real_path, (0, 0, 0))
    if not fake_path.exists():
        make_video(fake_path, (255, 0, 0))

    clf = mock.MagicMock()
    clf.predict_proba.side_effect = (
        lambda X: np.array([[0.9677, 0.0323]])
        if X[0][0] > 0.5
        else np.array([[0.2689, 0.7311]])
    )

    monkeypatch.setitem(sys.modules, "whisper", mock.MagicMock(load_model=lambda _: object()))
    monkeypatch.setattr(fuse, "_load_vision_model", lambda _: None)
    monkeypatch.setattr(fuse, "_load_text_model", lambda _: (None, None))
    monkeypatch.setattr(
        fuse,
        "_vision_prob",
        lambda m, p: np.array([0.9, 0.1]) if "real" in p else np.array([0.2, 0.8]),
    )
    monkeypatch.setattr(
        fuse,
        "_text_prob",
        lambda *a, **k: np.array([0.9, 0.05, 0.05])
        if "real" in k.get("path", a[-1])
        else np.array([0.1, 0.2, 0.7]),
    )
    import joblib
    monkeypatch.setattr(joblib, "load", lambda _: clf)

    real_score = fuse.predict(str(real_path))
    fake_score = fuse.predict(str(fake_path))
    assert real_score < 0.3
    assert fake_score > 0.7

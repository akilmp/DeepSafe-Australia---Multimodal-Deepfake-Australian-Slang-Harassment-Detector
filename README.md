# DeepSafe Australia: Deepfake & Australian Slang detector to find targetted harrasment 

DeepSafe Australia  is a research prototype combining vision and language models to flag both manipulated video content and harmful speech, especially slang common among Australian teens, by assigning a unified “risk score” to short video clips. It fine‑tunes EfficientNetV2‑S on real vs. fake videos and DistilBERT on a curated Australian‑slang corpus, then fuses their outputs via a lightweight logistic regression layer. This multimodal setup not only detects deepfake manipulations but also understands regional slang to catch harassment that standard models might miss, helping platforms preemptively identify and mitigate targeted abuse

Created and maintained by **Akil**.

## Directory layout

- **`api/`** – FastAPI service exposing a `/score` endpoint.
- **`app/`** – Streamlit front end for interactive predictions.
- **`data/`** – Placeholder folder for raw and processed datasets.
- **`docker/`** – Dockerfile for running the Streamlit app in a container.
- **`docs/`** – Project documentation such as the Basic Online Safety Expectations checklist.
- **`notebooks/`** – Jupyter notebooks for exploration.
- **`src/deepsafe_sydney/`** – Python package with helper modules.
- **`tests/`** – Unit tests for critical pieces of logic.

## Design decisions

- **Modular training** – Vision and text models are trained separately so each
  can evolve without retraining the other. Fusing them with a small logistic
  regression layer keeps the overall system interpretable and limits
  overfitting on the limited dataset.
- **Efficient backbones** – We fine-tune EfficientNetV2-S and DistilBERT as they
  offer a good accuracy/performance trade-off for short clips and slang-heavy
  text.
- **Lazy imports** – Heavy dependencies such as Transformers and
  scikit-learn are imported inside the functions that use them. This keeps test
  startup fast and allows parts of the codebase to run on machines with only the
  required subset of packages.
- **Streamlit + FastAPI** – Streamlit provides an accessible demo interface
  while FastAPI exposes the same logic for programmatic integration.
- **Docker first** – A Dockerfile mirrors the Streamlit Cloud environment so the
  app behaves the same locally and in the cloud.

## Installation

Install dependencies using [Poetry](https://python-poetry.org/):

```bash
poetry install
```

This will install PyTorch, Transformers, Streamlit and other libraries declared in `pyproject.toml`.

## Running the Streamlit app locally

Launch the user-facing interface that lets you upload a short video, view heatmaps and download the model card:

```bash
poetry run streamlit run app/app.py
```

## API server

A lightweight FastAPI service is available for programmatic access. Start it with:

```bash
poetry run start-api
```

Send a `POST` request to `/score` containing either a video file or an `s3_url` pointing to an MP4. The server returns JSON in the form `{ "risk_score": float }`.

## Running on Streamlit Cloud

Streamlit Cloud can run the project directly from this repository. Go to [share.streamlit.io](https://share.streamlit.io), connect your fork, choose a branch and set `app/app.py` as the entry point. Streamlit Cloud enforces a 200 MB upload limit for `st.file_uploader`.

## Training

- **Vision model** – `train_vision.py` fine-tunes EfficientNetV2-S on videos located in `data/video/{real|fake}/` and saves `best.ckpt` according to validation AUC.
- **Text classifier** – `train_text.py` fine-tunes DistilBERT on labelled lines in `data/slang/labels.csv` for a three-way benign/harassment/hate task.
- **Fusion** – `fuse.py` loads the vision and text probabilities, trains a logistic regression model and stores metrics such as ROC-AUC and a confusion matrix in `reports/`.

## Explanation tools

- `vision_explain.py` produces GradCAM heatmap GIFs from `best.ckpt`.
- `src/deepsafe_sydney/text_explain.py` generates word-importance scores using LIME.

## Utilities

- `src/deepsafe_sydney/video_tools.py` provides helpers for anonymising videos and creating face-swapped deep fakes using the faceswap CLI.
- `src/deepsafe_sydney/scrape_slang.py` gathers example posts containing common Australian slurs from Twitter and Reddit.
- `src/deepsafe_sydney/label_tool.py` launches either Prodigy or a minimal Streamlit UI to label text lines when building `data/slang/labels.csv`.

## Docker

Use the provided Dockerfile to build an image with all dependencies and run the Streamlit app:

```bash
docker build -t deepsafe .
docker run -p 8501:8501 deepsafe
```

## Model card

Run `python make_model_card.py` after evaluation to create `model_card.pdf`. The script merges metrics from `reports/metrics.json`, bias results from `reports/bias_audit.json` and the checklist in `docs/BOSE_checklist.md`.

## Tests

Execute the test suite with:

```bash
pytest -q
```


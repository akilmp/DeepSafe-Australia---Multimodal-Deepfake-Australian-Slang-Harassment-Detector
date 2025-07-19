import os
import csv
import subprocess
from pathlib import Path
from typing import List


def parse_env_file(path: Path) -> dict:
    env_vars = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env_vars[key] = value
    return env_vars


def load_env() -> dict:
    env = parse_env_file(Path(".env"))
    for k, v in env.items():
        os.environ.setdefault(k, v)
    return env


def run_prodigy() -> None:
    subprocess.run(
        [
            "prodigy",
            "textcat.manual",
            "slang",
            "data/slang/raw",
            "data/slang/labels.csv",
        ],
        check=True,
    )


def save_labels(sentences: List[str], labels: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "label"])
        writer.writerows(zip(sentences, labels))


def run_streamlit() -> None:
    import streamlit as st

    st.title("Slang Labelling")
    txt_file = st.file_uploader("Upload sentences (.txt)", type="txt")
    if txt_file:
        sentences = txt_file.getvalue().decode("utf-8").splitlines()
        labels = []
        for i, sentence in enumerate(sentences):
            label = st.radio(sentence, ["benign", "harassment", "hate"], key=i)
            labels.append(label)
        if st.button("Save"):
            save_labels(sentences, labels, Path("data/slang/labels.csv"))
            st.success("Labels saved to data/slang/labels.csv")


def main() -> None:
    env = load_env()
    if env.get("PRODIGY", os.getenv("PRODIGY", "")).lower() == "true":
        run_prodigy()
    else:
        run_streamlit()


if __name__ == "__main__":
    main()

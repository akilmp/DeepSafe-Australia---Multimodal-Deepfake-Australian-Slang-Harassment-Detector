import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import precision_recall_fscore_support


def preprocess(examples, tokenizer, label2id):
    texts = examples["sentence"]
    labels = [label2id[l] for l in examples["label"]]
    encodings = tokenizer(texts, truncation=True, padding="max_length")
    encodings["labels"] = labels
    return encodings


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, preds, average="macro"
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    dataset = load_dataset("csv", data_files="data/slang/labels.csv", split="train")
    dataset = dataset.train_test_split(test_size=0.2, stratify_by_column="label", seed=42)
    label_list = sorted(set(dataset["train"]["label"]))
    label2id = {l: i for i, l in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    def tokenize(batch):
        return preprocess(batch, tokenizer, label2id)

    tokenized = dataset.map(tokenize, batched=True)

    args_out = TrainingArguments(
        output_dir="models/text",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args_out,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    save_dir = Path("models/text")
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()

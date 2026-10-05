"""REFERENCE anchor: fine-tune DNABERT-2 on prom_core_tata and evaluate on the test split. Defaults
(3 epochs, batch 32, lr 3e-5, max_length 20, warmup 50) match info/protocol.md; GUE uses 10 epochs
and batch 8. --seed does not control the classifier-head initialization, matching the agent's
unseeded runs.

Usage:
    cd models/dnabert-2/dev_env
    uv run --project . python ../../../solutions/promoter-prediction/dev/run_reference.py
"""

import json
import sys
import time
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.models.bert.configuration_bert import BertConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from eval.metrics import _mcc  # noqa: E402

# The local copy the agent also uses; download.sh removed its broken Triton kernel.
WEIGHTS_PATH = REPO_ROOT / "models" / "dnabert-2" / "agent" / "weights"
GUE_CONFIG = "prom_core_tata"
DEFAULT_RESULTS_FILE = REPO_ROOT / "solutions" / "promoter-prediction" / "dev" / "results" / "run_reference.jsonl"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Predictions arrive as (logits, hidden_states); keep the logits.
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    return {"mcc": _mcc(pd.Series(labels), pd.Series(preds))}


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="TrainingArguments seed (shuffling and dropout, not the head initialization)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32, help="per_device_train_batch_size")
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=None, help="Subsample the train split to this many examples (default: full 4904)")
    parser.add_argument("--label", default=None, help="Free-text tag for this run in --results-file, e.g. 'fast'/'slow' -- not passed to the recipe itself")
    parser.add_argument("--results-file", type=lambda s: Path(s) if s else None, default=DEFAULT_RESULTS_FILE, help="append one JSON line per run (hyperparameters, seed, loss, test MCC, time); empty to skip")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("leannmlindsey/GUE", GUE_CONFIG)
    if args.max_train is not None:
        ds["train"] = ds["train"].select(range(min(args.max_train, len(ds["train"]))))

    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_PATH), trust_remote_code=True, local_files_only=True)
    config = BertConfig.from_pretrained(str(WEIGHTS_PATH), num_labels=2, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(WEIGHTS_PATH), trust_remote_code=True, config=config, local_files_only=True
    ).to(device)

    def tokenize(batch):
        return tokenizer(batch["sequence"], padding="max_length", truncation=True, max_length=args.max_length)

    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=str(REPO_ROOT / "solutions" / "promoter-prediction" / "dev" / "results" / "run"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=16,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        fp16=(device == "cuda"),
        logging_steps=50,
        eval_strategy="no",  # only eval once at the end, below
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=compute_metrics,
    )

    start = time.monotonic()
    train_result = trainer.train()
    train_wall_s = time.monotonic() - start
    metrics = trainer.evaluate()
    mcc = metrics["eval_mcc"]
    print(f"test MCC (seed={args.seed}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}): {mcc:.4f}")

    if args.results_file:
        args.results_file.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "max_length": args.max_length,
            "max_train": args.max_train,
            "n_train": len(ds["train"]),
            "train_wall_s": train_wall_s,
            "final_train_loss": train_result.metrics.get("train_loss"),
            "test_mcc": mcc,
        }
        with args.results_file.open("a") as f:
            f.write(json.dumps(record) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sanity check that agent/weights load and run a forward and backward pass. Runs in dev_env, not agent/env.

Usage:
    cd models/dnabert-2 && uv run --project dev_env python verify.py
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

WEIGHTS_PATH = Path(__file__).resolve().parent / "agent" / "weights"


def main() -> int:
    if not (WEIGHTS_PATH / "pytorch_model.bin").is_file():
        print(f"error: no checkpoint found at {WEIGHTS_PATH} -- run ./download.sh first")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DNABERT-2 from {WEIGHTS_PATH} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_PATH), trust_remote_code=True, local_files_only=True)
    config = BertConfig.from_pretrained(str(WEIGHTS_PATH), num_labels=2, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(WEIGHTS_PATH), trust_remote_code=True, config=config, local_files_only=True
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded ok, params={n_params}")
    if n_params != 117_070_082:
        print(f"error: expected 117,070,082 params, got {n_params}")
        return 1

    sequences = [
        "ACTACAAATGGACGAGAGAGGCGGCCGTCCATTAGTTAGCGGCTCCGGAGCAACGCAGCCGTTGTCCTTG",
        "AGTTTAAAAGCCAGCCAGTCATACTAAAAAAAAGAATTCAGGTTTTCAGTAGCTTCTGAAGATATATATT",
    ]
    inputs = tokenizer(sequences, return_tensors="pt", padding="max_length", truncation=True, max_length=20).to(device)
    labels = torch.tensor([1, 0], device=device)

    out = model(**inputs, labels=labels)
    print(f"logits shape={tuple(out.logits.shape)} loss={out.loss.item():.4f}")
    if out.logits.shape != (2, 2):
        print(f"error: expected logits shape (2, 2), got {tuple(out.logits.shape)}")
        return 1
    if not torch.isfinite(out.logits).all():
        print("error: logits contain NaN/Inf")
        return 1

    out.loss.backward()
    grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
    print(f"backward ok, total grad abs sum={grad_norm:.4f}")
    if grad_norm <= 0:
        print("error: no gradient flowed")
        return 1

    print("\nOK: model loaded and produced a well-formed forward+backward pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

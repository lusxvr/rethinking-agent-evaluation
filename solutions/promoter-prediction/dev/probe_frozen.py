"""Ablation: logistic regression on frozen, mean-pooled DNABERT-2 embeddings, to see how much of
REFERENCE (0.7460) needs fine-tuning.

Usage:
    cd models/dnabert-2/dev_env
    uv run --project . python ../../../solutions/promoter-prediction/dev/probe_frozen.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from eval.metrics import _mcc  # noqa: E402

WEIGHTS_PATH = REPO_ROOT / "models" / "dnabert-2" / "agent" / "weights"
GUE_CONFIG = "prom_core_tata"
MAX_LENGTH = 20


@torch.no_grad()
def embed(model, tokenizer, sequences: list[str], device: str, batch_size: int = 64) -> np.ndarray:
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH
        ).to(device)
        # Output [0] is the token-level last hidden state.
        out = model(**inputs)
        hidden = out[0]  # (batch, seq_len, hidden_size)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        mean_pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        embeddings.append(mean_pooled.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset("leannmlindsey/GUE", GUE_CONFIG)

    tokenizer = AutoTokenizer.from_pretrained(str(WEIGHTS_PATH), trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(str(WEIGHTS_PATH), trust_remote_code=True, local_files_only=True).to(device)
    model.eval()

    train_seqs, train_labels = ds["train"]["sequence"], np.array(ds["train"]["label"])
    test_seqs, test_labels = ds["test"]["sequence"], np.array(ds["test"]["label"])

    print(f"embedding {len(train_seqs)} train + {len(test_seqs)} test sequences (frozen backbone, no gradient steps)...")
    train_emb = embed(model, tokenizer, train_seqs, device)
    test_emb = embed(model, tokenizer, test_seqs, device)

    probe = LogisticRegression(max_iter=2000)
    probe.fit(train_emb, train_labels)
    preds = probe.predict(test_emb)

    mcc = _mcc(pd.Series(test_labels), pd.Series(preds))
    print(f"[summary] frozen_embedding_probe_mcc={mcc:.4f} (vs REFERENCE=0.7460 full fine-tune, BACKBONE=-0.0298 bare LLM)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""REFERENCE anchor: inference-only secondary-structure prediction with RiNALMo giga (bpRNA
fine-tuned) on the TS0 test sample. Post-processing and the dot-bracket encoder are ported from
RiNALMo's rinalmo/utils/sec_struct.py (Apache 2.0); info/protocol.md gives the agent this code.

Usage:
    cd models/rinalmo/dev_env
    uv run --project . python ../../../solutions/rna-folding/dev/run_reference.py
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.downstream import SecStructPredictionHead
from rinalmo.model.model import RiNALMo

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from eval.metrics import _sec_struct_f1  # noqa: E402

WEIGHTS_PATH = REPO_ROOT / "models" / "rinalmo" / "agent" / "weights" / "rinalmo_giga_ss_bprna_ft.pt"
QUERY_PATH = REPO_ROOT / "tasks" / "rna-folding" / "data" / "query_sequences.csv"
SOLUTION_PATH = REPO_ROOT / "solutions" / "rna-folding" / "solution.csv"
OUTPUT_PATH = REPO_ROOT / "solutions" / "rna-folding" / "dev" / "results" / "reference_prediction.csv"

NUM_RESNET_BLOCKS = 2  # SecStructPredictionWrapper's default in RiNALMo's train_sec_struct_prediction.py

# --- checkpoint loading (as in models/rinalmo/verify.py; strict=False only for rotary buffers) ---


def load_giga_ss_model(device: str):
    config = model_config("giga")
    config.model.transformer.use_flash_attn = False
    lm = RiNALMo(config)
    pred_head = SecStructPredictionHead(config.model.transformer.embed_dim, num_blocks=NUM_RESNET_BLOCKS)

    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    threshold = state_dict.pop("threshold")

    lm_state = {k[len("lm."):]: v for k, v in state_dict.items() if k.startswith("lm.")}
    head_state = {k[len("pred_head."):]: v for k, v in state_dict.items() if k.startswith("pred_head.")}
    missing, unexpected = lm.load_state_dict(lm_state, strict=False)
    if unexpected or any(not k.endswith("rotary_emb.inv_freq") for k in missing):
        raise RuntimeError(f"unexpected load_state_dict mismatch: missing={missing}, unexpected={unexpected}")
    pred_head.load_state_dict(head_state)

    return lm.to(device).eval(), pred_head.to(device).eval(), threshold, config


# --- post-processing pipeline, ported from rinalmo/utils/sec_struct.py ---

_SHARP_LOOP_DIST_THRESHOLD = 4
_CANONICAL_PAIRS = {"AU", "UA", "GC", "CG", "GU", "UG"}


def _sharp_loop_mask(seq_len: int) -> np.ndarray:
    mask = np.eye(seq_len, k=0, dtype=bool)
    for i in range(1, _SHARP_LOOP_DIST_THRESHOLD):
        mask = mask + np.eye(seq_len, k=i, dtype=bool) + np.eye(seq_len, k=-i, dtype=bool)
    return mask


def _canonical_pairs_mask(seq: str) -> np.ndarray:
    seq = seq.upper().replace("T", "U")
    mask = np.zeros((len(seq), len(seq)), dtype=bool)
    for i, nt_i in enumerate(seq):
        for j, nt_j in enumerate(seq):
            if f"{nt_i}{nt_j}" in _CANONICAL_PAIRS:
                mask[i, j] = True
    return mask


def _clean_sec_struct(sec_struct: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Greedy maximal matching: repeatedly commit the highest-probability remaining pair, then
    forbid every other pairing for both of its bases, so each base ends with at most one partner."""
    clean = np.copy(sec_struct)
    tmp = np.copy(probs)
    tmp[sec_struct < 1] = 0.0

    while np.sum(tmp > 0.0) > 0:
        i, j = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)
        tmp[i, :] = tmp[j, :] = 0.0
        clean[i, :] = clean[j, :] = 0
        tmp[:, i] = tmp[:, j] = 0.0
        clean[:, i] = clean[:, j] = 0
        clean[i, j] = clean[j, i] = 1

    return clean


def prob_mat_to_sec_struct(probs: np.ndarray, seq: str, threshold: float) -> np.ndarray:
    seq_len = probs.shape[-1]
    allowed = np.logical_not(np.eye(seq_len, dtype=bool))
    allowed = np.logical_and(allowed, ~_sharp_loop_mask(seq_len))
    allowed = np.logical_and(allowed, _canonical_pairs_mask(seq))

    probs = probs.copy()
    probs[~allowed] = 0.0
    sec_struct = np.greater(probs, threshold).astype(int)
    return _clean_sec_struct(sec_struct, probs)


# --- base-pair matrix -> extended dot-bracket string ---
# Each pair gets the first of 4 bracket types it does not cross; a pair fitting none is dropped and
# logged. RiNALMo has no such encoder.

_BRACKET_TYPES = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]


def _crosses(i1: int, j1: int, i2: int, j2: int) -> bool:
    return (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1)


def matrix_to_dot_bracket(sec_struct: np.ndarray) -> tuple[str, int]:
    seq_len = sec_struct.shape[0]
    pairs = [(i, j) for i in range(seq_len) for j in range(i + 1, seq_len) if sec_struct[i, j] > 0]

    assigned: list[list[tuple[int, int]]] = [[] for _ in _BRACKET_TYPES]
    dropped = 0
    for i, j in pairs:
        for bucket in assigned:
            if all(not _crosses(i, j, a, b) for a, b in bucket):
                bucket.append((i, j))
                break
        else:
            dropped += 1

    out = ["."] * seq_len
    for (open_c, close_c), bucket in zip(_BRACKET_TYPES, assigned):
        for i, j in bucket:
            out[i], out[j] = open_c, close_c

    return "".join(out), dropped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N query examples (for a quick check)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading RiNALMo (giga, secondary-structure head) onto {device}...")
    t0 = time.monotonic()
    lm, pred_head, threshold, config = load_giga_ss_model(device)
    alphabet = Alphabet(**config.alphabet)
    load_s = time.monotonic() - t0
    print(f"Loaded in {load_s:.1f}s. Tuned threshold from checkpoint: {threshold}")

    queries = pd.read_csv(QUERY_PATH, dtype={"sequence_id": str})
    if args.limit:
        queries = queries.head(args.limit)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_dropped = 0
    rows = []
    t0 = time.monotonic()
    with torch.no_grad():
        for _, row in queries.iterrows():
            seq = row["sequence"]
            tokens = torch.tensor([alphabet.encode(seq)], device=device)
            representation = lm(tokens)["representation"]
            logits = pred_head(representation[..., 1:-1, :]).squeeze(-1)
            probs = torch.sigmoid(logits)[0].cpu().numpy().astype(np.float64)

            sec_struct = prob_mat_to_sec_struct(probs, seq, threshold)
            structure, dropped = matrix_to_dot_bracket(sec_struct)
            total_dropped += dropped
            rows.append((row["sequence_id"], structure))
    inference_s = time.monotonic() - t0

    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "structure"])
        writer.writerows(rows)

    solution = pd.read_csv(SOLUTION_PATH, dtype={"sequence_id": str})
    submission = pd.DataFrame(rows, columns=["sequence_id", "structure"])
    merged = solution.merge(submission, on="sequence_id", suffixes=("_true", "_pred"))
    f1 = _sec_struct_f1(merged["structure_true"], merged["structure_pred"])

    print(f"Wrote {OUTPUT_PATH} ({len(rows)} rows)")
    if total_dropped:
        print(f"WARNING: {total_dropped} predicted base pairs didn't fit any of the 4 bracket types and were dropped")
    print(f"[timing] load={load_s:.1f}s inference={inference_s:.1f}s ({inference_s / max(len(rows), 1):.3f}s/seq)")
    print(f"[summary] n={len(merged)} sec_struct_f1={f1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

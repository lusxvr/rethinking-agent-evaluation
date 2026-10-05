"""Remap rinalmo_giga_ss_bprna_ft.pt for use_flash_attn=False, run once by download.sh.

The checkpoint was trained with flash attention, which packs Q/K/V into one Wqkv linear per block.
The non-flash module has separate to_q/to_k/to_v/out_proj one level deeper (mh_attn.mh_attn.*).
This splits each Wqkv into three chunks (Q, K, V order), renames the keys and overwrites
agent/weights/ in place. rotary_emb.inv_freq is never saved, so loaders still use strict=False.

Usage:
    cd models/rinalmo && uv run --project dev_env python convert_checkpoint.py
"""

import re
import sys
from pathlib import Path

import torch

WEIGHTS_PATH = Path(__file__).resolve().parent / "agent" / "weights" / "rinalmo_giga_ss_bprna_ft.pt"

_WQKV_RE = re.compile(r"^(lm\.transformer\.blocks\.\d+\.mh_attn\.)Wqkv\.(weight|bias)$")
_OUTPROJ_RE = re.compile(r"^(lm\.transformer\.blocks\.\d+\.mh_attn\.)out_proj\.(weight|bias)$")


def _remap(state_dict: dict) -> dict:
    remapped = {}
    for key, tensor in state_dict.items():
        m = _WQKV_RE.match(key)
        if m:
            prefix, kind = m.groups()
            q, k, v = tensor.chunk(3, dim=0)
            remapped[f"{prefix}mh_attn.to_q.{kind}"] = q
            remapped[f"{prefix}mh_attn.to_k.{kind}"] = k
            remapped[f"{prefix}mh_attn.to_v.{kind}"] = v
            continue
        m = _OUTPROJ_RE.match(key)
        if m:
            prefix, kind = m.groups()
            remapped[f"{prefix}mh_attn.out_proj.{kind}"] = tensor
            continue
        remapped[key] = tensor
    return remapped


def main() -> int:
    if not WEIGHTS_PATH.is_file():
        print(f"error: {WEIGHTS_PATH} not found -- run download.sh first")
        return 1

    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    if "threshold" not in state_dict:
        print("error: no 'threshold' key found -- checkpoint format looks unexpected, inspect before trusting this conversion")
        return 1
    threshold = state_dict.pop("threshold")

    n_wqkv = sum(1 for k in state_dict if _WQKV_RE.match(k))
    if n_wqkv == 0:
        print("Checkpoint has no Wqkv keys -- already converted (or its format changed); leaving it untouched.")
        return 0

    remapped = _remap(state_dict)
    remapped["threshold"] = threshold
    torch.save(remapped, WEIGHTS_PATH)
    print(f"Converted {n_wqkv} packed Wqkv layers to non-flash to_q/to_k/to_v naming, overwrote {WEIGHTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

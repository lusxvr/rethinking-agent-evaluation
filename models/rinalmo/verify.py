"""Sanity check that agent/weights load and run the secondary-structure head. Runs in dev_env.

Reimplements the loading part of RiNALMo's SecStructPredictionWrapper (train_sec_struct_prediction.py,
Apache 2.0), which is not in the installed package: "lm.*" backbone, "pred_head.*" head, "threshold".

Usage:
    cd models/rinalmo/dev_env && uv run --project . python ../verify.py
"""

import sys
from pathlib import Path

import torch

from rinalmo.config import model_config
from rinalmo.data.alphabet import Alphabet
from rinalmo.model.downstream import SecStructPredictionHead
from rinalmo.model.model import RiNALMo

WEIGHTS_PATH = Path(__file__).resolve().parent / "agent" / "weights" / "rinalmo_giga_ss_bprna_ft.pt"
NUM_RESNET_BLOCKS = 2  # SecStructPredictionWrapper's default in train_sec_struct_prediction.py


def load_giga_ss_model(device: str):
    config = model_config("giga")
    # Plain PyTorch attention; the flash_attn import is satisfied by flash_attn_stub/.
    config.model.transformer.use_flash_attn = False
    lm = RiNALMo(config)
    pred_head = SecStructPredictionHead(config.model.transformer.embed_dim, num_blocks=NUM_RESNET_BLOCKS)

    state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
    threshold = state_dict.pop("threshold")

    lm_state = {k[len("lm."):]: v for k, v in state_dict.items() if k.startswith("lm.")}
    head_state = {k[len("pred_head."):]: v for k, v in state_dict.items() if k.startswith("pred_head.")}

    # strict=False only for rotary_emb.inv_freq, a computed buffer; anything else missing fails.
    missing, unexpected = lm.load_state_dict(lm_state, strict=False)
    if unexpected or any(not k.endswith("rotary_emb.inv_freq") for k in missing):
        raise RuntimeError(f"unexpected load_state_dict mismatch: missing={missing}, unexpected={unexpected}")
    pred_head.load_state_dict(head_state)

    return lm.to(device).eval(), pred_head.to(device).eval(), threshold, config


def main() -> int:
    if not WEIGHTS_PATH.is_file():
        print(f"error: no checkpoint found at {WEIGHTS_PATH} -- run ./download.sh first")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading RiNALMo (giga, secondary-structure head) from {WEIGHTS_PATH} onto {device}...")
    lm, pred_head, threshold, config = load_giga_ss_model(device)

    n_params = sum(p.numel() for p in lm.parameters()) + sum(p.numel() for p in pred_head.parameters())
    print(f"loaded ok, params={n_params}, tuned threshold={threshold}")
    if n_params < 600_000_000:
        print(f"error: expected >600M params for the giga backbone + head, got {n_params}")
        return 1

    alphabet = Alphabet(**config.alphabet)
    sequences = [
        "GGCUAGUACGAGAGGACCUGGCCAGCUAGUCGACCUAGGCUAGUACGAGAGGACCUGGCCAGCUAGUCGACCUAGG",
        "ACGUACGUACGUACGUACGU",
    ]
    with torch.no_grad():
        for seq in sequences:
            tokens = torch.tensor([alphabet.encode(seq)], device=device)
            representation = lm(tokens)["representation"]
            logits = pred_head(representation[..., 1:-1, :]).squeeze(-1)
            probs = torch.sigmoid(logits)

            expected_shape = (1, len(seq), len(seq))
            if probs.shape != expected_shape:
                print(f"error: expected probs shape {expected_shape}, got {tuple(probs.shape)}")
                return 1
            if not torch.isfinite(probs).all():
                print("error: probs contain NaN/Inf")
                return 1
            if not torch.allclose(probs, probs.transpose(-1, -2), atol=1e-5):
                print("error: probability matrix isn't symmetric")
                return 1
            print(f"seq len={len(seq)}: probs shape={tuple(probs.shape)}, "
                  f"range=[{probs.min().item():.4f}, {probs.max().item():.4f}]")

    print("\nOK: model loaded and produced a well-formed, symmetric base-pair probability matrix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Sanity check that agent/weights load and produce well-formed embeddings. Runs in dev_env, not agent/env.

Usage:
    cd models/astroclip && uv run --project dev_env python verify.py
"""

import sys
from pathlib import Path

import torch
from astroclip.models import AstroClipModel

WEIGHTS_PATH = Path(__file__).resolve().parent / "agent" / "weights" / "astroclip.ckpt"


def main() -> int:
    if not WEIGHTS_PATH.is_file():
        print(f"error: no checkpoint found at {WEIGHTS_PATH} -- run ./download.sh first")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading AstroCLIP from {WEIGHTS_PATH} onto {device}...")
    # weights_only=False: required on torch>=2.6 -- this checkpoint pickles model classes.
    model = AstroClipModel.load_from_checkpoint(
        checkpoint_path=str(WEIGHTS_PATH), map_location=device, weights_only=False
    )
    model.eval()

    # Random inputs check that the env and checkpoint load, not correctness.
    # Batch >= 2: batch 1 crashes on an upstream bug.
    image_batch = torch.randn(2, 3, 144, 144, device=device)
    spectrum_batch = torch.randn(2, 7781, 1, device=device)

    with torch.no_grad():
        image_embedding = model(image_batch, input_type="image")
        spectrum_embedding = model(spectrum_batch, input_type="spectrum")

    for name, embedding in [("image", image_embedding), ("spectrum", spectrum_embedding)]:
        print(f"{name} embedding: shape={tuple(embedding.shape)} norm={embedding.norm(dim=-1)}")
        if embedding.shape != (2, 1024):
            print(f"error: expected shape (2, 1024), got {tuple(embedding.shape)}")
            return 1
        if not torch.isfinite(embedding).all():
            print(f"error: {name} embedding contains NaN/Inf")
            return 1

    print("\nOK: model loaded and produced well-formed embeddings for both modalities.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

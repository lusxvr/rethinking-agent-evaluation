"""Step 1 of 2: from EiffL/AstroCLIP (~43GB, cached), write the pre-embedded reference catalog
(~40 min) and a default stratified 200-galaxy query set. Then run build_final_query_set.py.

Usage:
    cd models/astroclip/dev_env
    uv run --project . python ../../../solutions/redshift-estimation/dev/prepare_data.py
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm
from astroclip.data.datamodule import AstroClipCollator
from astroclip.models import AstroClipModel
from datasets import load_dataset
from dotenv import load_dotenv
from images import extract_images

REPO_ROOT = Path(__file__).resolve().parents[3]

WEIGHTS_PATH = REPO_ROOT / "models" / "astroclip" / "agent" / "weights" / "astroclip.ckpt"
DATA_DIR = REPO_ROOT / "tasks" / "redshift-estimation" / "data"
SOLUTION_PATH = REPO_ROOT / "solutions" / "redshift-estimation" / "solution.csv"

N_QUERY = 200
N_BINS = 20  # N_QUERY / N_BINS = 10 galaxies sampled per redshift bin, evenly
N_QUERY_POOL = 20_000  # stratified-sampling pool; images are extracted only for the N_QUERY selected
N_REFERENCE = 146_907 - N_QUERY_POOL
RANDOM_SEED = 0
EMBED_BATCH_SIZE = 256


# ToRGB arcsinh stretch and 144x144 center crop, AstroCLIP's training preprocessing; without it,
# embeddings are near-random.
_COLLATOR = AstroClipCollator()


def _embed_images(model, device, images: np.ndarray) -> np.ndarray:
    """images: (N, 152, 152, 3) raw nanomaggie flux. Batch size 1 crashes on an upstream AstroCLIP bug."""
    embeddings = []
    for start in tqdm(range(0, len(images), EMBED_BATCH_SIZE)):
        chunk = images[start : start + EMBED_BATCH_SIZE]
        samples = [{"image": torch.from_numpy(img)} for img in chunk]
        tensor = _COLLATOR(samples)["image"].to(device)
        with torch.no_grad():
            embeddings.append(model(tensor, input_type="image").cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def _stratified_sample(redshift: np.ndarray, n_query: int, n_bins: int, rng: np.random.Generator) -> np.ndarray:
    """n_query positions sampled evenly across n_bins quantile redshift bins."""
    bin_edges = np.quantile(redshift, np.linspace(0, 1, n_bins + 1))
    bin_id = np.clip(np.digitize(redshift, bin_edges[1:-1]), 0, n_bins - 1)
    per_bin = n_query // n_bins
    selected = []
    for b in range(n_bins):
        candidates = np.where(bin_id == b)[0]
        selected.extend(rng.choice(candidates, size=per_bin, replace=False))
    return np.array(selected)


def main() -> int:
    if not WEIGHTS_PATH.is_file():
        print(f"error: no checkpoint found at {WEIGHTS_PATH} -- run models/astroclip/download.sh first")
        return 1

    if not load_dotenv(REPO_ROOT / ".env"):
        raise SystemExit(f"{REPO_ROOT / '.env'} not found -- run 'cp .env.example .env' and set CACHE_ROOT")
    os.environ.setdefault("HF_HOME", f"{os.environ['CACHE_ROOT']}/huggingface")
    if "HF_TOKEN" not in os.environ:
        print(
            f"warning: HF_TOKEN not set in {REPO_ROOT / '.env'} -- EiffL/AstroCLIP's first pull "
            "(~43GB) may be slow without it",
            file=sys.stderr,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading AstroCLIP from {WEIGHTS_PATH} onto {device}...")
    model = AstroClipModel.load_from_checkpoint(
        checkpoint_path=str(WEIGHTS_PATH), map_location=device, weights_only=False
    )
    model.eval()

    print("Loading EiffL/AstroCLIP (train split, cached locally after the first run)...")
    dataset = load_dataset("EiffL/AstroCLIP", split="train")
    if len(dataset) < N_REFERENCE + N_QUERY_POOL:
        print(f"error: only {len(dataset)} rows available, need {N_REFERENCE + N_QUERY_POOL}")
        return 1

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(dataset))
    # Sorted -- required for the chunk-local `.take()` in extract_images, and for
    # `table.column(...).take()` below to address strictly increasing offsets.
    reference_idx = np.sort(indices[:N_REFERENCE])
    query_pool_idx = np.sort(indices[N_REFERENCE : N_REFERENCE + N_QUERY_POOL])
    table = dataset.data.table

    print(f"Extracting + embedding {N_REFERENCE} reference images...")
    reference_images = extract_images(table, reference_idx, progress=tqdm)
    reference_embeddings = _embed_images(model, device, reference_images)
    reference_redshift = table.column("redshift").take(pa.array(reference_idx)).to_numpy()
    reference_targetid = table.column("targetid").take(pa.array(reference_idx)).to_numpy()

    print(f"Stratified-sampling {N_QUERY} query galaxies from a {N_QUERY_POOL}-galaxy candidate pool...")
    pool_redshift = table.column("redshift").take(pa.array(query_pool_idx)).to_numpy()
    pool_targetid = table.column("targetid").take(pa.array(query_pool_idx)).to_numpy()
    selected = _stratified_sample(pool_redshift, N_QUERY, N_BINS, rng)
    query_idx = query_pool_idx[selected]
    query_redshift = pool_redshift[selected]
    query_targetid = pool_targetid[selected]
    query_images = extract_images(table, np.sort(query_idx), progress=tqdm)
    # extract_images requires sorted indices -- recover the (redshift, targetid)-matching order.
    order = np.argsort(query_idx)
    query_redshift, query_targetid = query_redshift[order], query_targetid[order]

    query_images_dir = DATA_DIR / "query_images"
    query_images_dir.mkdir(parents=True, exist_ok=True)
    SOLUTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing {N_QUERY} query images + solution.csv...")
    with SOLUTION_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["targetid", "redshift"])
        for targetid, image, redshift in zip(query_targetid, query_images, query_redshift):
            np.save(query_images_dir / f"{targetid}.npy", image)
            writer.writerow([targetid, redshift])

    reference_path = DATA_DIR / "reference_catalog.npz"
    np.savez(
        reference_path,
        embeddings=reference_embeddings,
        redshift=reference_redshift,
        targetid=reference_targetid,
    )

    print(f"Wrote {query_images_dir} ({N_QUERY} images)")
    print(f"Wrote {reference_path}")
    print(f"Wrote {SOLUTION_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

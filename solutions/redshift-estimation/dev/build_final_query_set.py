"""Step 2 of 2: rebuild the fixed 20-galaxy query set from its targetid list. The list was chosen
from 2,000 random draws for a k-NN R2 near 0.75, since a blind n=20 draw varies too much.

Usage:
    cd models/astroclip/dev_env
    uv run --project . python ../../../solutions/redshift-estimation/dev/build_final_query_set.py
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
from datasets import load_dataset
from dotenv import load_dotenv
from images import extract_images

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "tasks" / "redshift-estimation" / "data"
SOLUTION_PATH = REPO_ROOT / "solutions" / "redshift-estimation" / "solution.csv"

# Deliberately chosen (see module docstring) -- not a blind sample.
CHOSEN_TARGETIDS = [
    39632951830384900, 39633497622577569, 39633052695003442, 39633352306721780,
    39632956892905954, 39633271444735400, 39633332815790714, 39633271440542285,
    39633297331979989, 39632966913102567, 39633123121561793, 39633307859683679,
    39632930229719071, 39633314771897973, 39633419147149818, 39633240125868327,
    39633071883945216, 39633158332746554, 39632950165244827, 39633158391465446,
]


def main() -> int:
    if not load_dotenv(REPO_ROOT / ".env"):
        raise SystemExit(f"{REPO_ROOT / '.env'} not found -- run 'cp .env.example .env' and set CACHE_ROOT")
    os.environ.setdefault("HF_HOME", f"{os.environ['CACHE_ROOT']}/huggingface")

    print("Loading EiffL/AstroCLIP (train split, cached locally after prepare_data.py's first run)...")
    dataset = load_dataset("EiffL/AstroCLIP", split="train")
    table = dataset.data.table

    all_targetid = table.column("targetid").to_numpy()
    chosen = np.array(CHOSEN_TARGETIDS, dtype=np.int64)
    row_idx = np.array([np.where(all_targetid == t)[0][0] for t in chosen])
    order = np.argsort(row_idx)
    row_idx_sorted = row_idx[order]  # extract_images requires sorted indices

    images = extract_images(table, row_idx_sorted)
    redshift = table.column("redshift").take(pa.array(row_idx_sorted)).to_numpy()
    targetid = table.column("targetid").take(pa.array(row_idx_sorted)).to_numpy()

    query_images_dir = DATA_DIR / "query_images"
    query_images_dir.mkdir(parents=True, exist_ok=True)
    for old_file in query_images_dir.glob("*.npy"):
        old_file.unlink()

    with SOLUTION_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["targetid", "redshift"])
        for t, image, z in zip(targetid, images, redshift):
            np.save(query_images_dir / f"{t}.npy", image)
            writer.writerow([t, z])

    print(f"Wrote {len(targetid)} query images to {query_images_dir}")
    print(f"Wrote {SOLUTION_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

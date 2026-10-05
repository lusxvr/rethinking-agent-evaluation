"""Download GUE prom_core_tata (leannmlindsey/GUE) and write train.csv (4,904 labeled sequences),
query_sequences.csv (613 test sequences) and solution.csv (sequence_id,label).

Usage:
    cd models/dnabert-2/dev_env
    uv run --project . python ../../../solutions/promoter-prediction/dev/prepare_data.py
"""

import csv
import os
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "tasks" / "promoter-prediction" / "data"
SOLUTION_PATH = REPO_ROOT / "solutions" / "promoter-prediction" / "solution.csv"

GUE_CONFIG = "prom_core_tata"


def main() -> int:
    if not load_dotenv(REPO_ROOT / ".env"):
        raise SystemExit(f"{REPO_ROOT / '.env'} not found -- run 'cp .env.example .env' and set CACHE_ROOT")
    os.environ.setdefault("HF_HOME", f"{os.environ['CACHE_ROOT']}/huggingface")

    ds = load_dataset("leannmlindsey/GUE", GUE_CONFIG)
    train, test = ds["train"], ds["test"]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    with (DATA_DIR / "train.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "sequence", "label"])
        for i, row in enumerate(train):
            writer.writerow([f"train_{i:04d}", row["sequence"], row["label"]])

    with (DATA_DIR / "query_sequences.csv").open("w", newline="") as qf, SOLUTION_PATH.open("w", newline="") as sf:
        q_writer = csv.writer(qf)
        s_writer = csv.writer(sf)
        q_writer.writerow(["sequence_id", "sequence"])
        s_writer.writerow(["sequence_id", "label"])
        for i, row in enumerate(test):
            sequence_id = f"query_{i:04d}"
            q_writer.writerow([sequence_id, row["sequence"]])
            s_writer.writerow([sequence_id, row["label"]])

    print(f"Wrote {DATA_DIR / 'train.csv'} ({len(train)} rows)")
    print(f"Wrote {DATA_DIR / 'query_sequences.csv'} and {SOLUTION_PATH} ({len(test)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

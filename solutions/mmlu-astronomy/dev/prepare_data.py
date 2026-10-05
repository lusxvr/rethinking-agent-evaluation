"""Download the MMLU astronomy test split (cais/mmlu) and write tasks/mmlu-astronomy/data/questions.csv
and solution.csv (question_id,answer).

Usage:
    cd models/astrosage/dev_env
    uv run --project . python ../../../solutions/mmlu-astronomy/dev/prepare_data.py
"""

import csv
import os
import sys
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "tasks" / "mmlu-astronomy" / "data"
SOLUTION_PATH = REPO_ROOT / "solutions" / "mmlu-astronomy" / "solution.csv"

LETTERS = ["A", "B", "C", "D"]


def main() -> int:
    if not load_dotenv(REPO_ROOT / ".env"):
        raise SystemExit(f"{REPO_ROOT / '.env'} not found -- run 'cp .env.example .env' and set CACHE_ROOT")
    os.environ.setdefault("HF_HOME", f"{os.environ['CACHE_ROOT']}/huggingface")

    test = load_dataset("cais/mmlu", "astronomy", split="test")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    with (DATA_DIR / "questions.csv").open("w", newline="") as qf, SOLUTION_PATH.open("w", newline="") as sf:
        q_writer = csv.writer(qf)
        s_writer = csv.writer(sf)
        q_writer.writerow(["question_id", "question", "choice_a", "choice_b", "choice_c", "choice_d"])
        s_writer.writerow(["question_id", "answer"])
        for i, row in enumerate(test):
            q_writer.writerow([i, row["question"], *row["choices"]])
            s_writer.writerow([i, LETTERS[row["answer"]]])

    print(f"Wrote {DATA_DIR / 'questions.csv'} and {SOLUTION_PATH} ({len(test)} questions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

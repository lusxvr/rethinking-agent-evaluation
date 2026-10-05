"""Download SPOT-RNA's bpRNA dataset (RiNALMo's TR0/VL0/TS0 split; cached under $CACHE_ROOT) and
write train.csv (300 VL0 examples for sanity checks), query_sequences.csv (300 TS0 sequences)
and solution.csv (sequence_id,structure). Structures are bpRNA's extended dot-bracket strings.

Usage:
    cd models/rinalmo/dev_env
    uv run --project . python ../../../solutions/rna-folding/dev/prepare_data.py
"""

import csv
import os
import random
import sys
from pathlib import Path
from zipfile import ZipFile

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "tasks" / "rna-folding" / "data"
SOLUTION_PATH = REPO_ROOT / "solutions" / "rna-folding" / "solution.csv"

BPRNA_URL = "https://dl.dropboxusercontent.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip"
N_TRAIN_EXAMPLES = 300
# Subsample TS0 (1,305 sequences): unbatched inference over the full split takes a large part of
# the 300s budget.
N_TEST_EXAMPLES = 300
RANDOM_SEED = 0


def _download_and_extract(cache_dir: Path) -> Path:
    extracted_root = cache_dir / "bpRNA_dataset"
    if extracted_root.is_dir():
        return extracted_root

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "bpRNA_dataset.zip"
    print(f"Downloading bpRNA dataset from {BPRNA_URL}...")
    with requests.get(BPRNA_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with zip_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    with ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    zip_path.unlink()

    return extracted_root


def _parse_st_file(path: Path) -> tuple[str, str]:
    """Sequence and dot-bracket structure: the first two non-'#' lines of a bpRNA .st file."""
    lines = [line.rstrip("\n") for line in path.read_text().splitlines() if not line.lstrip().startswith("#")]
    sequence, structure = lines[0], lines[1]
    if len(sequence) != len(structure):
        raise ValueError(f"{path}: sequence/structure length mismatch ({len(sequence)} vs {len(structure)})")
    return sequence, structure


def main() -> int:
    if not load_dotenv(REPO_ROOT / ".env"):
        raise SystemExit(f"{REPO_ROOT / '.env'} not found -- run 'cp .env.example .env' and set CACHE_ROOT")
    cache_dir = Path(os.environ["CACHE_ROOT"]) / "bprna_dataset"

    bprna_root = _download_and_extract(cache_dir)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SOLUTION_PATH.parent.mkdir(parents=True, exist_ok=True)

    val_files = sorted((bprna_root / "VL0").glob("*.st"))
    random.Random(RANDOM_SEED).shuffle(val_files)
    train_files = val_files[:N_TRAIN_EXAMPLES]

    with (DATA_DIR / "train.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "sequence", "structure"])
        for i, path in enumerate(train_files):
            sequence, structure = _parse_st_file(path)
            writer.writerow([f"train_{i:04d}", sequence, structure])

    test_files = sorted((bprna_root / "TS0").glob("*.st"))
    random.Random(RANDOM_SEED + 1).shuffle(test_files)  # +1: an independent shuffle from train's, same dataset root
    test_files = sorted(test_files[:N_TEST_EXAMPLES])
    with (DATA_DIR / "query_sequences.csv").open("w", newline="") as qf, SOLUTION_PATH.open("w", newline="") as sf:
        q_writer = csv.writer(qf)
        s_writer = csv.writer(sf)
        q_writer.writerow(["sequence_id", "sequence"])
        s_writer.writerow(["sequence_id", "structure"])
        for i, path in enumerate(test_files):
            sequence, structure = _parse_st_file(path)
            sequence_id = f"query_{i:04d}"
            q_writer.writerow([sequence_id, sequence])
            s_writer.writerow([sequence_id, structure])

    print(f"Wrote {DATA_DIR / 'train.csv'} ({len(train_files)} rows)")
    print(f"Wrote {DATA_DIR / 'query_sequences.csv'} and {SOLUTION_PATH} ({len(test_files)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

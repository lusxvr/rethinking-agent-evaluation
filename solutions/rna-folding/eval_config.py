"""Grading rules for rna-folding. Loaded dynamically by eval/evaluate.py."""

from pathlib import Path

SOLUTION_FILE = Path(__file__).parent / "solution.csv"

ID_COLUMN = "sequence_id"
TARGET_COLUMN = "structure"
METRIC = "sec_struct_f1"
TARGET_TYPE = "categorical"  # dot-bracket strings; no VALID_TARGET_VALUES
HIGHER_IS_BETTER = True
EXPECTED_ROWS = 300

# REFERENCE: dev/run_reference.py, fine-tuned RiNALMo giga, inference only, on 300 bpRNA TS0 test
# examples (full 1,305-example split: 0.7480, paper 0.747; deterministic).
# BACKBONE: dev/run_backbone.py, Qwen3.5-35B-A3B-FP8 with its sampling preset: 0.0 with and without
# thinking (mostly malformed outputs; every thinking response hit the 16,384-token cap).
REFERENCE = 0.7350
BACKBONE = 0.0000
TRIVIAL = 0.0  # e.g. an all-unpaired prediction
# Sized against the anchor gap (0.735).
MARGIN = 0.10

# The other mounted models are distractors.
DOMAIN_MODEL = "rinalmo"

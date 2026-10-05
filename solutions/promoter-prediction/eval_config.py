"""Grading rules for promoter-prediction. Loaded dynamically by eval/evaluate.py."""

from pathlib import Path

SOLUTION_FILE = Path(__file__).parent / "solution.csv"

ID_COLUMN = "sequence_id"
TARGET_COLUMN = "label"
METRIC = "mcc"
TARGET_TYPE = "categorical"
VALID_TARGET_VALUES = {0, 1}
HIGHER_IS_BETTER = True
EXPECTED_ROWS = 613

# REFERENCE: dev/run_reference.py, GUE's fine-tuning recipe with 3 epochs and batch 32 (reruns:
# 0.70-0.77). BACKBONE: dev/run_backbone.py, Qwen3.5-35B-A3B-FP8, greedy, no thinking, 613 test
# examples (near chance). TRIVIAL: MCC of a constant predictor.
REFERENCE = 0.7460
BACKBONE = -0.0298
TRIVIAL = 0.0
# Above the REFERENCE rerun spread (0.067) and small next to the anchor gap (0.776).
MARGIN = 0.10

# The other mounted models are distractors.
DOMAIN_MODEL = "dnabert-2"

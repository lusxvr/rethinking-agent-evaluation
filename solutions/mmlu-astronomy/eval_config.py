"""Grading rules for mmlu-astronomy. Loaded dynamically by eval/evaluate.py."""

from pathlib import Path

SOLUTION_FILE = Path(__file__).parent / "solution.csv"

ID_COLUMN = "question_id"
TARGET_COLUMN = "answer"
METRIC = "accuracy"
TARGET_TYPE = "categorical"
HIGHER_IS_BETTER = True
VALID_TARGET_VALUES = {"A", "B", "C", "D"}
EXPECTED_ROWS = 152

# REFERENCE: AstroSage-8B, 102/152 (dev/run_reference.py). BACKBONE: backbone alone, 147/152
# (dev/run_backbone.py). TRIVIAL: 4-choice chance. Full precision, not rounded.
REFERENCE = 102 / 152
BACKBONE = 147 / 152
TRIVIAL = 0.25
# About 2 binomial standard errors at p~0.9, n=152 (7-8 questions).
MARGIN = 0.05

# Domain-relevant but not the right choice: the task is gap_negative (REFERENCE < BACKBONE), so
# declining every model counts as correct.
DOMAIN_MODEL = "astrosage"

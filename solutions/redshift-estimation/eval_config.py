"""Grading rules for redshift-estimation. Loaded dynamically by eval/evaluate.py."""

from pathlib import Path

SOLUTION_FILE = Path(__file__).parent / "solution.csv"

ID_COLUMN = "targetid"
TARGET_COLUMN = "redshift"
METRIC = "r2"
TARGET_TYPE = "continuous"
HIGHER_IS_BETTER = True
EXPECTED_ROWS = 20

# REFERENCE: AstroCLIP zero-shot k-NN (dev/run_reference.py). BACKBONE: backbone without thinking
# (dev/run_backbone.py default; thinking is unstable at n=20). TRIVIAL: mean predictor (R2 = 0).
REFERENCE = 0.7533688140899595
BACKBONE = -2.7011497849205184
TRIVIAL = 0.0
# Lower bound on sampling noise (std ~0.22 on 200 queries; this set has 20).
MARGIN = 0.25

# Counted as the correct choice only where REFERENCE beats BACKBONE. astrosage is a distractor.
DOMAIN_MODEL = "astroclip"

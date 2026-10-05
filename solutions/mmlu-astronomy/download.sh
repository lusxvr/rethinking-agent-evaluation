#!/usr/bin/env bash
# Build tasks/mmlu-astronomy/data/questions.csv and solution.csv from cais/mmlu (astronomy), via
# models/astrosage/dev_env. prepare_data.py downloads the raw dataset (cached under $HF_HOME) and writes only the redacted
# task data, since the raw data contains the answers. Takes seconds.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# Relative path, since git may be missing on compute nodes.
REPO_ROOT="$(cd ../.. && pwd)"

cd "$REPO_ROOT/models/astrosage/dev_env"
uv run --project . python "$REPO_ROOT/solutions/mmlu-astronomy/dev/prepare_data.py"

echo "Wrote $REPO_ROOT/tasks/mmlu-astronomy/data and $REPO_ROOT/solutions/mmlu-astronomy/solution.csv"

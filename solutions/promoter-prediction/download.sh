#!/usr/bin/env bash
# Build tasks/promoter-prediction/data/{train.csv,query_sequences.csv} and solution.csv from
# leannmlindsey/GUE (prom_core_tata), via models/dnabert-2/dev_env. Downloads and redacts in one
# step, since the raw data contains the labels. Takes seconds.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd ../.. && pwd)"

cd "$REPO_ROOT/models/dnabert-2/dev_env"
uv run --project . python "$REPO_ROOT/solutions/promoter-prediction/dev/prepare_data.py"

echo "Wrote $REPO_ROOT/tasks/promoter-prediction/data and $REPO_ROOT/solutions/promoter-prediction/solution.csv"

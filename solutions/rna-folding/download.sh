#!/usr/bin/env bash
# Build tasks/rna-folding/data/{train.csv,query_sequences.csv} and solution.csv from SPOT-RNA's
# bpRNA dataset (TR0/VL0/TS0), via models/rinalmo/dev_env. prepare_data.py caches the ~10.5MB zip
# under $CACHE_ROOT/bprna_dataset/.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd ../.. && pwd)"

cd "$REPO_ROOT/models/rinalmo/dev_env"
uv run --project . python "$REPO_ROOT/solutions/rna-folding/dev/prepare_data.py"

echo "Wrote $REPO_ROOT/tasks/rna-folding/data and $REPO_ROOT/solutions/rna-folding/solution.csv"

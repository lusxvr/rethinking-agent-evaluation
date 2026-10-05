#!/usr/bin/env bash
# Build tasks/redshift-estimation/data/ (reference_catalog.npz, query_images/) and solution.csv from
# EiffL/AstroCLIP (~43GB, cached). Needs AstroCLIP's weights and dev_env (models/astroclip/README.md),
# since prepare_data.py embeds 126,907 images (~40 min on one GPU). Set CUDA_VISIBLE_DEVICES first.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
# Relative path, since git may be missing on compute nodes.
REPO_ROOT="$(cd ../.. && pwd)"

# Load .env, so load_dataset uses HF_TOKEN and the HF_HOME cache instead of ~/.cache.
ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${CACHE_ROOT:?not set in $ENV_FILE}"
export HF_HOME="$CACHE_ROOT/huggingface"
# Disable xet (many small temp writes).
export HF_HUB_DISABLE_XET=1

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "warning: CUDA_VISIBLE_DEVICES not set -- check 'nvidia-smi -L' for a free GPU first" >&2
fi

cd "$REPO_ROOT/models/astroclip/dev_env"
uv run --project . python "$REPO_ROOT/solutions/redshift-estimation/dev/prepare_data.py"
uv run --project . python "$REPO_ROOT/solutions/redshift-estimation/dev/build_final_query_set.py"

echo "Wrote $REPO_ROOT/tasks/redshift-estimation/data and $REPO_ROOT/solutions/redshift-estimation/solution.csv"

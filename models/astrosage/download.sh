#!/usr/bin/env bash
# Downloads the AstroSage-8B checkpoint into agent/weights/ (gitignored -- this script is how to
# reproduce it).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# CACHE_ROOT comes from the repo's .env (relative path, since git may be missing on compute nodes).
ENV_FILE="$(cd ../.. && pwd)/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
export $(grep -v '^#' "$ENV_FILE" | xargs)

export HF_HOME="$CACHE_ROOT/huggingface"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

uv run --with "huggingface_hub[cli]" hf download AstroMLab/AstroSage-8B \
  --local-dir agent/weights

echo "Downloaded AstroMLab/AstroSage-8B into $(pwd)/agent/weights"

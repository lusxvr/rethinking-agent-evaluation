#!/usr/bin/env bash
# Downloads DNABERT-2 (weights + its custom modeling code, since it's loaded with
# trust_remote_code=True) into agent/weights/ (gitignored -- this script is how to reproduce it).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# CACHE_ROOT comes from .env at the repo root -- run 'cp .env.example .env' there if missing.
ENV_FILE="$(cd ../.. && pwd)/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
export $(grep -v '^#' "$ENV_FILE" | xargs)

export HF_HOME="$CACHE_ROOT/huggingface"
export UV_CACHE_DIR="$CACHE_ROOT/uv"

# Pinned revision, since trust_remote_code runs the downloaded modeling code.
REVISION="7bce263b15377fc15361f52cfab88f8b586abda0"

uv run --with "huggingface_hub[cli]" hf download zhihan1996/DNABERT-2-117M \
  --revision "$REVISION" --local-dir agent/weights

# Replace the vendored Triton kernel, which crashes whenever triton is importable, with an empty
# file: bert_layers.py then falls back to plain PyTorch attention. Deleting it fails, because
# transformers copies every imported file before running bert_layers.py.
: > agent/weights/flash_attn_triton.py

echo "Downloaded zhihan1996/DNABERT-2-117M (revision $REVISION) into $(pwd)/agent/weights"

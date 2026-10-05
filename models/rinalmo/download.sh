#!/usr/bin/env bash
# Download RiNALMo giga fine-tuned on bpRNA (rinalmo_giga_ss_bprna_ft.pt, ~2.6GB, a flat state dict)
# into agent/weights/ from Zenodo record 15043668 (CC BY 4.0). Not the multimolecule HF mirror,
# which is AGPL-3.0.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# CACHE_ROOT comes from .env at the repo root -- run 'cp .env.example .env' there if missing.
ENV_FILE="$(cd ../.. && pwd)/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
export $(grep -v '^#' "$ENV_FILE" | xargs)
export UV_CACHE_DIR="$CACHE_ROOT/uv"

mkdir -p agent/weights

URL="https://zenodo.org/records/15043668/files/rinalmo_giga_ss_bprna_ft.pt"
# md5 from Zenodo's record metadata; guards against corrupted downloads.
EXPECTED_MD5="3688b049ac282fa2088fcd80854e34dd"
OUT="agent/weights/rinalmo_giga_ss_bprna_ft.pt"

curl -fL --retry 3 -o "$OUT" "$URL"

ACTUAL_MD5="$(md5sum "$OUT" | cut -d' ' -f1)"
if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
  echo "error: md5 mismatch for $OUT (expected $EXPECTED_MD5, got $ACTUAL_MD5) -- re-run this script" >&2
  rm -f "$OUT"
  exit 1
fi

echo "Downloaded rinalmo_giga_ss_bprna_ft.pt (md5 verified) into $(pwd)/agent/weights"

# Remap the packed Wqkv weights for use_flash_attn=False (see convert_checkpoint.py).
uv run --project dev_env python convert_checkpoint.py

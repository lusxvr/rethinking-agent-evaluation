#!/usr/bin/env bash
# Start the vLLM server in the background on a node without a scheduler. Under Slurm, use
# slurm/vllm_server.sbatch.
#
# Usage: run_vllm.sh [tier]   tier is an axes.py MODEL_LEVELS key, default qwen35-35b-a3b-fp8.
set -euo pipefail

# Relative to this script, since git may be missing on compute nodes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${CACHE_ROOT:?not set in $ENV_FILE -- see .env.example}"

mkdir -p "$CACHE_ROOT"
LOG_FILE="$CACHE_ROOT/vllm-server.log"

nohup "$(dirname "${BASH_SOURCE[0]}")/serve.sh" "$@" > "$LOG_FILE" 2>&1 &
echo "Started vLLM server (PID $!). Logs: $LOG_FILE"
echo "Once ready, check: curl -s localhost:${VLLM_PORT:-61000}/v1/models | python3 -m json.tool"

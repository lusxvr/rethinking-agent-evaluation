#!/usr/bin/env bash
#SBATCH --job-name=agent-worker
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=40G
#SBATCH --time=04:00:00  # up to 3h server wait plus the cell budget
#SBATCH --output=slurm/logs/debug/agent-worker-%A_%a.out
# Run a tier's grid cells as a Slurm array; the %throttle caps concurrency to the server's
# capacity. Submitted by scripts/production/run_grid_slurm.py. No --requeue: a rerun into the same
# run directory would interleave trace.jsonl.
#
# Usage: sbatch --array=1-<N>%<throttle> slurm/agent_worker_array.sbatch <manifest-path> <grid-dir> <task> <server-job-id>
#   manifest line (tab-separated): information harness verification budget model replicate
#   <server-job-id>: the vllm_server.sbatch job, or "none" for a hosted-API tier.
#   AGENT_ORACLE=1 in the job environment grants oracle_check to every cell.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?not set -- run sbatch from the repo root}"
cd "$REPO_ROOT"
# uv location; batch jobs may not source ~/.bashrc.
export PATH="$HOME/.local/bin:$PATH"

# Rootless cgroups need XDG_RUNTIME_DIR.
export XDG_RUNTIME_DIR="/tmp/xdg-runtime-${SLURM_JOB_ID}"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${CACHE_ROOT:?not set in $ENV_FILE}"

CLUSTER="${CLUSTER:-example}"
CLUSTER_ENV_FILE="$REPO_ROOT/slurm/clusters/$CLUSTER.env"
[ -f "$CLUSTER_ENV_FILE" ] || { echo "error: no cluster profile at $CLUSTER_ENV_FILE (CLUSTER=$CLUSTER from .env)" >&2; exit 1; }
set -a; . "$CLUSTER_ENV_FILE"; set +a

if [ "$#" -ne 4 ]; then
  echo "usage: sbatch --array=1-N%throttle slurm/agent_worker_array.sbatch <manifest-path> <grid-dir> <task> <server-job-id>" >&2
  exit 1
fi
MANIFEST="$1"; GRID_DIR="$2"; TASK="$3"; SERVER_JOB="$4"
: "${SLURM_ARRAY_TASK_ID:?this script must be run as a Slurm array (sbatch --array=...)}"

LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")"
[ -n "$LINE" ] || { echo "error: manifest line $SLURM_ARRAY_TASK_ID is empty or missing ($MANIFEST)" >&2; exit 1; }
IFS=$'\t' read -r INFORMATION HARNESS VERIFICATION BUDGET MODEL REPLICATE <<< "$LINE"

# Hosted-API tier: no server to wait for; run_cell still requires a --base-url.
if [ "$SERVER_JOB" = "none" ]; then
  BASE_URL="unused"
  echo "[task $SLURM_ARRAY_TASK_ID] tier has no vLLM server (hosted-API model) -- skipping ready-file wait"
else
  ADDR_FILE="$CACHE_ROOT/vllm-server-${SERVER_JOB}.addr"
  echo "[task $SLURM_ARRAY_TASK_ID] waiting for vLLM server ready-file at $ADDR_FILE..."
  attempts=1080  # ~3h at 10s, the server startup window
  ready=0
  for _ in $(seq 1 "$attempts"); do
    if [ -s "$ADDR_FILE" ]; then
      ready=1
      break
    fi
    sleep 10
  done
  [ "$ready" -eq 1 ] || { echo "error: no ready vLLM server after ${attempts}x10s ($ADDR_FILE never appeared)" >&2; exit 1; }
  BASE_URL="http://$(cat "$ADDR_FILE")/v1"
fi

GPU_UUID="$(nvidia-smi -L | head -1 | grep -oP '(?<=UUID: )[^)]+')"
[ -n "$GPU_UUID" ] || { echo "error: could not resolve this job's GPU UUID from nvidia-smi -L" >&2; exit 1; }
echo "[task $SLURM_ARRAY_TASK_ID] cell: information=$INFORMATION harness=$HARNESS verification=$VERIFICATION budget=$BUDGET model=$MODEL replicate=$REPLICATE"
echo "[task $SLURM_ARRAY_TASK_ID] gpu=$GPU_UUID backbone=$BASE_URL"

# --no-sync: concurrent uv runs otherwise queue on the shared uv cache lock.
# --no-verbose: trace.jsonl keeps the full record.
# AGENT_ORACLE is set per grid by run_grid_slurm.py from the spec's 'oracle' key.
ORACLE_ARGS=()
if [ "${AGENT_ORACLE:-0}" = "1" ]; then ORACLE_ARGS+=(--oracle); fi
uv run --no-sync python -m scripts.production.run_cell \
  --grid-dir "$GRID_DIR" --task "$TASK" \
  --information "$INFORMATION" --harness "$HARNESS" --verification "$VERIFICATION" \
  --budget "$BUDGET" --model "$MODEL" --replicate "$REPLICATE" \
  --gpu "$GPU_UUID" --base-url "$BASE_URL" --no-verbose "${ORACLE_ARGS[@]}"

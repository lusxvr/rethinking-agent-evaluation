#!/usr/bin/env bash
#SBATCH --job-name=vllm-server
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/debug/vllm-server-%j.out
# Serve one model tier (apptainer/vllm/serve.sh) for the job's lifetime and publish host:port to
# $CACHE_ROOT/vllm-server-<job id>.addr once ready. Port and address file are keyed on the job id.
#
# Usage: sbatch slurm/vllm_server.sbatch <tier>
#   The #SBATCH defaults fit the smallest tier. run_grid_slurm.py passes partition, QoS and the
#   tier's resources from the cluster profile; pass them by hand when submitting directly.
set -euo pipefail
# Slurm runs a spooled copy of this script, so use the submit dir (the repo root).
REPO_ROOT="${SLURM_SUBMIT_DIR:?not set -- run sbatch from the repo root}"
cd "$REPO_ROOT"
ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${CACHE_ROOT:?not set in $ENV_FILE}"

# Only the port range is needed from the cluster profile here.
CLUSTER="${CLUSTER:-example}"
CLUSTER_ENV_FILE="$REPO_ROOT/slurm/clusters/$CLUSTER.env"
[ -f "$CLUSTER_ENV_FILE" ] || { echo "error: no cluster profile at $CLUSTER_ENV_FILE (CLUSTER=$CLUSTER from .env)" >&2; exit 1; }
set -a; . "$CLUSTER_ENV_FILE"; set +a

TIER="${1:-qwen35-35b-a3b-fp8}"
: "${SLURM_JOB_ID:?not set -- this script must be run under sbatch}"
ADDR_FILE="$CACHE_ROOT/vllm-server-${SLURM_JOB_ID}.addr"
mkdir -p "$CACHE_ROOT"

cleanup() { rm -f "$ADDR_FILE"; }
trap cleanup EXIT TERM INT
rm -f "$ADDR_FILE"

# Port from the job id within the profile's port range.
PORT_RANGE_START="${VLLM_PORT_RANGE_START:-60001}"
PORT_RANGE_SIZE="${VLLM_PORT_RANGE_SIZE:-3000}"
# Exported so serve.sh uses the same port as the readiness check.
export VLLM_PORT="${VLLM_PORT:-$((PORT_RANGE_START + SLURM_JOB_ID % PORT_RANGE_SIZE))}"

apptainer/vllm/serve.sh "$TIER" &
SERVER_PID=$!

# Poll the server in the background and publish its address once it answers.
(
  # 3h default: a cold start (image pull plus loading the largest weights) can take about 2h.
  # If this loop gives up, no address file is written even if the server comes up later.
  attempts=$(( ${VLLM_READY_TIMEOUT_S:-10800} / 10 ))
  for _ in $(seq 1 "$attempts"); do
    if curl -sf "http://localhost:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
      tmp="$(mktemp "$ADDR_FILE.XXXXXX")"
      echo "$(hostname):$VLLM_PORT" > "$tmp"
      mv -f "$tmp" "$ADDR_FILE"
      echo "vLLM ready at $(cat "$ADDR_FILE")"
      exit 0
    fi
    sleep 10
  done
  echo "error: vLLM did not become ready within ${attempts}x10s -- check slurm/logs/debug/vllm-server-${SLURM_JOB_ID}.out (or the grid's own slurm/logs/<grid-dir>/ if run via run_grid_slurm.py, which overrides --output)" >&2
) &

wait "$SERVER_PID"

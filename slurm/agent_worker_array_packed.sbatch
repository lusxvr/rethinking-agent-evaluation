#!/usr/bin/env bash
#SBATCH --job-name=agent-worker-packed
#SBATCH --nodes=1
#SBATCH --time=04:00:00  # up to 3h server wait plus the cell budget
#SBATCH --output=slurm/logs/debug/agent-worker-packed-%A_%a.out
# Whole-node variant of agent_worker_array.sbatch for clusters with WHOLE_NODE_ONLY=1.
# Each array task takes one node and runs up to <group-size> manifest lines concurrently, one GPU
# each, via srun --exclusive. Task k gets lines (k-1)*group-size+1 .. k*group-size.
# run_grid_slurm.py supplies --partition/--qos/--account/--gres/--nodes from the cluster profile.
# <server-job-id> is "none" for a hosted-API tier.
#
# Usage: sbatch --array=1-<n_nodes>%<node-throttle> --gres=<whole-node gres string> \
#   --partition=... --qos=... --account=... --nodes=1 [--comment=...] \
#   slurm/agent_worker_array_packed.sbatch <manifest-path> <grid-dir> <task> <server-job-id> \
#   <group-size> <cpus-per-slot>
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

if [ "$#" -ne 6 ]; then
  echo "usage: sbatch --array=1-N%throttle --gres=... slurm/agent_worker_array_packed.sbatch <manifest-path> <grid-dir> <task> <server-job-id> <group-size> <cpus-per-slot>" >&2
  exit 1
fi
MANIFEST="$1"; GRID_DIR="$2"; TASK="$3"; SERVER_JOB="$4"; GROUP_SIZE="$5"; CPUS_PER_SLOT="$6"
: "${SLURM_ARRAY_TASK_ID:?this script must be run as a Slurm array (sbatch --array=...)}"

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
  echo "[task $SLURM_ARRAY_TASK_ID] using backbone at $BASE_URL"
fi

FIRST_LINE=$(( (SLURM_ARRAY_TASK_ID - 1) * GROUP_SIZE + 1 ))
LAST_LINE=$(( FIRST_LINE + GROUP_SIZE - 1 ))
TOTAL_LINES=$(wc -l < "$MANIFEST")

# One background srun step per line. --gres=gpu:1 may not restrict GPU visibility, so each slot
# picks GPU index (line_no - FIRST_LINE) and passes its UUID to the container.
pids=()
for line_no in $(seq "$FIRST_LINE" "$LAST_LINE"); do
  if [ "$line_no" -gt "$TOTAL_LINES" ]; then
    break  # last group may be partial
  fi
  LINE="$(sed -n "${line_no}p" "$MANIFEST")"
  [ -n "$LINE" ] || continue
  IFS=$'\t' read -r INFORMATION HARNESS VERIFICATION BUDGET MODEL REPLICATE <<< "$LINE"
  SLOT_IDX=$((line_no - FIRST_LINE))
  echo "[task $SLURM_ARRAY_TASK_ID] line $line_no (slot $SLOT_IDX): information=$INFORMATION harness=$HARNESS verification=$VERIFICATION budget=$BUDGET model=$MODEL replicate=$REPLICATE"
  srun --exclusive -n1 --gres=gpu:1 --cpus-per-task="$CPUS_PER_SLOT" \
    --output="$REPO_ROOT/slurm/logs/$GRID_DIR/agent-worker-packed-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}-line${line_no}.out" \
    bash -c '
      set -euo pipefail
      GPU_UUID="$(nvidia-smi -L | sed -n "$(($1 + 1))p" | grep -oP "(?<=UUID: )[^)]+")"
      [ -n "$GPU_UUID" ] || { echo "error: could not resolve GPU UUID for slot $1 from nvidia-smi -L" >&2; exit 1; }
      echo "slot=$1 gpu=$GPU_UUID"
      shift
      # AGENT_ORACLE is inherited from the job environment.
      ORACLE_ARGS=()
      if [ "${AGENT_ORACLE:-0}" = "1" ]; then ORACLE_ARGS+=(--oracle); fi
      uv run --no-sync python -m scripts.production.run_cell \
        --grid-dir "$1" --task "$2" \
        --information "$3" --harness "$4" --verification "$5" \
        --budget "$6" --model "$7" --replicate "$8" \
        --gpu "$GPU_UUID" --base-url "$9" --no-verbose "${ORACLE_ARGS[@]}"
    ' _ "$SLOT_IDX" "$GRID_DIR" "$TASK" "$INFORMATION" "$HARNESS" "$VERIFICATION" "$BUDGET" "$MODEL" "$REPLICATE" "$BASE_URL" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"

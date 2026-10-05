#!/usr/bin/env bash
# Run stage2_cluster.py --phase all for all 6 dimensions in parallel against one shared judge
# server. Every dimension uses --keep-server; this script cancels the server after all exit.
#
# Usage: scripts/taxonomy/run_stage2.sh <stage1-dir> [<stage1-dir> ...]
# Environment:
#   STAGE2_OUTPUT_DIR           output dir (default: stage2_cluster.py's grid_label name)
#   STAGE2_CLIENT_MAX_NUM_SEQS  per-dimension request concurrency (default 4)
#   STAGE2_SERVER_MAX_NUM_SEQS  shared server batch size (default 24 = 6 x 4)
#   STAGE2_MAX_MODEL_LEN        server context length (default 256000, leaves room for merge output)
#   STAGE2_EXTRA_ARGS           extra stage2_cluster.py flags, word-split
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <stage1-dir> [<stage1-dir> ...]" >&2
  exit 1
fi

DIMENSIONS=(error_category execution_quality model_usage planning_exploration result verification_behavior)
STAGE1_DIRS=("$@")
CLIENT_MAX_NUM_SEQS="${STAGE2_CLIENT_MAX_NUM_SEQS:-4}"
SERVER_MAX_NUM_SEQS="${STAGE2_SERVER_MAX_NUM_SEQS:-24}"
MAX_MODEL_LEN="${STAGE2_MAX_MODEL_LEN:-256000}"
LOG_DIR="slurm/logs/taxonomy-judge"
mkdir -p "$LOG_DIR"

out_dir_args=()
[ -n "${STAGE2_OUTPUT_DIR:-}" ] && out_dir_args=(--output-dir "$STAGE2_OUTPUT_DIR")
read -ra extra_args <<< "${STAGE2_EXTRA_ARGS:-}"

first="${DIMENSIONS[0]}"
rest=("${DIMENSIONS[@]:1}")
first_log="$LOG_DIR/stage2-${first}.log"

# Find the new server job by diffing squeue; the submitting process's log is buffered.
before_jobs=$(squeue -u "$USER" -h -o "%i %j" | awk '$2=="taxonomy-judge-vllm"{print $1}')

echo "[run_stage2] submitting shared judge server via dimension '$first' (client max-num-seqs=$CLIENT_MAX_NUM_SEQS, server max-num-seqs=$SERVER_MAX_NUM_SEQS, max-model-len=$MAX_MODEL_LEN)"
uv run python -m scripts.taxonomy.stage2_cluster --dimension "$first" --phase all \
  --stage1-dirs "${STAGE1_DIRS[@]}" --max-num-seqs "$CLIENT_MAX_NUM_SEQS" \
  --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
  --keep-server --label "stage2-$first" "${out_dir_args[@]}" "${extra_args[@]}" \
  > "$first_log" 2>&1 &
pids=("$!")

job_id=""
for _ in $(seq 1 300); do
  now_jobs=$(squeue -u "$USER" -h -o "%i %j" | awk '$2=="taxonomy-judge-vllm"{print $1}')
  job_id=$(comm -13 <(echo "$before_jobs" | sort) <(echo "$now_jobs" | sort) | head -1)
  [ -n "$job_id" ] && break
  # the process could also have already exited (e.g. cluster phase crashed before submitting)
  kill -0 "${pids[0]}" 2>/dev/null || break
  sleep 2
done
if [ -z "$job_id" ]; then
  echo "[run_stage2] never saw a new job id for '$first' after 10 minutes (or its process exited first) -- check $first_log" >&2
  wait "${pids[0]}" || true
  exit 1
fi
echo "[run_stage2] shared job id: $job_id -- launching the remaining ${#rest[@]} dimensions"

for dim in "${rest[@]}"; do
  log="$LOG_DIR/stage2-${dim}.log"
  uv run python -m scripts.taxonomy.stage2_cluster --dimension "$dim" --phase all \
    --stage1-dirs "${STAGE1_DIRS[@]}" --max-num-seqs "$CLIENT_MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
    --attach-job-id "$job_id" --keep-server --label "stage2-$dim" "${out_dir_args[@]}" "${extra_args[@]}" \
    > "$log" 2>&1 &
  pids+=("$!")
done

echo "[run_stage2] waiting on all ${#pids[@]} dimension processes (pids: ${pids[*]})"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

echo "[run_stage2] all dimensions finished (exit status: $([ $fail -eq 0 ] && echo ok || echo 'one or more failed -- check logs')), tearing down job $job_id"
scancel "$job_id" || true
exit $fail

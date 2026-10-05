#!/usr/bin/env bash
#SBATCH --job-name=vllm-teardown
#SBATCH --time=01:00:00
# Cancel a vLLM server job once its worker array has finished. Submitted by run_grid_slurm.py.
#
# Usage: sbatch --dependency=afterany:<array-job> slurm/vllm_teardown.sbatch <server-job-id>
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: sbatch --dependency=afterany:<array-job> slurm/vllm_teardown.sbatch <server-job-id>" >&2
  exit 1
fi
SERVER_JOB="$1"
echo "Tearing down vLLM server job $SERVER_JOB"
# Not fatal if the job is already gone.
scancel "$SERVER_JOB" || echo "scancel returned nonzero (job likely already gone) -- not treating as an error"

# Backstop for vllm_server.sbatch's trap, which cannot fire on SIGKILL.
ENV_FILE="${SLURM_SUBMIT_DIR:-.}/.env"
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
  if [ -n "${CACHE_ROOT:-}" ]; then
    rm -f "$CACHE_ROOT/vllm-server-${SERVER_JOB}.addr"
  fi
fi

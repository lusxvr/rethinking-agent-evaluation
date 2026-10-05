#!/usr/bin/env bash
#SBATCH --job-name=sweep-grid
#SBATCH --time=01:00:00
# One generation of preempted-cell recovery (scripts/production/sweep_grid_slurm.py). Submitted
# by run_grid_slurm.py --preemptible and by sweep_grid_slurm.py itself.
#
# Usage: sbatch --dependency=afterany:<array-job> slurm/sweep_grid.sbatch <sweep_grid_slurm.py args...>
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?not set -- run sbatch from the repo root}"
cd "$REPO_ROOT"
# uv location; batch jobs may not source ~/.bashrc.
export PATH="$HOME/.local/bin:$PATH"

ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${CACHE_ROOT:?not set in $ENV_FILE}"

uv run --no-sync python -m scripts.production.sweep_grid_slurm "$@"

#!/usr/bin/env bash
# Build models/<model>/agent/env/.venv inside the agent container, mounted at the same path as in
# the sandbox, so the venv's python symlink resolves there. Called by orchestrate.py when missing.
# Runs agent/env/post_install.sh afterwards if the model has one.
#
# Usage: ./scripts/production/build_agent_model_env.sh <model-name>
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <model-name>" >&2
  exit 1
fi
MODEL="$1"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_ENV_DIR="$REPO_ROOT/models/$MODEL/agent/env"
[ -d "$MODEL_ENV_DIR" ] || { echo "error: $MODEL_ENV_DIR not found" >&2; exit 1; }
AGENT_SIF="$REPO_ROOT/apptainer/agent/agent.sif"
[ -f "$AGENT_SIF" ] || { echo "error: $AGENT_SIF not found -- build it first (see README.md)" >&2; exit 1; }

ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
set -a; . "$ENV_FILE"; set +a
# Same agent-side cache orchestrate.py mounts, deliberately not this host account's own
# $CACHE_ROOT/uv -- see its UV_CACHE_HOST_DIR comment.
UV_CACHE_HOST_DIR="$CACHE_ROOT/agent-uv"
mkdir -p "$UV_CACHE_HOST_DIR" "$CACHE_ROOT/apptainer"

# Replaces the 64MB session tmpfs; the persistent uv cache is bound separately.
WORKDIR="$(mktemp -d "$CACHE_ROOT/apptainer/build-env-workdir.XXXXXX")"
trap 'rm -rf "$WORKDIR"' EXIT

POST_INSTALL=""
if [ -f "$MODEL_ENV_DIR/post_install.sh" ]; then
  POST_INSTALL="bash /models/$MODEL/env/post_install.sh"
fi

# --cleanenv keeps host UV_CACHE_DIR/HF_HOME out. UV_HTTP_TIMEOUT is raised for large CUDA
# wheels on aarch64.
apptainer exec --cleanenv --contain --workdir "$WORKDIR" \
  --bind "$MODEL_ENV_DIR:/models/$MODEL/env:rw" \
  --bind "${UV_CACHE_HOST_DIR}:/uv_cache" \
  --env UV_CACHE_DIR=/uv_cache \
  --env UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-30}" \
  --pwd "/models/$MODEL/env" \
  "$AGENT_SIF" \
  sh -c "
    # The image sets UV_PROJECT_ENVIRONMENT=/app/.venv for its own agent.cli entrypoint -- override
    # it here or uv silently syncs into that (read-only, wrong-project) venv instead.
    export UV_PROJECT_ENVIRONMENT=/models/$MODEL/env/.venv
    uv sync
    $POST_INSTALL
  "

echo "Built models/$MODEL/agent/env/.venv (portable to the /models/$MODEL/env container mount path)."

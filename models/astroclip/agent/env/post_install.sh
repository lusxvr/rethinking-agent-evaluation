#!/usr/bin/env bash
# Run inside the build container by scripts/production/build_agent_model_env.sh, immediately after `uv sync`.
# --no-deps: both packages declare stale/conflicting transitive pins (torch==2.0.0, cuml-cu11,
# xformers==0.0.18) that would break this environment if resolved normally.
set -euo pipefail
uv pip install --no-deps \
  "git+https://github.com/facebookresearch/dinov2.git@2302b6bf46953431b969155307b9bed152754069" \
  "git+https://github.com/PolymathicAI/AstroCLIP.git"

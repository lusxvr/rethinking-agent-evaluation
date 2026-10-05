#!/usr/bin/env bash
# Run inside the build container by scripts/production/build_agent_model_env.sh, immediately after `uv sync`.
#
# RiNALMo's own rinalmo/config.py defaults use_flash_attn to True -- but flash-attn can't be built
# in this environment at all (no matching CUDA toolchain, see flash_attn_stub/), and the mounted
# checkpoint (models/rinalmo/download.sh's convert_checkpoint.py) was converted to the *non-flash*
# module's key naming specifically because that's the only path that can ever run here. So an
# agent that calls model_config("giga") without knowing to override this one flag hits an
# immediate key-mismatch crash that has nothing to do with the actual task -- confirmed directly:
# a real information=identity/none agent run (no access to interface.md's explicit override) hit
# exactly this and had to burn its budget working around it or fall back to a different tool
# entirely. Fixed at the source, same as DNABERT-2's Triton kernel stub (models/dnabert-2/README.md)
# -- flip the installed package's own default so a completely uninstructed load already works, at
# every information level, rather than requiring every caller to know this flag exists.
set -euo pipefail

# build_agent_model_env.sh runs this with cwd=/models/$MODEL/env -- .venv/bin/python3, not the
# bare `python3` on PATH (which is the container's system interpreter, doesn't have rinalmo).
CONFIG_PATH="$(.venv/bin/python3 -c 'import rinalmo.config, pathlib; print(pathlib.Path(rinalmo.config.__file__))')"
sed -i 's/"use_flash_attn": True,/"use_flash_attn": False,/' "$CONFIG_PATH"
grep -q '"use_flash_attn": False,' "$CONFIG_PATH" || {
  echo "error: patch didn't apply to $CONFIG_PATH -- rinalmo/config.py's format may have changed upstream" >&2
  exit 1
}
echo "Patched $CONFIG_PATH: use_flash_attn now defaults to False"

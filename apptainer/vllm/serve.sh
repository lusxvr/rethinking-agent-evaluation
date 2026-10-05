#!/usr/bin/env bash
# Serve one model tier with vLLM inside apptainer, in the foreground. Wrapped by run_vllm.sh and
# slurm/vllm_server.sbatch.
#
# Usage: serve.sh [tier]   tier is an axes.py MODEL_LEVELS key, default qwen35-35b-a3b-fp8.
set -euo pipefail

# Relative to this script, since git may be missing on compute nodes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${CACHE_ROOT:?not set in $ENV_FILE -- see .env.example}"

TIER="${1:-qwen35-35b-a3b-fp8}"

# Sets MODEL, TP_ARGS, QUANT_ARGS and MODEL_ARGS for the tier.
source "$(dirname "${BASH_SOURCE[0]}")/resolve_tier.sh"
resolve_tier "$TIER"

# Extra vllm flags, space-separated, e.g.
# VLLM_EXTRA_ARGS='--speculative-config {"method":"mtp","num_speculative_tokens":1}'.
EXTRA_ARGS=()
if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($VLLM_EXTRA_ARGS)
fi

# vLLM leaves prefix caching off for hybrid models; on here unless VLLM_ENABLE_PREFIX_CACHING=0.
PREFIX_CACHING_ARGS=()
if [ "${VLLM_ENABLE_PREFIX_CACHING:-1}" = "1" ]; then
  PREFIX_CACHING_ARGS=(--enable-prefix-caching)
fi

# Optional tuned fused-MoE kernel configs, checked by vLLM before its bundled ones.
MOE_CONFIG_BIND_ARGS=()
MOE_CONFIG_ENV_ARGS=()
if [ -n "${VLLM_TUNED_CONFIG_FOLDER:-}" ]; then
  mkdir -p "$VLLM_TUNED_CONFIG_FOLDER"
  MOE_CONFIG_BIND_ARGS=(--bind "${VLLM_TUNED_CONFIG_FOLDER}:/data-cache/moe-configs")
  MOE_CONFIG_ENV_ARGS=(--env VLLM_TUNED_CONFIG_FOLDER=/data-cache/moe-configs)
fi

# run_grid_slurm.py sets this from --max-concurrent; 1 is the manual-use default.
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
# Precedence: VLLM_GPU_MEM_UTIL, then the tier default from resolve_tier.sh, then 0.97.
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-${GPU_MEM_UTIL_DEFAULT:-0.97}}"
# vllm_server.sbatch derives the port from its job id within the profile's port range; 61000 is
# the manual fallback.
VLLM_PORT="${VLLM_PORT:-61000}"

HF_CACHE_DIR="$CACHE_ROOT/huggingface"
VLLM_CACHE_DIR="$CACHE_ROOT/vllm"
WORKDIR="$CACHE_ROOT/apptainer/vllm-workdir"
SIF_PATH="$CACHE_ROOT/apptainer/vllm-openai.sif"
mkdir -p "$HF_CACHE_DIR" "$VLLM_CACHE_DIR" "$WORKDIR" "$(dirname "$SIF_PATH")"

# Persistent OCI blob cache.
export APPTAINER_CACHEDIR="$CACHE_ROOT/apptainer/cache"
mkdir -p "$APPTAINER_CACHEDIR"

# Build scratch, removed after the pull. APPTAINER_TMPDIR_MODE (cluster profile): cache_root, or
# local for node-local $TMPDIR when builds on a network filesystem are slow.
if [ "${APPTAINER_TMPDIR_MODE:-cache_root}" = "local" ]; then
  export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/apptainer-build-${SLURM_JOB_ID:-$$}"
else
  export APPTAINER_TMPDIR="$CACHE_ROOT/apptainer/tmp"
fi
mkdir -p "$APPTAINER_TMPDIR"

# Pull the image once, then reuse it.
if [ ! -f "$SIF_PATH" ]; then
  echo "No vLLM sif found at $SIF_PATH -- pulling docker://vllm/vllm-openai:latest (several GB, may take a while)..."
  apptainer pull "$SIF_PATH" docker://vllm/vllm-openai:latest
fi
rm -rf "$APPTAINER_TMPDIR"

# Outside Slurm the GPU comes from VLLM_GPU_UUID. Under Slurm, --nvccli still needs
# NVIDIA_VISIBLE_DEVICES, so it is resolved from the GPUs nvidia-smi lists for this job.
if [ -z "${SLURM_JOB_ID:-}" ]; then
  : "${VLLM_GPU_UUID:?not set in $ENV_FILE -- see .env.example (only required outside Slurm)}"
  export NVIDIA_VISIBLE_DEVICES="$VLLM_GPU_UUID"
elif [ -n "${VLLM_GPU_UUID:-}" ]; then
  export NVIDIA_VISIBLE_DEVICES="$VLLM_GPU_UUID"
else
  NVIDIA_VISIBLE_DEVICES="$(nvidia-smi -L | grep -oP '(?<=UUID: )[^)]+' | paste -sd, -)"
  [ -n "$NVIDIA_VISIBLE_DEVICES" ] || { echo "error: could not resolve this job's GPU UUID(s) from nvidia-smi -L" >&2; exit 1; }
  export NVIDIA_VISIBLE_DEVICES
fi

echo "Serving $MODEL (tier $TIER, max-num-seqs=$MAX_NUM_SEQS, gpu-mem-util=$GPU_MEM_UTIL)"

# --cleanenv drops HF_TOKEN; forward it if set.
HF_TOKEN_ENV_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  HF_TOKEN_ENV_ARGS=(--env "HF_TOKEN=${HF_TOKEN}")
fi

# Opt-in: vLLM's startup calls the HF Hub even for cached models, which fails during an outage.
HF_OFFLINE_ENV_ARGS=()
if [ -n "${HF_HUB_OFFLINE:-}" ]; then
  HF_OFFLINE_ENV_ARGS=(--env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}")
fi

# --nvccli fails under setuid Apptainer; APPTAINER_GPU_FLAG=nv uses --nv instead.
GPU_ARGS=(--nvccli)
if [ "${APPTAINER_GPU_FLAG:-nvccli}" = "nv" ]; then
  GPU_ARGS=(--nv --writable-tmpfs)
fi

# --workdir moves the 64MB session overlay ($HOME, /tmp) to disk. --cleanenv keeps host env vars
# such as HF_HOME out. Persistent caches get explicit binds.
exec apptainer exec --cleanenv "${GPU_ARGS[@]}" --contain --workdir "$WORKDIR" \
  --bind "${HF_CACHE_DIR}:/data-cache/huggingface" \
  --bind "${VLLM_CACHE_DIR}:/data-cache/vllm" \
  --env HF_HOME=/data-cache/huggingface \
  --env VLLM_CACHE_ROOT=/data-cache/vllm \
  "${HF_TOKEN_ENV_ARGS[@]}" "${HF_OFFLINE_ENV_ARGS[@]}" \
  "${MOE_CONFIG_BIND_ARGS[@]}" "${MOE_CONFIG_ENV_ARGS[@]}" \
  "$SIF_PATH" \
  vllm serve "$MODEL" \
  --port "$VLLM_PORT" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --enable-auto-tool-choice \
  "${TP_ARGS[@]}" \
  "${QUANT_ARGS[@]}" \
  "${MODEL_ARGS[@]}" \
  "${PREFIX_CACHING_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=taxonomy-judge-vllm
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/taxonomy-judge/vllm-server-%j.out
# vLLM server for one taxonomy judge tier on a 4-GPU node (tier lookup and launch in one file).
# Submitted by judge_server.py, which adds partition, QoS, account, GPUs, CPUs and memory from the
# cluster profile.
#
# Usage: sbatch scripts/taxonomy/vllm_server.sbatch <tier> [max-model-len]
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:?not set -- run sbatch from the repo root}"
cd "$REPO_ROOT"
ENV_FILE="$REPO_ROOT/.env"
[ -f "$ENV_FILE" ] || { echo "error: $ENV_FILE not found -- run 'cp .env.example .env' from the repo root" >&2; exit 1; }
set -a; . "$ENV_FILE"; set +a
: "${CACHE_ROOT:?not set in $ENV_FILE}"
# Port range, APPTAINER_TMPDIR_MODE and APPTAINER_GPU_FLAG from the cluster profile.
CLUSTER="${CLUSTER:-example}"
CLUSTER_ENV_FILE="$REPO_ROOT/slurm/clusters/$CLUSTER.env"
[ -f "$CLUSTER_ENV_FILE" ] || { echo "error: no cluster profile at $CLUSTER_ENV_FILE (CLUSTER=$CLUSTER from .env)" >&2; exit 1; }
set -a; . "$CLUSTER_ENV_FILE"; set +a

TIER="${1:?usage: sbatch vllm_server.sbatch <tier> [max-model-len]}"
MAX_MODEL_LEN="${2:-262144}"
: "${SLURM_JOB_ID:?not set -- this script must be run under sbatch}"
ADDR_FILE="$CACHE_ROOT/taxonomy-judge-${SLURM_JOB_ID}.addr"
mkdir -p "$CACHE_ROOT"

cleanup() { rm -f "$ADDR_FILE"; }
trap cleanup EXIT TERM INT
rm -f "$ADDR_FILE"

PORT_RANGE_START="${VLLM_PORT_RANGE_START:-60001}"
PORT_RANGE_SIZE="${VLLM_PORT_RANGE_SIZE:-3000}"
export VLLM_PORT="${VLLM_PORT:-$((PORT_RANGE_START + SLURM_JOB_ID % PORT_RANGE_SIZE))}"

# --- tier lookup: MODEL, TP_ARGS, DP_ARGS, PARSER_ARGS, GPU_MEM_UTIL ---
# The installed vLLM 0.28.0 image registers every architecture and parser used below.
GPU_MEM_UTIL="0.97"
case "$TIER" in
  qwen35-122b-a10b-fp8)
    # Same model as resolve_tier.sh's tier; two TP=2 replicas use the whole node.
    MODEL="Qwen/Qwen3.5-122B-A10B-FP8"
    TP_ARGS=(--tensor-parallel-size 2)
    DP_ARGS=(--data-parallel-size 2)
    PARSER_ARGS=(--tool-call-parser qwen3_coder --reasoning-parser qwen3)
    ;;
  deepseek-v4-flash-0731-fp8)
    # 155.4GiB fp8 at TP=4. MLA needs --kv-cache-dtype fp8_ds_mla and --block-size 256. 0.90 memory
    # utilization, since long prefills OOM at 0.97 (the sparse indexer's scratch grows with length).
    # --no-enable-flashinfer-autotune follows the vLLM Hopper recipes.
    MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
    TP_ARGS=(--tensor-parallel-size 4)
    DP_ARGS=()
    PARSER_ARGS=(--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4
                 --kv-cache-dtype fp8_ds_mla --block-size 256 --no-enable-flashinfer-autotune)
    GPU_MEM_UTIL="0.90"
    ;;
  mimo-v2.5-fp8)
    # Broken in vLLM 0.28.0: TP=4 crashes in mimo_v2.py::_shard_fp8_qkv_proj (fused fp8 QKV with
    # kv_heads_per_rank > 1; related: vllm-project/vllm#42803). Use mimo-v2-flash-fp8 instead.
    MODEL="XiaomiMiMo/MiMo-V2.5"
    TP_ARGS=(--tensor-parallel-size 4)
    DP_ARGS=()
    PARSER_ARGS=(--tool-call-parser mimo --reasoning-parser mimo)
    ;;
  mimo-v2-flash-fp8)
    # Fallback for mimo-v2.5-fp8. Context 262,144, which truncates the longest traces (~280K tokens).
    MODEL="XiaomiMiMo/MiMo-V2-Flash"
    TP_ARGS=(--tensor-parallel-size 4)
    DP_ARGS=()
    PARSER_ARGS=(--tool-call-parser mimo --reasoning-parser mimo)
    ;;
  *)
    echo "error: unknown tier '$TIER' -- see this file's case statement for valid tiers" >&2
    exit 1
    ;;
esac

# --- environment/cache setup, then the vLLM launch itself ---
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-10}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-$GPU_MEM_UTIL}"  # tier's own default above, VLLM_GPU_MEM_UTIL overrides

HF_CACHE_DIR="$CACHE_ROOT/huggingface"
VLLM_CACHE_DIR="$CACHE_ROOT/vllm"
WORKDIR="$CACHE_ROOT/apptainer/taxonomy-judge-workdir"
SIF_PATH="$CACHE_ROOT/apptainer/vllm-openai.sif"
mkdir -p "$HF_CACHE_DIR" "$VLLM_CACHE_DIR" "$WORKDIR" "$(dirname "$SIF_PATH")"

export APPTAINER_CACHEDIR="$CACHE_ROOT/apptainer/cache"
mkdir -p "$APPTAINER_CACHEDIR"

# Build scratch as in apptainer/vllm/serve.sh; the image is shared with the agent grid.
if [ "${APPTAINER_TMPDIR_MODE:-cache_root}" = "local" ]; then
  export APPTAINER_TMPDIR="${TMPDIR:-/tmp}/apptainer-build-${SLURM_JOB_ID}"
else
  export APPTAINER_TMPDIR="$CACHE_ROOT/apptainer/tmp"
fi
mkdir -p "$APPTAINER_TMPDIR"

if [ ! -f "$SIF_PATH" ]; then
  echo "No vLLM sif found at $SIF_PATH -- pulling docker://vllm/vllm-openai:latest (several GB, may take a while)..."
  apptainer pull "$SIF_PATH" docker://vllm/vllm-openai:latest
fi
rm -rf "$APPTAINER_TMPDIR"

# Same GPU-visibility resolution as apptainer/vllm/serve.sh (NVIDIA_VISIBLE_DEVICES needs setting
# explicitly even under Slurm's --gres scoping). Always under sbatch here, so no non-Slurm branch.
if [ -n "${VLLM_GPU_UUID:-}" ]; then
  export NVIDIA_VISIBLE_DEVICES="$VLLM_GPU_UUID"
else
  NVIDIA_VISIBLE_DEVICES="$(nvidia-smi -L | grep -oP '(?<=UUID: )[^)]+' | paste -sd, -)"
  [ -n "$NVIDIA_VISIBLE_DEVICES" ] || { echo "error: could not resolve this job's GPU UUID(s) from nvidia-smi -L" >&2; exit 1; }
  export NVIDIA_VISIBLE_DEVICES
fi

echo "Serving $MODEL (tier $TIER, tp=${TP_ARGS[1]:-1}, dp=${DP_ARGS[1]:-1}, max-num-seqs=$MAX_NUM_SEQS/replica, max-model-len=$MAX_MODEL_LEN, gpu-mem-util=$GPU_MEM_UTIL)"

HF_TOKEN_ENV_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  HF_TOKEN_ENV_ARGS=(--env "HF_TOKEN=${HF_TOKEN}")
fi

# As in apptainer/vllm/serve.sh: --nv for setuid Apptainer, else --nvccli.
GPU_ARGS=(--nvccli)
if [ "${APPTAINER_GPU_FLAG:-nvccli}" = "nv" ]; then
  GPU_ARGS=(--nv --writable-tmpfs)
fi

# Opt-in offline mode (export HF_HUB_OFFLINE=1) once the judge weights are cached.
HF_OFFLINE_ENV_ARGS=()
if [ -n "${HF_HUB_OFFLINE:-}" ]; then
  HF_OFFLINE_ENV_ARGS=(--env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}" --env "TRANSFORMERS_OFFLINE=${HF_HUB_OFFLINE}")
fi

apptainer exec --cleanenv "${GPU_ARGS[@]}" --contain --workdir "$WORKDIR" \
  --bind "${HF_CACHE_DIR}:/data-cache/huggingface" \
  --bind "${VLLM_CACHE_DIR}:/data-cache/vllm" \
  --env HF_HOME=/data-cache/huggingface \
  --env VLLM_CACHE_ROOT=/data-cache/vllm \
  "${HF_OFFLINE_ENV_ARGS[@]}" \
  "${HF_TOKEN_ENV_ARGS[@]}" \
  "$SIF_PATH" \
  vllm serve "$MODEL" \
  --port "$VLLM_PORT" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --enable-prefix-caching \
  "${TP_ARGS[@]}" \
  "${DP_ARGS[@]}" \
  "${PARSER_ARGS[@]}" &
SERVER_PID=$!

(
  # 3h default covers a cold weight pull (up to ~293GiB, MiMo tier); override via VLLM_READY_TIMEOUT_S.
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
  echo "error: vLLM did not become ready within ${attempts}x10s -- check slurm/logs/taxonomy-judge/vllm-server-${SLURM_JOB_ID}.out" >&2
) &

wait "$SERVER_PID"

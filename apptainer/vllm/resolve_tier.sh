# Tier lookup for serve.sh. Source this file, then call resolve_tier "$TIER". Sets MODEL, TP_ARGS,
# QUANT_ARGS, MODEL_ARGS (family-specific parsers) and GPU_MEM_UTIL_DEFAULT. axes.py's
# MODEL_GPU_COUNT must match the tensor-parallel sizes.
resolve_tier() {
  local tier="$1"
  # Per-tier default; VLLM_GPU_MEM_UTIL overrides it.
  GPU_MEM_UTIL_DEFAULT="0.97"
  case "$tier" in
    qwen35-397b-a17b-fp8)
      # 406GB weights: 8 GPUs (TP must divide its 32 attention heads).
      MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
      TP_ARGS=(--tensor-parallel-size 8)
      QUANT_ARGS=()
      MODEL_ARGS=(--max-model-len 262144 --tool-call-parser qwen3_coder --reasoning-parser qwen3)
      ;;
    qwen35-122b-a10b-fp8)
      # 125GB weights at TP=2.
      MODEL="Qwen/Qwen3.5-122B-A10B-FP8"
      TP_ARGS=(--tensor-parallel-size 2)
      QUANT_ARGS=()
      MODEL_ARGS=(--max-model-len 262144 --tool-call-parser qwen3_coder --reasoning-parser qwen3)
      ;;
    qwen35-35b-a3b-fp8)
      # Default model level.
      MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
      TP_ARGS=()
      QUANT_ARGS=()
      MODEL_ARGS=(--max-model-len 262144 --tool-call-parser qwen3_coder --reasoning-parser qwen3)
      ;;
    step37-198b-a11b-fp8)
      # 213GB weights at TP=4 with expert parallelism (the 1280-wide expert FFNs do not shard across
      # fp8 128-blocks). step3p5 parsers and --trust-remote-code; thinking via reasoning_effort.
      # 0.85 memory utilization, since 0.97 OOMs during CUDA-graph warmup.
      GPU_MEM_UTIL_DEFAULT="0.85"
      MODEL="stepfun-ai/Step-3.7-Flash-FP8"
      TP_ARGS=(--tensor-parallel-size 4)
      QUANT_ARGS=()
      MODEL_ARGS=(--max-model-len 262144 --tool-call-parser step3p5 --reasoning-parser step3p5
                  --enable-expert-parallel --disable-cascade-attn --trust-remote-code)
      ;;
    *)
      echo "error: unknown tier '$tier' -- must be a vLLM tier of axes.py's MODEL_LEVELS" >&2
      exit 1
      ;;
  esac
}

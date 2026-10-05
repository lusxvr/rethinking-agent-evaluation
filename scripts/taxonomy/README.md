# Outcome taxonomy

An LLM judge codes every trajectory, and the findings are clustered into a small category system
per dimension (extract-then-cluster, Chirkova et al., arXiv:2506.09147). The judge server runs as a
Slurm job on one 4-GPU node, configured by the cluster profile (`slurm/clusters/`).

Pipeline: `stage1_extract.py` (Stage 1) -> `stage2_cluster.py` (Stage 2) ->
`scripts/analysis/analysis_taxonomy.py` (join with axis levels and `score.json`; `--plots` runs
`plot_categories.py`).

## Judge model

Fixed judge: DeepSeek-V4-Flash-0731 (`deepseek-v4-flash-0731-fp8`). It is not one of the grid's
backbones, matched the other candidates' speed, and is the only candidate whose context
(1,048,576 tokens) exceeds the longest trace (~280K tokens). `--tier` overrides it; the 262K tiers
also need `--max-model-len 262144`.

| Model | Tier id | Params (total / active) | Weights | GPUs | Context |
|---|---|---|---|---|---|
| Qwen3.5-122B-A10B | `qwen35-122b-a10b-fp8` | 122B / 10B | ~125GB FP8 | 2 (x2 DP) | 262,144 |
| DeepSeek-V4-Flash-0731 | `deepseek-v4-flash-0731-fp8` | 304B / ~13B | 155.4GiB FP8 | 4 | 1,048,576 |
| MiMo-V2.5 (does not load) | `mimo-v2.5-fp8` | 310B / 15B | 293.4GiB FP8 | 4 | 1,048,576 |
| MiMo-V2-Flash | `mimo-v2-flash-fp8` | 309B / 15B | 313.1GiB FP8 | 4 | 262,144 |

MiMo-V2.5 crashes in vLLM 0.28.0's fused fp8 QKV sharding at TP=4 (see `vllm_server.sbatch`).

## Stage 1

Six free-text dimensions per trajectory (`result`, `error_category`, `model_usage`,
`verification_behavior`, `planning_exploration`, `execution_quality`), each given the same context
(`build_common_context`). Findings use generic role names instead of specific model or library
names; with specific names, Stage 2 clustered by task (93-100% single-grid clusters).
`--domain-specific` disables this. Sampling follows each model's card; DeepSeek also gets
`presence_penalty=0.3` against runaway generations.

Throughput: the three working judges took 112-117s for 60 calls on one node, about 3.8h per
dimension for the full corpus. Run one job per dimension in parallel.

## Stage 2

`--phase all` runs `cluster` (embeddings and HDBSCAN, all grids pooled, no LLM), `label` (one LLM
call per cluster), `merge` (one LLM call assigns every label to at most `MAX_CATEGORIES`
categories) and `valence` (scores each category from -1 to +1). `run_stage2.sh` runs all six
dimensions against one shared judge server.

## Files

- `vllm_server.sbatch`: judge server job (tier lookup and launch).
- `judge_server.py`: submit, wait for and inspect the judge server.
- `stage1_extract.py`: Stage 1; output in `stage1_output/<grid>/<dimension>.jsonl`.
- `stage2_cluster.py`: Stage 2; output in `stage2_output/<grid_label>/` (clusters, labels, final
  `categories.json` and `assignments.jsonl`).
- `run_stage2.sh`: all Stage 2 dimensions on one server.

## Manual server

```bash
sbatch scripts/taxonomy/vllm_server.sbatch deepseek-v4-flash-0731-fp8
tail -f slurm/logs/taxonomy-judge/vllm-server-<jobid>.out
cat $CACHE_ROOT/taxonomy-judge-<jobid>.addr
```

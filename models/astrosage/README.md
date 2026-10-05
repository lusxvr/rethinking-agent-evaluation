# AstroSage-8B

`AstroMLab/AstroSage-8B`, a Llama-3.1-8B model further trained on astronomy arXiv papers
(2007-2024) and synthetic QA, for astronomy question answering. The domain-relevant model for
`mmlu-astronomy`, where it is worse than the backbone (gap_negative). Llama 3.1 Community License.

- Model card: https://huggingface.co/AstroMLab/AstroSage-8B
- Paper (AstroMLab 3): https://arxiv.org/abs/2411.09012
- Base: Meta-Llama-3.1-8B, merged with Meta-Llama-3.1-8B-Instruct (75/25, per the model card)

## Contents

- `agent/`: what the agent sees, mounted read-only at `/models/astrosage`: `weights/` (safetensors,
  ~15GB, gitignored), `env/` (uv project; `.venv` built by `orchestrate.py`) and `README.md`.
- `dev_env/`: host environment for `verify.py` and `solutions/mmlu-astronomy/dev/` (see
  `models/astroclip/README.md`, "Two separate environments").
- `verify.py`: loads the model and answers sample questions.
- `download.sh`: fetches the weights into `agent/weights/` (needs `.env`).

```bash
./download.sh
cd dev_env && uv sync && uv run --project . python ../verify.py
```

## Reproducing the mmlu-astronomy anchors

The task uses MMLU astronomy (`cais/mmlu`, 152 test questions), since AstroSage's own Astrobench
MCQ keys are not public. Prompting is 0-shot generation with letter parsing, as the agent would use
it; `run_reference.py` adds AstroSage's recommended prefix (`mcq.ASTROSAGE_PREFIX`). All commands
run from `dev_env/`; step 1 is also `../../solutions/mmlu-astronomy/download.sh`.

```bash
uv run --project . python ../../../solutions/mmlu-astronomy/dev/prepare_data.py    # questions, solution.csv
uv run --project . python ../../../solutions/mmlu-astronomy/dev/run_reference.py   # REFERENCE
uv run --project . python ../../../solutions/mmlu-astronomy/dev/run_backbone.py    # BACKBONE (needs vLLM)
uv run --project . python ../../../solutions/mmlu-astronomy/dev/run_base_llama.py  # base Llama (gated, HF_TOKEN)
uv run --project . python ../../../solutions/mmlu-astronomy/dev/eval.py \
  ../../../solutions/mmlu-astronomy/dev/results/astrosage-8b.json \
  ../../../solutions/mmlu-astronomy/dev/results/backbone.json
```

Results: AstroSage-8B 102/152 (67.1%, 95% CI 59.3-74.1%), backbone 147/152 (96.7%, 92.5-98.6%),
McNemar exact p < 0.001. The right agent behavior is therefore to answer with the backbone and
decline AstroSage.

## Key findings

- AstroSage-8B does not beat its base model either (`base-llama*.json`).
- Greedy decoding on vLLM is not exactly reproducible (continuous batching, MoE routing).
- `astroclip` and the other models are mounted as distractors.

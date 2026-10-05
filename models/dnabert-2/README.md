# DNABERT-2

`zhihan1996/DNABERT-2-117M`, a BERT-style encoder for DNA, pretrained with masked language
modeling on multi-species genomes with BPE tokenization. The specialist for `promoter-prediction`
(GUE `prom_core_tata`); a distractor elsewhere. Apache 2.0.

- Model card: https://huggingface.co/zhihan1996/DNABERT-2-117M
- Paper: https://arxiv.org/abs/2306.15006
- Code: https://github.com/MAGICS-LAB/DNABERT_2

## Contents

- `agent/`: what the agent sees, mounted read-only at `/models/dnabert-2`.
  - `weights/`: checkpoint and its custom modeling code (`trust_remote_code=True`), gitignored.
  - `env/`: uv project (PyPI only); `.venv` built by `orchestrate.py` when missing.
  - `README.md`: the agent's documentation. Loading code and the fine-tuning recipe are left to
    the `interface` and `protocol` information levels.
- `dev_env/`: host environment for `verify.py` and `solutions/promoter-prediction/dev/` (see
  `models/astroclip/README.md`, "Two separate environments").
- `verify.py`: loads the checkpoint and runs a forward and backward pass.
- `download.sh`: fetches a pinned revision into `agent/weights/` and stubs out the broken Triton
  kernel (below). Needs `.env`.

```bash
./download.sh
../../scripts/production/build_agent_model_env.sh dnabert-2   # manual rebuild of agent/env
cd dev_env && uv sync && cd .. && uv run --project dev_env python verify.py
```

## Triton kernel stub

`bert_layers.py` uses a vendored Triton flash-attention kernel whenever `triton` is importable
(it is, via torch), and that kernel crashes on CPU and GPU. Its import is wrapped in
`try/except ImportError` with a plain PyTorch fallback, so `download.sh` replaces
`flash_attn_triton.py` with an empty file. Deleting the file does not work: transformers copies
every imported file before running the module.

## Reproducing the promoter-prediction anchors

All commands run from `dev_env/`; step 1 is also `../../solutions/promoter-prediction/download.sh`.

```bash
uv run --project . python ../../../solutions/promoter-prediction/dev/prepare_data.py           # task data, solution.csv
uv run --project . python ../../../solutions/promoter-prediction/dev/run_reference.py --seed 0 # REFERENCE (~15s on H100)
uv run --project . python ../../../solutions/promoter-prediction/dev/run_backbone.py \
  --base-url http://localhost:<port>/v1 [--thinking]                                          # BACKBONE
```

## Key findings

- `transformers>=5` breaks the checkpoint's custom `__init__` (ALiBi tensor during meta-device
  init); both environments pin `transformers<5`.
- The custom `SequenceClassifierOutput` always sets `hidden_states`, so `Trainer` predictions
  arrive as `(logits, hidden_states)`.
- GUE `prom_core_tata` is on Hugging Face (`leannmlindsey/GUE`); no Google Drive download needed.
- 3 epochs at batch size 32 instead of 10 at 8 cost about 2% MCC and run 12x faster (13.6s vs.
  171.7s on H100), which fits the 5-minute budget.
- `agent/env` includes `datasets`, which the `protocol` recipe needs; installing it at runtime
  overflows the 64MB sandbox overlay.

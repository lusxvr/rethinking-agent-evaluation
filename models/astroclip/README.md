# AstroCLIP

`polymathic-ai/astroclip` (370M), a non-generative model aligning a DINOv2 galaxy image encoder and
a SpecFormer spectrum encoder in one embedding space, trained on DESI imaging and spectra. The
specialist for `redshift-estimation`; a distractor elsewhere. Weights and code: MIT.

- Model card: https://huggingface.co/polymathic-ai/astroclip
- Paper: https://arxiv.org/abs/2310.03024
- Code: https://github.com/PolymathicAI/AstroCLIP

## Contents

- `agent/`: what the agent sees, mounted read-only at `/models/astroclip`.
  - `weights/astroclip.ckpt`: Lightning checkpoint (~1.6GB, gitignored; `download.sh`).
  - `env/`: uv project for the agent. `post_install.sh` adds the git-only `dinov2` and `astroclip`
    packages with `--no-deps`. The `.venv` is built by `orchestrate.py` when missing.
  - `README.md`: the agent's documentation. It omits the k-NN recipe and the `ToRGB` preprocessing
    on purpose (see Key findings).
- `dev_env/`: host environment for `verify.py` and `solutions/redshift-estimation/dev/`.
- `verify.py`: checks that the checkpoint loads and both encoders give finite embeddings.
- `download.sh`: fetches the checkpoint into `agent/weights/` (needs `.env`).

## Two separate environments

`agent/env` and `dev_env` have the same dependencies but are never interchangeable. This applies
to every model.

- `agent/env/.venv` is built inside the agent container, mounted at `/models/<model>/env`, by
  `scripts/production/build_agent_model_env.sh <model>`. A venv links to the Python that built it,
  and the host's Python does not exist in the container. Never run `uv sync` on `agent/env` from
  the host.
- `dev_env` is an ordinary host uv project.

```bash
../../scripts/production/build_agent_model_env.sh astroclip   # manual rebuild of agent/env

cd dev_env && uv sync && uv pip install --no-deps \
  "git+https://github.com/facebookresearch/dinov2.git@2302b6bf46953431b969155307b9bed152754069" \
  "git+https://github.com/PolymathicAI/AstroCLIP.git"
cd .. && uv run --project dev_env python verify.py
```

## Reproducing the redshift-estimation anchors

`../../solutions/redshift-estimation/download.sh` runs steps 1 and 2. All commands run from
`dev_env/`.

1. Reference catalog and a default query set (~40 min, embeds 126,907 images, ~43GB download;
   set `HF_TOKEN` and `CUDA_VISIBLE_DEVICES`):
   `uv run --project . python ../../../solutions/redshift-estimation/dev/prepare_data.py`
2. The fixed 20-galaxy query set used in the paper:
   `uv run --project . python ../../../solutions/redshift-estimation/dev/build_final_query_set.py`
3. REFERENCE (expected R² 0.7534). Cap BLAS threads on many-core hosts, or it segfaults:
   ```bash
   export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
   uv run --project . python ../../../solutions/redshift-estimation/dev/run_reference.py
   ```
4. BACKBONE (needs a vLLM server; expected R² about -3 to -4.4 without thinking):
   `uv run --project . python ../../../solutions/redshift-estimation/dev/run_backbone.py [--thinking --concurrency 4]`.
   Thinking mode needs a server with `--max-model-len 65536 --max-num-seqs 4`.

## Key findings

- The embedding dimension is 1024, not the 512 the paper implies. Embeddings are not
  unit-normalized (norms ~9 for images, ~53 for spectra), so standardize them. Inference needs no
  `cuml`, and current torch versions work despite the upstream `torch==2.0.0` pin.
- The zero-shot recipe exists only in the evaluation code
  (`downstream_tasks/property_estimation/property_utils/models.py`, `zero_shot()`):
  `KNeighborsRegressor(n_neighbors=64, weights="distance")` on `StandardScaler`-standardized
  embeddings.
- `EiffL/AstroCLIP` images are raw nanomaggie flux. Without the `ToRGB` arcsinh stretch
  (`astroclip.data.datamodule.AstroClipCollator`), embeddings degenerate (R² 0.09-0.2). This is
  undocumented upstream and deliberately not mentioned to the agent.
- Random n=20 query sets vary widely (R² std 0.2-0.3; the stable value at n=5,000-10,000 is about
  0.52). The 20 galaxies were chosen from 2,000 draws for R² near 0.75. The comparison stays fair
  because agent and reference run on the same fixed set, and a correct agent reproduces the
  reference's deterministic k-NN exactly.

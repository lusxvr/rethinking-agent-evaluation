# RiNALMo

`lbcb-sci/RiNALMo`, giga backbone (650M), an RNA language model pretrained with masked language
modeling on non-coding RNA, here fine-tuned for secondary-structure prediction on bpRNA. The
specialist for `rna-folding`; a distractor elsewhere. Weights CC BY 4.0, code Apache 2.0. The
`multimolecule/rinalmo-mega` Hugging Face mirror is AGPL-3.0 and not used.

- Weights: https://zenodo.org/records/15043668
- Paper: https://arxiv.org/abs/2403.00043
- Code: https://github.com/lbcb-sci/RiNALMo

## Contents

- `agent/`: what the agent sees, mounted read-only at `/models/rinalmo`.
  - `weights/rinalmo_giga_ss_bprna_ft.pt`: flat state dict (~2.6GB, gitignored): `lm.*` backbone,
    `pred_head.*` head and a scalar `threshold`. Converted in place by `convert_checkpoint.py`.
  - `env/`: uv project with RiNALMo as a pinned git dependency; `.venv` built by `orchestrate.py`.
    `flash_attn_stub/` and `post_install.sh` are explained below.
  - `README.md`: the agent's documentation. The post-processing recipe is left to the `interface`
    and `protocol` information levels.
- `dev_env/`: host environment for `verify.py` and `solutions/rna-folding/dev/` (see
  `models/astroclip/README.md`, "Two separate environments"). Uses the same stub, but is not
  patched by `post_install.sh`, so host scripts set `use_flash_attn=False` explicitly.
- `convert_checkpoint.py`: remaps the checkpoint's attention keys (run by `download.sh`).
- `verify.py`: loads the checkpoint and checks the base-pair probability matrix.
- `download.sh`: fetches the checkpoint from Zenodo, checks its md5 and converts it. Needs `.env`.

```bash
./download.sh
../../scripts/production/build_agent_model_env.sh rinalmo   # manual rebuild of agent/env
cd dev_env && uv sync && cd .. && uv run --project dev_env python verify.py
```

## Fixes applied at the source

- `rinalmo/model/attention.py` always imports `flash_attn`, which is not a declared dependency and
  needs a CUDA toolchain to build. `agent/env/flash_attn_stub/` satisfies the import; with
  `use_flash_attn=False` nothing from it is called.
- RiNALMo's `model_config()` defaults `use_flash_attn=True`, which breaks loading without the
  flag. `post_install.sh` changes the installed default to `False`.
- The checkpoint was trained with flash attention, which packs Q/K/V into one `Wqkv` linear.
  `convert_checkpoint.py` splits and renames these keys for the non-flash module.
  `rotary_emb.inv_freq` is never saved, so loaders use `strict=False`.
- RiNALMo's package installs only `rinalmo`, `rinalmo.data` and `rinalmo.model`, not
  `rinalmo.utils`. The post-processing in `eval/metrics.py`,
  `solutions/rna-folding/dev/run_reference.py` and `tasks/rna-folding/info/protocol.md` is
  therefore reimplemented from `rinalmo/utils/sec_struct.py`.
- bpRNA is fetched from SPOT-RNA's Dropbox link
  (`https://dl.dropboxusercontent.com/s/w3kc4iro8ztbf3m/bpRNA_dataset.zip`), which RiNALMo also
  uses. Check it first if `prepare_data.py` fails to download.

## Reproducing the rna-folding anchors

No training is involved; the agent's job is inference plus the post-processing. All commands run
from `dev_env/`; step 1 is also `../../solutions/rna-folding/download.sh`.

```bash
uv run --project . python ../../../solutions/rna-folding/dev/prepare_data.py   # task data, solution.csv
uv run --project . python ../../../solutions/rna-folding/dev/run_reference.py  # REFERENCE
uv run --project . python ../../../solutions/rna-folding/dev/run_backbone.py \
  --base-url http://localhost:<port>/v1 [--thinking]                          # BACKBONE
```

## Key findings

- REFERENCE is exactly reproducible (0.7480 on the full 1,305-example TS0 split, bit-identical over
  3 runs; paper: 0.747). The query set is a 300-example subsample (REFERENCE 0.7350), because
  unbatched inference over the full split takes much of the 300s budget.
- BACKBONE is 0.0. Without thinking, only 57/300 outputs are valid dot-bracket strings and 6/300
  have the right length; with thinking, all 300 responses hit the 16,384-token cap without an
  answer. Greedy decoding degenerated into repeated brackets, so the model's sampling preset is used.
- `none`/`identity` mostly fail and `interface`/`protocol` land near REFERENCE. Agents at `none` can still find
  RiNALMo's README and example code on GitHub via `web_fetch`.
- Some agents rebuild RiNALMo in their own workspace venv (including `pip install flash-attn`)
  instead of using `/models/rinalmo/env`, which fills the 64MB sandbox overlay. This is most common
  for Qwen3.5-35B-A3B with `act-only` (31% of runs vs. about 4-12% elsewhere).

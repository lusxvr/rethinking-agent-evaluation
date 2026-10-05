# Agents Are Systems, Not Models: Rethinking Agentic Evaluation

Code for the paper ([arXiv:2610.01618](https://arxiv.org/abs/2610.01618)). A benchmark of how
system design affects an autonomous research agent that must find, operate and evaluate domain
specialist models. An LLM backbone drives a sandboxed agent that reads a task, researches what it
needs, writes and runs code, and submits a result. The result is scored against two measured
anchors: the specialist model operated correctly (`REFERENCE`) and the backbone alone (`BACKBONE`).

The agent trajectories from the paper's runs are on Hugging Face:
[lusxvr/agentic-science-trajectories](https://huggingface.co/datasets/lusxvr/agentic-science-trajectories).

Five axes are varied per run: how much of the solution the agent is told (**information**), how
its reasoning is kept (**harness**), how much self-verification is asked for (**verification**),
its wall-clock **budget**, and the backbone **model**.

| Task | Specialist model | Metric | Regime |
|---|---|---|---|
| `redshift-estimation` | AstroCLIP | R² | gap_positive |
| `promoter-prediction` | DNABERT-2 | MCC | gap_positive |
| `rna-folding` | RiNALMo | relaxed F1 | gap_positive |
| `mmlu-astronomy` | AstroSage-8B | accuracy | gap_negative (the specialist is worse) |

Every task mounts all four models, so three are distractors.

## Setup

1. Install `uv` and Apptainer.
2. `cp .env.example .env` and set `CACHE_ROOT`, the GPU UUIDs (single node) or `CLUSTER` (Slurm),
   `HF_TOKEN`, and `ANTHROPIC_API_KEY` if you use the Claude tier.
3. Download each model's weights and check them (details in each `models/<name>/README.md`):
   ```bash
   for m in astroclip astrosage dnabert-2 rinalmo; do ./models/$m/download.sh; done
   ```
4. Build each task's data and answer key:
   ```bash
   ./solutions/mmlu-astronomy/download.sh        # seconds
   ./solutions/promoter-prediction/download.sh   # seconds
   ./solutions/rna-folding/download.sh           # minutes
   ./solutions/redshift-estimation/download.sh   # ~40 min on a GPU, ~43GB download
   uv run python -m scripts.production.check_task <task>
   ```
5. Build the agent sandbox image:
   ```bash
   apptainer build --fakeroot --force apptainer/agent/agent.sif apptainer/agent/agent.def
   ```
6. `uv run pytest` runs the framework tests (no GPU, container or backbone needed).

## Running

**Backbone server** (vLLM in Apptainer; the image is pulled to `$CACHE_ROOT` on first use):

```bash
./apptainer/vllm/run_vllm.sh [tier]    # tier: a vLLM tier of axes.py's MODEL_LEVELS
curl -s localhost:61000/v1/models
./apptainer/vllm/stop_vllm.sh
```

`apptainer/vllm/resolve_tier.sh` maps each tier to its weights and serving flags. Every run checks
that the server serves its model level before starting. `scripts/production/test_tool_call.py`
checks tool calling against a live server; `test_claude_call.py` does the same for the Anthropic
API (one billed request).

**One run:**

```bash
uv run python orchestrate.py --task redshift-estimation --base-url http://localhost:61000/v1 \
  [--information ... --harness ... --verification ... --budget ... --model ... --oracle]
```

This creates `runs/<timestamp>/`, starts a GPU monitor, launches the agent container, then writes
`score.json`, `trace_report.md` and `analysis_generated.json`. Model agent environments are built
on first use.

**A grid** is a YAML spec with an explicit value list per axis:

```yaml
task: redshift-estimation
axes:
  information: [none, identity, interface, protocol]
  harness: [act-only, think-act, react]
  verification: [none, asked, reported, binding]
  budget: [short, medium, long]
  model: [qwen35-35b-a3b-fp8]
replicates: 5
oracle: false   # optional: grant oracle_check to every cell
```

```bash
uv run python -m scripts.production.run_grid redshift-estimation/smoke-small-fp8 --gpus <uuid>,<uuid>
uv run python -m scripts.production.run_grid_slurm redshift-estimation/full-small-fp8   # Slurm
```

`run_grid` runs cells locally, one per GPU in `--gpus` (never the backbone's GPU). Each call
creates a new `runs/<spec>-<timestamp>/` with one directory per cell,
`<task>__<information>__<harness>__<verification>__<budget>__<model>__r<n>/`.
`run_grid_slurm` submits per tier a server job, a throttled worker array and a teardown job, all
chained through Slurm dependencies; `continue_grid_slurm` resumes a grid whose server stopped.
Cluster settings (partitions, QoS, GPU request syntax, whole-node allocation, internet flags)
live in a profile: copy `slurm/clusters/example.env` to `slurm/clusters/<name>.env`, fill it in and
set `CLUSTER=<name>` in `.env`. Each tier's server runs on one node, so qwen35-397b-a17b-fp8 needs
a node with 8 GPUs of 80GB or more.

The paper's grids are `specs/<task>/full-*.yaml` (main experiments and model-family ablations) and
`specs/<task>/oracle-full-*.yaml` (oracle ablation); `smoke-small-fp8.yaml` is a small test grid.

**Re-scoring** a submission by hand:

```bash
uv run python -m eval.evaluate --task redshift-estimation --submission <file.csv> \
  [--model-used astroclip --expected-score 0.7]
```

## Sandbox

The agent runs only inside `agent.sif`, with `--contain` and `--cleanenv`: `/task` (data and the
composed description) and `/models` are read-only, `/agent_run/workspace` is the only writable run
path, and only explicitly passed variables reach the agent. The trace is written host-side, so the
agent cannot read its own condition labels. Tools: `read_file`, `write_file`, `run_bash`,
`web_fetch` (allow/deny lists; loopback always blocked), `finish`, and `oracle_check` when granted.

Accepted risks: the container runs as your own uid, and the network is not isolated (`run_bash`
could reach the internet directly).

## The experiment axes

`axes.py` defines all levels; `RunSpec` validates them.

| Axis | Levels | What it varies |
|---|---|---|
| `information` | `none`, `identity`, `interface`, `protocol` | how much of the task's solution the agent is given |
| `harness` | `act-only`, `think-act`, `react` | whether reasoning is generated and whether it persists |
| `verification` | `none`, `asked`, `reported`, `binding` | how much self-verification is asked for |
| `budget` | `short` (300s), `medium` (600s), `long` (1200s) | wall-clock time |
| `model` | `qwen35-35b-a3b-fp8`, `qwen35-122b-a10b-fp8`, `qwen35-397b-a17b-fp8`, `step37-198b-a11b-fp8`, `claude-sonnet-5` | backbone |

- **Information** is cumulative: each level appends one `tasks/<task>/info/` snippet to
  `description.md`. Only the composed description is mounted.
- **Harness**: `act-only` generates no reasoning, `think-act` generates and discards it each turn,
  `react` keeps it in context as `Thought N:` (Yao et al., 2022). The system prompt is identical
  across levels. `think-act` -> `react` isolates persistence; `act-only` -> `think-act` also
  switches to the vendor's non-thinking sampling preset.
- **Verification** changes only the prompt wording and which `finish()` parameters exist
  (`reported` and `binding` require `expected_score` and `verification_evidence`).
- **Budget** is wall-clock only, checked before each LLM call and after each tool call. Each level
  also has a hidden iteration cap as a runaway guard.
- **Model** bundles weights with their vendor's sampling preset per thinking mode, since presets
  differ within a family. Step-3.7-Flash cannot disable thinking, so its specs omit `act-only`.
  `claude-sonnet-5` is a hosted API with a fixed effort instead of sampling parameters; see
  [docs/anthropic-integration.md](docs/anthropic-integration.md).
- **Oracle** (`--oracle`, spec key `oracle`) is a separate ablation: a tool that scores a
  submission against the reference, independent of the verification level.

`trace_render.py` checks that each manipulation reached the agent (e.g. that `react` actually
persisted reasoning) and records mismatches as `check_violations` in `analysis_generated.json`.

## Scoring

Grading rules live in `solutions/<task>/eval_config.py`: columns, `METRIC`, direction and four
anchors in metric units. `REFERENCE` is the specialist operated by us
(`solutions/<task>/dev/run_reference.py`), `BACKBONE` the backbone alone (`run_backbone.py`),
`TRIVIAL` the null predictor, and `MARGIN` the noise threshold for every comparison.

`score.json` reports:
- `regime`: `gap_positive` (REFERENCE beats BACKBONE by more than MARGIN), `gap_negative` or
  `gap_negligible`.
- `gap_closed`: (score - BACKBONE) / (REFERENCE - BACKBONE), only in gap_positive. The analysis
  scripts generalize it to gap_negative tasks, where 1 means matching the better anchor (BACKBONE).
- `beat_trivial` and `below_both`: margin-gated flags.
- `model_choice_correct`: the self-reported model against the right choice. In gap_negative the
  right choice is to use no model.
- `calibration_error`: `expected_score` - score, at `reported` and `binding`.
- The run's cost: iterations, wall clock and tokens.

## Analysis

```bash
uv run python -m scripts.analysis.analysis_single_task runs/<grid> [runs/<grid> ...] --plots
uv run python -m scripts.analysis.analysis_cross_task runs/<grid> [...] --plots
uv run python -m scripts.analysis.analysis_cross_models runs/<grid> [...] [--group-by oracle] --plots
uv run python -m scripts.analysis.analysis_taxonomy runs/<grid> [...] --stage2-dir <dir> --plots
```

All scripts build one run table from the grid directories (`scripts/analysis/utils.py`) and write
to `analysis/<prefix>_<grid label>/`. `analysis_single_task` reports completion, replicate
variance, the noise floor and axis effect sizes for one task; `analysis_cross_task` pools tasks
on `gap_closed`; `analysis_cross_models` compares model families or oracle vs. baseline.
Definitions: [docs/analysis-metrics.md](docs/analysis-metrics.md); reading the figures:
[docs/analysis-plots.md](docs/analysis-plots.md). The outcome taxonomy (an LLM judge plus
clustering) is in `scripts/taxonomy/`.

Robustness checks and trace-derived tables:

```bash
uv run python -m scripts.analysis.trace_timing runs/<grid> [...] --out analysis/trace_facts.pkl
uv run python -m scripts.analysis.analysis_cross_task runs/<grid> [...] --robustness --trace-facts analysis/trace_facts.pkl
uv run python -m scripts.analysis.analysis_cross_models runs/<grid> [...] --trace-facts analysis/trace_facts.pkl [--group-by oracle]
uv run python -m scripts.analysis.analysis_tool_usage runs/<grid> [...]
```

`--robustness` adds bootstrap rank stability and partial η², budget effects on matched
configurations, margin sensitivity and the gap_negative analysis on raw score. With trace facts,
`analysis_cross_models` adds generation speed and, for oracle grids, oracle usage and hill-climbing.
`analysis_tool_usage` counts tool calls, specialist-model usage and web_fetch destinations.
`--table-cache <file.pkl>` caches the run table, which takes minutes to build from all traces.

Other tools: `trace_reconstruct.py` (exact model input at an iteration) and `trace_stats.py`
(per-run LLM latency and retries).

## Task contract

A task is three directories; `scripts.production.check_task` validates all of them.

**`tasks/<name>/`**, what the agent sees: `description.md` (task, data, metric; never which model
or how), `data/` (gitignored, built by `download.sh`), `models.txt` (every mounted model) and
`info/` (ladder snippets, never mounted directly).

**The information ladder** is defined by where each rung's content comes from:

| Rung | Answers | Content | Must not contain |
|---|---|---|---|
| `none` | | `description.md` only | any model name |
| `identity` | which model? | the relevant model's name | code or usage details |
| `interface` | how do I call it? | facts from the model's own docs, valid for any task | anything task-specific |
| `protocol` | what exactly do I do? | exactly what `run_reference.py` does | anything beyond it |

Running `protocol.md` must reproduce REFERENCE within MARGIN. `protocol` is a ceiling condition,
which splits the shortfall into a discovery gap (protocol - none) and an execution gap
(REFERENCE - protocol).

**`solutions/<name>/`**, never mounted: `solution.csv`, `eval_config.py`, `download.sh` and `dev/`
(`prepare_data.py`, `run_reference.py`, `run_backbone.py`). `dev/` scripts run in the model's
`dev_env`.

**`models/<name>/`**: `agent/` is mounted read-only at `/models/<name>` (`weights/`, `env/` built
inside the container, a minimal `README.md` without the task recipe); `dev_env/`, `verify.py` and
`download.sh` are for humans. See [models/astroclip/README.md](models/astroclip/README.md).

## Repo layout

```
agent/                 the agent loop, tools and prompts (runs only in the sandbox)
apptainer/             agent image definition; vLLM serving scripts
axes.py                experiment axes, model levels, pricing, RunSpec
orchestrate.py         launches, scores and renders one run
eval/                  scoring (evaluate.py, metrics.py) and the oracle server
specs/<task>/          grid specs
scripts/production/    grid runners (local and Slurm), check_task, env builder, API checks
scripts/analysis/      run tables, statistics and figures
scripts/taxonomy/      outcome taxonomy (Stage 1 extraction, Stage 2 clustering)
slurm/                 sbatch scripts; clusters/<name>.env profiles
tasks/ solutions/ models/   see Task contract
docs/                  analysis metrics, plots, Anthropic integration
tests/                 framework tests
runs/                  run outputs (gitignored)
```

## Data and models

The answer keys (`solutions/*/solution.csv`) are derived from public datasets; please cite the
original sources when using them.

| Task | Data | Specialist model |
|---|---|---|
| `mmlu-astronomy` | MMLU astronomy test split (`cais/mmlu`, MIT) | AstroSage-8B (Llama 3.1 Community License) |
| `promoter-prediction` | GUE `prom_core_tata` (Zhou et al., 2023; via `leannmlindsey/GUE`), derived from EPDnew | DNABERT-2 (code Apache-2.0) |
| `redshift-estimation` | DESI imaging and spectra via `EiffL/AstroCLIP` | AstroCLIP (code MIT) |
| `rna-folding` | bpRNA (Danaee et al., 2018), TR0/VL0/TS0 split from SPOT-RNA | RiNALMo (weights CC BY 4.0, code Apache-2.0) |

Model weights are downloaded by `models/<name>/download.sh` and not redistributed here.

## License

MIT, see [LICENSE](LICENSE). Datasets and model weights keep their own licenses.

## Citation

```bibtex
@misc{wiedmann2026agentssystemsmodelsrethinking,
      title={Agents Are Systems, Not Models: Rethinking Agentic Evaluation},
      author={Luis Wiedmann and Leander Girrbach and Cordelia Schmid and Zeynep Akata},
      year={2026},
      eprint={2610.01618},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2610.01618},
}
```

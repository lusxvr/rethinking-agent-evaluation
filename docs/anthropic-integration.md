# Claude Sonnet 5 as a model tier

`claude-sonnet-5` is the only hosted-API `model` level: every request is billed, and there is no
vLLM server. This document records how it differs from the vLLM tiers.

## Architecture

The vLLM path is written against vLLM's OpenAI-compatible API. Claude differs in client, tool
schema (`input_schema`), response shape (`thinking`/`text`/`tool_use` blocks), sampling parameters
and reasoning persistence. It is therefore a separate backend, `agent/backends/claude.py`, selected
by `config.provider`.

Both backends produce the same OpenAI-shaped message, raw output, usage and `RawToolCall` objects,
so tool dispatch, budget checks, trace logging and all analysis treat Claude runs like any other.
`agent/tools.py` stays the only tool definition; `_to_claude_tools` converts at the API boundary.

`trace_render.py`'s two LLM sections (summary and assessment) also route by provider, but are
skipped for Claude runs by default, since each would be a second billed call per run.

## Model axis: effort instead of sampling

Sonnet 5 rejects `temperature`, `top_p` and `top_k`; its only per-request lever is
`output_config.effort`. `axes.py` fixes `{"effort": "high"}` for both thinking modes, so effort
never varies with the harness level. "high" is the documented maximum for disabled thinking on
Opus 5 (unstated for Sonnet 5). The value travels as `EFFORT=high` through the same
`sampling_for()` path as vLLM presets; `Config.from_env` requires `EFFORT` instead of the vLLM
sampling variables when `PROVIDER=anthropic`.

The tier needs no server GPU (`MODEL_GPU_COUNT = 0`); the sandbox still gets one for the agent's
own code. `check_anthropic_credentials` replaces `check_backbone_model` and only checks that
`ANTHROPIC_API_KEY` is set.

## Harness axis: thinking blocks

vLLM's chat template drops reasoning from history, so the vLLM path writes it into the message
content as `Thought N:`. Claude keeps reasoning as signed `thinking` blocks, and requires the turn
that produced a `tool_use` to be replayed unchanged in the request carrying its `tool_result`.
`Conversation` caches each turn's native content blocks for this.

- `act-only`: `thinking: {"type": "disabled"}`. Disabled thinking can still leak reasoning into
  visible text; `check_content_chars` measures that, as on vLLM.
- `think-act`: adaptive thinking; thinking blocks are removed from every turn except the one whose
  `tool_result` is in the current request.
- `react`: adaptive thinking; all thinking blocks are kept.

`display: "summarized"` is required: the default (`"omitted"`) bills thinking but returns empty
text, which would break the `react` fold and make every run fail the manipulation check.

Removing old thinking blocks is allowed on Sonnet 5. Fable 5.1 and Mythos 5.1 reject edited
thinking history, so `think-act` would need server-side context management on those models.

## Tool calls

- Parallel tool results go back in one user message; splitting them discourages parallel calls.
- `tool_use.input` is already a dict, so there is no malformed-JSON case.
- Results starting with `"error:"` are sent with `is_error: true`.
- `tool_choice` is `auto`; `scripts/production/test_claude_call.py` checks it.
- All tools use `strict: true` with `additionalProperties: false`. Without it, Claude once omitted
  the required `expected_score` from `finish()`. Properties outside `required` (`model_used`)
  stay optional. Strict mode checks structure, not the content of string fields.

## Turns

- `MAX_TOKENS = 32000` covers thinking, text and the tool call. Requests are streamed, because
  the SDK refuses non-streaming requests it expects to take over ~10 minutes. A truncated turn is
  logged as `llm_truncated` and continues with whatever came back.
- `stop_reason: "refusal"` is logged as `llm_refusal` and handled like a turn without a tool call.
- Retries match the vLLM path. The per-call timeout ceiling is 600s (vLLM: 180s), capped by the
  remaining run budget.

## Infrastructure

- No extra network setup: the sandbox shares the host network, and on clusters that need an
  internet flag for workers (`WORKER_NEEDS_INTERNET_FLAG`), it also covers API calls.
- The API key reaches the container through `--env-file` (a 0600 file in the run's workdir,
  deleted afterwards), never `--env`, so it does not appear in the process list.

## Cost

List price: $2 / $10 per 1M input / output tokens. `estimated_cost_usd` prices every prompt token
at the full rate and is the number comparable across tiers. Prompt caching is on for every
request, and most of each turn's input is a cache read (10% of the input price), so
`axes.anthropic_actual_cost_usd` gives the billed cost from the four usage fields.

`think-act` costs more than `react` on this tier: removing thinking blocks changes the request
prefix and invalidates the cache from that point on. This follows from what `think-act` does and
is not a confound to remove, but it matters when comparing cost across harness levels.

## Slurm dispatch

`run_grid_slurm.py` and `continue_grid_slurm.py` give hosted-API tiers no server or teardown job.
The worker array gets the server id `"none"` (the worker scripts then skip the server wait) and
serves as the tier's completion job (`afterany`). `--preemptible` is rejected for these tiers.
`tests/test_slurm_hosted_api.py` covers the job chaining with a stubbed `sbatch`, not the sbatch
scripts themselves.

## Before a real run

1. `uv run pytest` (`tests/test_anthropic_backend.py` uses a stubbed client, no key needed).
2. Set `ANTHROPIC_API_KEY` in `.env`.
3. `uv run python -m scripts.production.test_claude_call` (one billed request).
4. One small run: `orchestrate.py --task mmlu-astronomy --model claude-sonnet-5 --budget short` on
   a GPU node, or a one-cell grid via `run_grid_slurm.py` on a cluster without interactive GPU access.
   Check that `trace_report.md`, `score.json` and the token counts look like a vLLM run's.

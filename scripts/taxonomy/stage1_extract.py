"""Stage 1: an LLM judge (default DeepSeek-V4-Flash-0731, not a Qwen model) describes each trajectory
along six free-text dimensions: result, error_category, model_usage, verification_behavior,
planning_exploration, execution_quality.

Every dimension gets the same context (build_common_context): task description, full trace,
string-matched model-engagement facts, filtered score facts (self-reports flagged), the task's
info/ snippets merged into one passage, and the final workspace. The run's axis levels are never
shown, so they cannot bias the later cross-tabulation.

Output: --output-dir/<grid-dir>/<dimension>.jsonl, resumable per (run, dimension).

Usage (one job per dimension):
  uv run python -m scripts.taxonomy.stage1_extract --dimensions model_usage
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from scripts.analysis import trace_render
from scripts.taxonomy import judge_server

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_TIMEOUT_S = 1200.0

SYSTEM_PROMPT = (
    "You are extracting structured information from a log of an autonomous research agent's "
    "attempt at a task, for building a taxonomy of run outcomes. You will be shown the task "
    "description, the complete raw event log (trace.jsonl -- one JSON object per line: run_start, "
    "llm_response, tool_call, status_message, run_end, in chronological order), mechanical facts "
    "computed from the trace and the task's scoring, the task's intended approach, and the agent's "
    "final workspace. Read everything before answering; do not guess from a partial read. Some "
    "scoring facts are the agent's own self-reported claims, marked as such below -- treat those as "
    "claims to confirm or contradict using the trace, not as established fact. Respond with a "
    "single JSON object matching exactly the fields requested, and nothing else -- no markdown "
    "fencing, no commentary outside the JSON object. Make sure the JSON is syntactically valid: "
    "escape any quotation marks that appear inside a string value."
)

# Shared response fields. "finding" and "why" replace specific model, library and dataset names
# with generic roles, so Stage 2 clusters by behavior rather than by task. "evidence_quote" stays
# verbatim.
RESPONSE_FORMAT_NOTE = (
    "\nRespond with a single JSON object with exactly these three fields:\n"
    "- \"finding\": {finding_hint}. Keep it short -- a few words, like a label, not a full "
    "sentence. Do not combine it with an explanation using a colon or \"because\"; put the "
    "explanation in \"why\" instead. Write it domain-agnostically: replace every specific "
    "model/library/dataset name with a generic role description (e.g. \"DNABERT-2\" -> \"the "
    "provided domain-specific model\", \"load_from_checkpoint\" -> \"the documented model-loading "
    "interface\"), so the same behavior in a different task reads the same way.\n"
    "- \"evidence_quote\": a short verbatim quote or tool-call excerpt from the trace that "
    "supports the finding -- keep this one exactly as it appears, task-specific names included. "
    "Quote the single most representative spot if the evidence spans several places.\n"
    "- \"why\": one sentence connecting the evidence to the finding, also written domain-"
    "agnostically like \"finding\" above."
)

# --domain-specific opts out of the above, back to the original wording (no name replacement) --
# kept for comparison re-runs, not the default.
RESPONSE_FORMAT_NOTE_DOMAIN_SPECIFIC = (
    "\nRespond with a single JSON object with exactly these three fields:\n"
    "- \"finding\": {finding_hint}. Keep it short -- a few words, like a label, not a full "
    "sentence. Do not combine it with an explanation using a colon or \"because\"; put the "
    "explanation in \"why\" instead.\n"
    "- \"evidence_quote\": a short verbatim quote or tool-call excerpt from the trace that "
    "supports the finding. Quote the single most representative spot if the evidence spans "
    "several places.\n"
    "- \"why\": one sentence connecting the evidence to the finding."
)

DIMENSIONS = {
    "result": {
        "label": "Result",
        "instruction": (
            "Describe, in your own words, how this run concluded and whether the result holds up, "
            "drawing on the scoring facts above. A clean, genuine success; a result that looks "
            "gamed or exploited somehow (hardcoded/memorized answers instead of a real method, "
            "peeking at solution or ground-truth files, exploiting a metric loophole, a "
            "suspiciously trivial method for a result this good); a crash or a submission that "
            "never became genuinely scoreable, and specifically what happened; or something else "
            "entirely. Say plainly if there's nothing questionable about it."
        ),
        "finding_hint": "how the run concluded, and whether it holds up",
    },
    # Root-cause discipline and the benign/harmful distinction paraphrase Zhu et al. (arXiv:2509.25370)
    # and Wang et al. (arXiv:2606.02060) -- cited for a human reader, not in the prompt itself.
    "error_category": {
        "label": "Error/failure category",
        "instruction": (
            "Name the single root cause of any problem with this run, not every symptom -- a "
            "cascading failure usually traces back to one earlier cause, and downstream symptoms "
            "of that same cause don't each count separately. Distinguish a genuine error from "
            "something that only looks like one: normal exploration, a failed attempt the agent "
            "recognized and moved past, or a hypothesis it later revised are not errors on their "
            "own. If there's no error attributable to a specific cause -- including for a run that "
            "succeeded cleanly -- say so explicitly with \"no_identifiable_error\"; don't invent a "
            "cause to fill the field."
        ),
        "finding_hint": "the root cause, or \"no_identifiable_error\"",
    },
    "model_usage": {
        "label": "Model/domain-model usage",
        "instruction": (
            "How did the agent engage with the domain-specific pretrained model(s) available for "
            "this task? Describe, in your own words, specifically what happened -- name the "
            "model(s) involved and what was actually done with them (or not done). Draw on the "
            "mechanical engagement signals and the agent's own self-reported model choice given "
            "above, alongside the trace itself. A few examples of the kind of thing to look for, "
            "not a list to pick from -- if what you observe is something else entirely, describe "
            "that instead: reading documentation before using a model correctly; using one "
            "correctly without apparently reading its docs first; loading a model but using it "
            "incorrectly in some specific way (say exactly how -- wrong input format, wrong "
            "preprocessing, misreading its output, using an untrained/randomly-initialized part of "
            "it, applying it to the wrong data); never loading or running it at all; using a "
            "different, unintended model instead."
        ),
        "finding_hint": "what was actually done with the model(s)",
    },
    "verification_behavior": {
        "label": "Verification behavior",
        "instruction": (
            "What did the agent actually do, if anything, to check its own work before "
            "submitting? Describe, in your own words, the specific check performed (or the "
            "absence of one) -- not just whether it was rigorous, but what it actually consisted "
            "of. This is about observed behavior, independent of what the task's own Verification "
            "condition nominally asked for or permitted."
        ),
        "finding_hint": "the verification actually performed, or its absence",
    },
    "planning_exploration": {
        "label": "Planning and exploration",
        "instruction": (
            "Describe the agent's overall approach and how it arrived at it, in your own words. "
            "Did it explore more than one substantively different approach before committing, or "
            "settle on one early and stick with it even where the trace shows evidence (an error, "
            "an unexpected result) that should have prompted reconsidering it? Name the specific "
            "approach(es) involved, not just whether it explored."
        ),
        "finding_hint": "the approach taken and how it was arrived at",
    },
    "execution_quality": {
        "label": "Execution quality",
        "instruction": (
            "Independent of whether the agent's plan was sound, was the plan implemented "
            "correctly? The agent's actual code is included above in its final workspace -- read "
            "it directly, not just the trace's account of it, to find concrete bugs. Describe, in "
            "your own words, any concrete implementation bugs found -- wrong preprocessing, "
            "off-by-one errors, misread columns or fields, output-format mismatches, incorrect use "
            "of an API or library -- or state plainly that none were found."
        ),
        "finding_hint": "the specific bug found, or that none was found",
    },
}

# Per-tier sampling from each model's HF README.
SAMPLING = {
    "qwen35-122b-a10b-fp8": {
        # Qwen3.5 README's "thinking mode, general tasks" preset, not "precise coding" (which
        # drops presence_penalty to 0) -- Stage 1 is general reasoning/judgment.
        "temperature": 1.0, "top_p": 0.95,
        "extra_sampling": {"top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0},
    },
    "deepseek-v4-flash-0731-fp8": {
        # Model card values for agentic use; presence_penalty=0.3 added to stop empty generations that
        # exhausted the token budget.
        "temperature": 1.0, "top_p": 0.95, "extra_sampling": {"presence_penalty": 0.3},
    },
    "mimo-v2-flash-fp8": {
        # XiaomiMiMo/MiMo-V2-Flash README's general-reasoning preset (temperature=0.8), not the
        # 0.3 preset scoped to agentic tool-use specifically.
        "temperature": 0.8, "top_p": 0.95, "extra_sampling": {},
    },
    "mimo-v2.5-fp8": {  # does not load in vLLM 0.28.0 (see vllm_server.sbatch)
        "temperature": 0.8, "top_p": 0.95, "extra_sampling": {},
    },
}


def find_trajectories(runs_dirs: list[Path]) -> list[Path]:
    # "-full-" skips smoke grids; matches runs/ or a single grid directory.
    matches = set()
    for runs_dir in runs_dirs:
        matches |= set(runs_dir.glob("*-full-*/*/trace.jsonl")) | set(runs_dir.glob("*/trace.jsonl"))
    return sorted(matches)


def grid_dir_name(trace_path: Path) -> str:
    """.../<grid-dir>/<run-dir>/trace.jsonl -> <grid-dir> -- output is keyed by this, mirroring
    runs/'s own structure, regardless of which --runs-dirs entry a trajectory came from."""
    return trace_path.parents[1].name


def load_context(trace_path: Path) -> tuple[str, str]:
    run_dir = trace_path.parent
    description_path = run_dir / "description.md"
    description_text = description_path.read_text(errors="replace") if description_path.exists() else ""
    trace_text = trace_path.read_text(errors="replace")
    return description_text, trace_text


def parse_events(trace_text: str) -> list[dict]:
    """Same parse as scripts/analysis/trace_render.py's render() -- one JSON object per line. Takes
    already-loaded text (load_context already reads the trace once) rather than re-reading the file."""
    return [json.loads(line) for line in trace_text.splitlines() if line]


# --- score.json, filtered and legended -- one block, same for every dimension ---

SCORE_FIELD_LEGEND = {
    "valid": "whether the submission was in a scoreable format at all",
    "metric": "which metric this task is scored on",
    "score": "this submission's value on that metric",
    "n": "number of items scored",
    "higher_is_better": "whether a higher metric value is better",
    "reference": "the reference (expert/specialist) solution's score",
    "backbone": "a plain backbone-only solution's score, with no domain model",
    "trivial": "a trivial baseline's score",
    "margin": "the noise floor for treating two scores as meaningfully different",
    "regime": "gap_positive (the domain model genuinely helps here), gap_negligible (doesn't "
              "matter), or gap_negative (would hurt) -- from comparing reference vs. backbone",
    "gap_closed": "only set in gap_positive: how much of the backbone-to-reference gap this "
                  "submission closed (1.0 = matched the reference, 0.0 = no better than the backbone)",
    "below_both": "whether this submission scored worse than both the reference and the backbone",
    "beat_trivial": "whether this submission beat the trivial baseline",
    "domain_model": "the domain-specific model this task made available (a task-level fact, not "
                     "specific to this run)",
    "expected_model": "which model choice is actually correct given the regime -- None where "
                       "neither choice is scoreable, or where declining every model is correct",
    "status": "how the run ended, e.g. 'finished', or a reason it was cut off",
    "iterations": "how many agent turns the run took",
    "wallclock_s": "how long the run took, in seconds",
    "prompt_tokens": "total prompt tokens consumed over the whole run",
    "completion_tokens": "total completion tokens consumed over the whole run",
}

# Dropped: derived from the unverified self-report and would pre-empt the judge's comparison.
EXCLUDED_SCORE_FIELDS = {"model_choice_correct", "calibration_error"}

# Self-reported via finish(); given to the judge as claims, not facts.
SELF_REPORTED_SCORE_FIELDS = {
    "model_used": "which model the agent itself claims to have used",
    "expected_score": "the agent's own guess at its score",
    "verification_evidence": "the agent's own description of how it verified its work",
}


def score_context(score: dict) -> str:
    if not score:
        return "## Scoring facts\n\nNo score.json found for this run.\n\n"
    lines = ["## Scoring facts (score.json -- computed by eval/evaluate.py, not LLM judgment, "
             "except where marked self-reported)\n"]
    for field, value in score.items():
        if field in EXCLUDED_SCORE_FIELDS:
            continue
        if field in SELF_REPORTED_SCORE_FIELDS:
            lines.append(f"- {field} = {value!r}: {SELF_REPORTED_SCORE_FIELDS[field]} -- "
                         f"SELF-REPORTED BY THE AGENT, NOT VERIFIED. Treat as a claim to confirm "
                         f"or contradict using the trace, not as fact.")
        else:
            desc = SCORE_FIELD_LEGEND.get(field, "(no description on file for this field)")
            lines.append(f"- {field} = {value!r}: {desc}")
    return "\n".join(lines) + "\n\n"


def mechanical_trace_facts(events: list[dict], task_name: str) -> str:
    """The task's candidate models and string-matched engagement signals per model (no LLM, no axis info)."""
    models = trace_render._candidate_models(task_name)
    if not models:
        return ""
    usage = trace_render._model_usage(events, models)
    lines = [
        "## Mechanical trace facts (model engagement -- computed by string-matching over tool "
        "calls, not LLM judgment; use these as a starting point, but read the trace yourself for "
        "the *how*, which these signals can't capture)\n",
        f"Candidate models for this task: {', '.join(models)}",
        "\nPer-candidate mechanical engagement signals "
        "(mentioned/read_docs/executed_referencing/researched_online):",
    ]
    for model in models:
        m = usage.get(model, {})
        lines.append(f"- {model}: mentioned={m.get('mentioned')}, read_docs={m.get('read_docs')}, "
                      f"executed_referencing={m.get('executed_referencing')}, "
                      f"researched_online={m.get('researched_online')}")
    return "\n".join(lines) + "\n\n"


# --- the task's intended approach, merged from its information ladder ---

INFO_LADDER_RUNGS = ["identity.md", "interface.md", "protocol.md"]  # same 3 files in all 4 tasks


def information_ladder_context(task_name: str) -> str:
    """All info/*.md snippets merged into one passage, regardless of the run's information level."""
    info_dir = REPO_ROOT / "tasks" / task_name / "info"
    parts = []
    for fname in INFO_LADDER_RUNGS:
        path = info_dir / fname
        if path.is_file():
            parts.append(path.read_text(errors="replace").strip())
    if not parts:
        return ""
    return "## Intended approach\n\n" + "\n\n".join(parts) + "\n\n"


# --- the agent's final workspace ---

WORKSPACE_PY_MAX_CHARS = 20_000  # larger files are flagged instead of embedded
WORKSPACE_CSV_HEAD_LINES = 20


def workspace_context(run_dir: Path) -> str:
    """Final workspace: .py files in full, the head of each .csv, else name and size."""
    workspace_dir = run_dir / "workspace"
    if not workspace_dir.is_dir():
        return "## Final workspace\n\nNo workspace/ directory found for this run.\n\n"

    sections = []
    for path in sorted(workspace_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace_dir)
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts[:-1]):
            continue
        size = path.stat().st_size
        if path.suffix == ".py":
            text = path.read_text(errors="replace")
            if len(text) <= WORKSPACE_PY_MAX_CHARS:
                sections.append(f"### {rel} ({size} bytes)\n```python\n{text}\n```")
            else:
                sections.append(f"### {rel} ({size} bytes) -- exceeds "
                                 f"{WORKSPACE_PY_MAX_CHARS} chars, name/size only")
        elif path.suffix == ".csv":
            lines = path.read_text(errors="replace").splitlines()
            head = "\n".join(lines[:WORKSPACE_CSV_HEAD_LINES])
            sections.append(f"### {rel} ({size} bytes, {len(lines)} lines) -- first "
                             f"{min(WORKSPACE_CSV_HEAD_LINES, len(lines))} lines:\n```\n{head}\n```")
        else:
            sections.append(f"### {rel} ({size} bytes)")
    if not sections:
        return "## Final workspace\n\nworkspace/ exists but is empty.\n\n"
    return "## Final workspace (files the agent left behind)\n\n" + "\n\n".join(sections) + "\n\n"


def build_common_context(trace_path: Path, score: dict) -> str:
    """Shared context: task description, full trace, then the facts, approach and workspace blocks."""
    description_text, trace_text = load_context(trace_path)
    events = parse_events(trace_text)
    run_start = next((e for e in events if e.get("event") == "run_start"), {})
    task_name = run_start.get("task_name", "unknown")

    return (
        f"## Task description\n\n{description_text}\n\n"
        f"## Raw agent trace (trace.jsonl)\n\n{trace_text}\n\n"
        f"{mechanical_trace_facts(events, task_name)}"
        f"{score_context(score)}"
        f"{information_ladder_context(task_name)}"
        f"{workspace_context(trace_path.parent)}"
    )


def build_messages(label: str, instruction: str, finding_hint: str, common_context: str,
                    domain_agnostic: bool = True) -> list[dict]:
    note = RESPONSE_FORMAT_NOTE if domain_agnostic else RESPONSE_FORMAT_NOTE_DOMAIN_SPECIFIC
    user = (
        f"{common_context}"
        f"## Extraction task: {label}\n\n{instruction}\n"
        f"{note.format(finding_hint=finding_hint)}"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _one_attempt(client: OpenAI, model: str, messages: list[dict], sampling: dict, max_tokens: int) -> dict:
    """One API call, parsed; split out so a parse failure (empty or truncated output) can be retried."""
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=sampling["temperature"], top_p=sampling["top_p"],
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, **sampling["extra_sampling"]},
    )
    usage = response.usage.model_dump() if response.usage else {}
    message = response.choices[0].message
    content = message.content or ""

    parsed, parse_error = None, None
    text = content.strip()
    if text.startswith("```"):  # defensive -- the system prompt says not to, but strip markdown fencing if present
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        parse_error = str(exc)

    return {
        "finish_reason": response.choices[0].finish_reason,
        "parsed": parsed, "parse_error": parse_error,
        "content": content, "reasoning_content": getattr(message, "reasoning_content", None),
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
    }


def call_one(trace_path: Path, dimension_key: str, base_url: str, model: str, tier: str, max_tokens: int,
             score: dict, domain_agnostic: bool = True) -> dict:
    """dimension_key is one of DIMENSIONS' keys."""
    client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=CLIENT_TIMEOUT_S, max_retries=0)
    run_name = trace_path.parent.name
    key = f"{run_name}::{dimension_key}"
    sampling = SAMPLING[tier]
    start = time.time()
    try:
        dim = DIMENSIONS[dimension_key]
        common_context = build_common_context(trace_path, score)
        messages = build_messages(dim["label"], dim["instruction"], dim["finding_hint"], common_context, domain_agnostic)
        attempt = _one_attempt(client, model, messages, sampling, max_tokens)
        if attempt["parse_error"]:
            attempt = _one_attempt(client, model, messages, sampling, max_tokens)  # one retry
        latency_s = time.time() - start

        return {
            "key": key, "run": run_name, "dimension": dimension_key, "ok": True,
            "latency_s": latency_s, **attempt,
        }
    except Exception as exc:
        return {
            "key": key, "run": run_name, "dimension": dimension_key, "ok": False,
            "error": f"{type(exc).__name__}: {exc}", "latency_s": time.time() - start,
        }


def run_extraction(args, base_url: str, model: str, trajectories: list[Path], out_dir: Path) -> None:
    # Resumable across restarts (not concurrent runs). Only ok:true records count as done, so failed
    # calls are retried. One output file per grid dir and dimension.
    by_grid: dict[str, list[Path]] = {}
    for trace_path in trajectories:
        by_grid.setdefault(grid_dir_name(trace_path), []).append(trace_path)

    for dimension_key in args.dimensions:
        for grid_name, grid_trajectories in by_grid.items():
            out_path = out_dir / grid_name / f"{dimension_key}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            done_keys: set[str] = set()
            if out_path.exists():
                for line in out_path.read_text().splitlines():
                    if line.strip():
                        record = json.loads(line)
                        if record.get("ok"):
                            done_keys.add(record["key"])

            tasks = []
            for trace_path in grid_trajectories:
                key = f"{trace_path.parent.name}::{dimension_key}"
                if key in done_keys:
                    continue
                score_path = trace_path.parent / "score.json"
                score = json.loads(score_path.read_text()) if score_path.is_file() else {}
                tasks.append((trace_path, score))

            label = f"{grid_name}/{dimension_key}"
            print(f"[{label}] {len(tasks)} call(s) to run ({len(done_keys)} already done), "
                  f"concurrency={args.max_num_seqs}")
            if not tasks:
                continue

            wall_start = time.time()
            n_done, n_ok, n_failed = 0, 0, 0
            with out_path.open("a") as f, ThreadPoolExecutor(max_workers=args.max_num_seqs) as pool:
                futures = {pool.submit(call_one, p, dimension_key, base_url, model, args.tier, args.max_tokens,
                                        score, args.domain_agnostic): p
                           for p, score in tasks}
                for future in as_completed(futures):
                    p = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        print(f"warning: {p.parent.name} crashed: {exc}", file=sys.stderr)
                        continue
                    f.write(json.dumps(record) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    n_done += 1
                    n_ok += int(record["ok"])
                    n_failed += int(not record["ok"])
                    if n_done % 25 == 0 or n_done == len(tasks):
                        elapsed = time.time() - wall_start
                        print(f"[{label}] {n_done}/{len(tasks)} done ({n_ok} ok, {n_failed} failed), "
                              f"{elapsed:.0f}s elapsed, {n_done/elapsed*3600:.0f} calls/hour")
            print(f"[{label}] finished: {n_ok} ok, {n_failed} failed, wrote to {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    all_dims = sorted(DIMENSIONS)
    parser.add_argument("--tier", default="deepseek-v4-flash-0731-fp8",
                         help="scripts/taxonomy/vllm_server.sbatch tier id -- fixed judge, see module docstring")
    parser.add_argument("--dimensions", nargs="+", choices=all_dims, default=None,
                         help=f"one or more of {all_dims} (default: all six)")
    parser.add_argument("--domain-specific", action="store_true",
                         help="keep specific model/library names in finding/why (comparison runs only; use a separate --output-dir)")
    parser.add_argument("--max-num-seqs", type=int, default=10,
                         help="server batch capacity and client request concurrency -- same number, one flag")
    parser.add_argument("--max-model-len", type=int, default=524288,
                         help="server context length; below the model's 1M, since long prefills can OOM")
    parser.add_argument("--gpu-mem-util", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=32768, help="completion budget per call, reasoning included")
    parser.add_argument("--ready-timeout-s", type=int, default=86400,
                         help="give up on (and cancel, unless --keep-server) a server not ready after this many seconds; queues can be long")
    parser.add_argument("--runs-dirs", nargs="+", default=[str(REPO_ROOT / "runs")],
                         help="grid directories, or a parent such as runs/ to find every *-full-* grid")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "scripts/taxonomy/stage1_output"),
                         help="base directory; output goes to <output-dir>/<grid-dir>/<dimension>.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="cap trajectory count -- for validation runs")
    parser.add_argument("--run-filter", default=None, help="only run names containing this substring -- for validation runs")
    parser.add_argument("--label", default="stage1-extract", help="only used for server job naming/logging")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--attach-job-id", default=None)
    parser.add_argument("--dependency", default=None,
                         help="Slurm --dependency for the judge server job (ignored with --attach-job-id)")
    args = parser.parse_args()
    if args.tier not in SAMPLING:
        raise SystemExit(f"no SAMPLING entry for tier {args.tier!r}")
    args.dimensions = args.dimensions or all_dims
    args.domain_agnostic = not args.domain_specific

    trajectories = find_trajectories([Path(d) for d in args.runs_dirs])
    if not trajectories:
        raise SystemExit(f"no trace.jsonl found under any of {args.runs_dirs}")
    if args.run_filter:
        trajectories = [p for p in trajectories if args.run_filter in p.parent.name]
    if args.limit:
        trajectories = trajectories[:args.limit]
    print(f"[{args.label}] {len(trajectories)} trajectories in scope, dimensions={args.dimensions}, tier={args.tier}")

    cache_root_result = judge_server._run(["bash", "-c", "set -a; . .env; set +a; echo $CACHE_ROOT"], cwd=REPO_ROOT)
    cache_root = cache_root_result.stdout.strip()
    if not cache_root:
        raise SystemExit("could not resolve CACHE_ROOT from .env")

    job_id = args.attach_job_id or judge_server.submit_server(args)
    try:
        log_path = judge_server.wait_ready(args, job_id, cache_root, args.ready_timeout_s)
        addr = Path(cache_root, f"taxonomy-judge-{job_id}.addr").read_text().strip()
        resources = judge_server.capture_resources(args, job_id, log_path, addr)
        if resources["served_model_id"] is None:
            raise SystemExit(f"[{args.label}] server answered readiness probe but /v1/models had no data")
        model = resources["served_model_id"]
        base_url = f"http://{addr}/v1"
        run_extraction(args, base_url, model, trajectories, Path(args.output_dir))
    finally:
        if not args.keep_server:
            print(f"[{args.label}] tearing down job {job_id}")
            judge_server._run(["scancel", job_id])
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Render trace.jsonl as a markdown report; the single place deriving summary quantities from a trace."""

import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import anthropic
from openai import OpenAI

from agent.config import generates_reasoning, is_verified, persists_reasoning
from axes import DEFAULT_MODEL_NAME, INFORMATION_LEVELS, info_snippets_for, provider_for

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MAX_BLOCK_CHARS = 3000
MAX_SUMMARY_INPUT_CHARS = 200_000  # ~50k tokens, comfortably inside the server's context budget
SUMMARY_PROMPT = """You are analyzing a log of an autonomous coding agent's attempt at a research \
task. You'll be given the full rendered trace (header stats, tool calls, reasoning, and the final \
result), and may also be given reference documentation for the candidate model(s) involved -- \
written by the humans who built this benchmark, describing the actual correct methodology and \
known gotchas, never shown to the agent itself. Write a short (4-6 sentence) analysis for a human \
skimming this trace: what the agent tried, the key turning point(s) or mistake(s) that shaped the \
outcome, and why it succeeded or failed. If reference documentation is given, use it to name the \
specific point(s) where the agent's approach diverged from the documented correct one, rather than \
just describing what the agent did in isolation. Be concrete (cite specific errors/decisions from \
the trace) rather than generic. Do not repeat the header stats verbatim, just interpret them."""
VERIFICATION_ASSESSMENT_PROMPT = """You are auditing a log of an autonomous research agent's \
attempt at a task. You'll be given the full rendered trace (tool calls, reasoning, and the final \
result). Judge two things a simple keyword search cannot: whether the agent's self-verification \
was actually adequate, not just present, and whether its first real mistake would have been \
visible to it at the time or not.

"Adequate" verification actually tests correctness -- checking a result against something \
independent, sanity-checking values against domain expectations, re-deriving a number a different \
way. Confirming a file exists, parses, or has the expected shape/columns is "superficial": real, \
but it cannot catch a wrong answer, only a malformed one. "loud" means a tool call actually \
errored or produced something the agent itself flagged as wrong. "silent" means nothing looked \
wrong to the agent at any point, but the approach was flawed anyway (a missing preprocessing \
step, an unvalidated assumption, a metric computed against the wrong baseline). If the run's \
approach looks sound throughout, use "none" for FAILURE_STAGE and FAILURE_MODE.

Answer in exactly this format, one line per field, nothing before or after:
VERIFICATION: <adequate|superficial|none>
VERIFICATION_REASON: <one sentence, citing what it did or didn't check>
FAILURE_STAGE: <a short phrase naming what step first went wrong, or "none">
FAILURE_MODE: <loud|silent|none>
FAILURE_REASON: <one sentence>"""
_READ_COMMAND_RE = re.compile(r"\b(cat|head|tail|less|more)\b")
_EXEC_COMMAND_RE = re.compile(r"\b(python3?|uv run)\b")
_URL_RE = re.compile(r"https?://[^\s)\]]+")


def _truncate(text: str, limit: int = MAX_BLOCK_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters -- see trace.jsonl for the full value]"


def _fence_for(text: str) -> str:
    """A fence longer than any backtick run already in the content -- a fixed 3-backtick fence closes early on agent content with its own nested ```-fences, corrupting everything after it."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _block(label: str, text: str, truncate: bool = False, lang: str = "") -> list[str]:
    """A labeled, fenced block -- the one visual pattern used for every kind of content in the transcript."""
    body = _truncate(text) if truncate else text
    fence = _fence_for(body)
    return [label, "", f"{fence}{lang}", body, fence, ""]


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _tool_call_ok(event: dict) -> bool:
    """Whether a tool_call event's result reads as a success -- run_bash checks its own exit_code, every other tool an "error: ..." prefix."""
    content = event.get("message", {}).get("content", "")
    if event.get("name") == "run_bash":
        match = re.match(r"exit_code: (-?\d+)", content)
        return bool(match) and match.group(1) == "0"
    return not content.startswith("error:")


def _tool_call_stats(events: list[dict]) -> list[tuple[str, int, int]]:
    """Per-tool (name, ok_count, fail_count), in first-seen order."""
    counts: dict[str, list[int]] = {}
    for e in events:
        if e["event"] != "tool_call":
            continue
        bucket = counts.setdefault(e.get("name", "unknown"), [0, 0])
        bucket[0 if _tool_call_ok(e) else 1] += 1
    return [(name, ok, fail) for name, (ok, fail) in counts.items()]


def _web_fetches(events: list[dict]) -> list[tuple[str, str]]:
    """(url, status) for every web_fetch call, in call order -- lets a reader see at a
    glance what the model actually pulled off the internet without reading every iteration."""
    results = []
    for e in events:
        if e["event"] != "tool_call" or e.get("name") != "web_fetch":
            continue
        url = e.get("arguments", {}).get("url", "?")
        content = e.get("message", {}).get("content", "")
        match = re.match(r"status: (\d+(?:\s*\(redirect[^)]*\))?)", content)
        if match:
            status = match.group(1)
        elif content.startswith("error:"):
            status = "error"
        else:
            status = "?"
        results.append((url, status))
    return results


def _candidate_models(task: str) -> list[str]:
    models_file = REPO_ROOT / "tasks" / task / "models.txt"
    if not models_file.is_file():
        return []
    return [line.strip() for line in models_file.read_text().splitlines() if line.strip()]


def _model_reference_urls(model: str) -> list[str]:
    """URLs (model card, paper, code repo) listed in a model's own agent/README.md -- the set of
    "look it up online" targets a well-behaved agent would plausibly fetch."""
    readme = REPO_ROOT / "models" / model / "agent" / "README.md"
    if not readme.is_file():
        return []
    return _URL_RE.findall(readme.read_text())


def _model_dev_readmes(models: list[str]) -> str:
    """The human-dev-targeted models/<name>/README.md, never mounted into the agent's sandbox --
    safe for the post-hoc summarizer since the run is already over by the time it's read."""
    sections = []
    for model in models:
        readme = REPO_ROOT / "models" / model / "README.md"
        if readme.is_file():
            sections.append(f"### {model}\n\n{readme.read_text()}")
    return "\n\n".join(sections)


_GITHUB_HOSTS = {"github.com", "raw.githubusercontent.com"}


def _github_repo_key(parsed) -> str | None:
    """"owner/repo" for github.com and raw.githubusercontent.com URLs, so both count as the same repo."""
    if parsed.netloc.lower().removeprefix("www.") not in _GITHUB_HOSTS:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    return "/".join(parts[:2]).lower() if len(parts) >= 2 else None


def _same_resource(url_a: str, url_b: str) -> bool:
    """Whether two URLs plausibly point at the same resource, not just an exact string match (e.g. an arxiv /pdf/ vs. /abs/ link)."""
    a, b = urlparse(url_a), urlparse(url_b)
    repo_a, repo_b = _github_repo_key(a), _github_repo_key(b)
    if repo_a and repo_b:
        return repo_a == repo_b
    host_a, host_b = a.netloc.lower().removeprefix("www."), b.netloc.lower().removeprefix("www.")
    if host_a != host_b:
        return False
    path_a, path_b = a.path.rstrip("/"), b.path.rstrip("/")
    if path_a == path_b:
        return True
    if path_a and path_b and (path_a.startswith(path_b) or path_b.startswith(path_a)):
        return True
    # Last path segment as a fallback (handles e.g. an arxiv id shared across /abs/ vs /pdf/ URLs).
    seg_a, seg_b = path_a.rsplit("/", 1)[-1], path_b.rsplit("/", 1)[-1]
    return bool(seg_a) and seg_a == seg_b


def _model_usage(events: list[dict], models: list[str]) -> dict[str, dict[str, bool]]:
    """Heuristic engagement evidence per candidate model, in increasing strength:
      - mentioned: its mount path appears in a tool call.
      - read_docs: read_file or cat/head/tail on its path.
      - executed_referencing: python/uv run against its path, exit 0.
      - researched_online: web_fetch of a URL linked in its agent/README.md.
    """
    usage = {
        model: {"mentioned": False, "read_docs": False, "executed_referencing": False, "researched_online": False}
        for model in models
    }
    reference_urls = {model: _model_reference_urls(model) for model in models}

    for e in events:
        if e["event"] != "tool_call":
            continue
        name = e.get("name")
        arguments = e.get("arguments") or {}
        result_text = (e.get("message") or {}).get("content", "") or ""
        blob = json.dumps(arguments) + "\n" + result_text

        if name == "web_fetch":
            fetched_url = arguments.get("url", "")
            for model in models:
                if any(_same_resource(fetched_url, ref) for ref in reference_urls[model]):
                    usage[model]["researched_online"] = True

        for model in models:
            mount = f"/models/{model}"
            if mount not in blob:
                continue
            usage[model]["mentioned"] = True

            if name == "read_file" and str(arguments.get("path", "")).startswith(mount):
                usage[model]["read_docs"] = True

            if name == "run_bash":
                command = arguments.get("command", "")
                if mount in command and _READ_COMMAND_RE.search(command):
                    usage[model]["read_docs"] = True
                if mount in command and _EXEC_COMMAND_RE.search(command) and "exit_code: 0" in result_text:
                    usage[model]["executed_referencing"] = True

    return usage


def _self_report_unsupported(self_reported: str | None, model_usage: dict[str, dict[str, bool]]) -> bool:
    """The agent named a model in finish() that the trace shows no successful execution against."""
    return bool(self_reported) and not model_usage.get(self_reported, {}).get("executed_referencing")


def _parse_float(value: str) -> float | None:
    """None for non-numeric fields -- nvidia-smi reports utilization.gpu as "[N/A]" on MIG."""
    try:
        return float(value)
    except ValueError:
        return None


def _resource_stats(resources_path: Path) -> list[dict]:
    """Per-physical-GPU memory/power/utilization min/max/mean -- shared by every MIG tenant, not isolated to this run's slice."""
    if not resources_path.is_file():
        return []
    per_gpu: dict[str, dict] = {}
    for line in resources_path.read_text().splitlines():
        if not line:
            continue
        for row in json.loads(line).get("gpus", []):
            if len(row) < 6:
                continue
            idx = row[0]
            mem_used, mem_total, power = _parse_float(row[3]), _parse_float(row[4]), _parse_float(row[5])
            if mem_used is None or mem_total is None or power is None:
                continue
            bucket = per_gpu.setdefault(idx, {"mem": [], "power": [], "util": [], "mem_total": mem_total})
            bucket["mem"].append(mem_used)
            bucket["power"].append(power)
            util = _parse_float(row[2])
            if util is not None:
                bucket["util"].append(util)
    stats = []
    for idx, bucket in sorted(per_gpu.items()):
        mem, power, util = bucket["mem"], bucket["power"], bucket["util"]
        stats.append(
            {
                "index": idx,
                "mem_total": bucket["mem_total"],
                "mem_min": min(mem),
                "mem_max": max(mem),
                "mem_mean": sum(mem) / len(mem),
                "power_min": min(power),
                "power_max": max(power),
                "power_mean": sum(power) / len(power),
                "util_min": min(util) if util else None,
                "util_max": max(util) if util else None,
                "util_mean": sum(util) / len(util) if util else None,
                "n": len(mem),
            }
        )
    return stats


def _truncate_for_summary(text: str, limit: int = MAX_SUMMARY_INPUT_CHARS) -> str:
    """Keeps the header plus as much of the tail as fits -- a run's resolution lives in the tail, not the middle."""
    if len(text) <= limit:
        return text
    head, tail_budget = text[:2000], limit - 2000
    return f"{head}\n\n... [middle of trace truncated for length] ...\n\n{text[-tail_budget:]}"


def _llm_section(system_prompt: str, user_content: str, base_url: str, model: str, max_tokens: int) -> str | None:
    """One post-hoc LLM judge call on a rendered trace; returns None on any error."""
    try:
        if provider_for(model) == "anthropic":
            response = anthropic.Anthropic(timeout=60.0).messages.create(
                model=model, max_tokens=max_tokens, system=system_prompt, thinking={"type": "disabled"},
                messages=[{"role": "user", "content": user_content}],
            )
            return "".join(b.text for b in response.content if b.type == "text") or None
        client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=60.0)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"warning: LLM trace section skipped ({exc!r})", file=sys.stderr)
        return None


# VERIFICATION_ASSESSMENT_PROMPT's reply format -> (analysis_generated.json field, allowed values or None).
_ASSESSMENT_FIELDS = {
    "VERIFICATION": ("verification", ("adequate", "superficial", "none")),
    "VERIFICATION_REASON": ("verification_reason", None),
    "FAILURE_STAGE": ("failure_stage", None),
    "FAILURE_MODE": ("failure_mode", ("loud", "silent", "none")),
    "FAILURE_REASON": ("failure_reason", None),
}


def parse_verification_assessment(text: str | None) -> dict:
    """The judge's reply as fields. Enum answers are lowercased and unwrapped from the markdown a model
    sometimes adds; one outside the allowed set is kept verbatim rather than dropped to None."""
    parsed = {field: None for field, _ in _ASSESSMENT_FIELDS.values()}
    for line in (text or "").splitlines():
        key, separator, value = line.partition(":")
        entry = _ASSESSMENT_FIELDS.get(key.strip().strip("*`# ").upper())
        if not separator or entry is None:
            continue
        field, allowed = entry
        value = value.strip()
        if allowed is not None:
            value = value.strip("*`_. ").lower()
        parsed[field] = value or None
    return parsed


def _snippet_text(task: str, name: str) -> str:
    """A task's info/<name>.md as compose_description embeds it; "" if the task has no such rung."""
    path = REPO_ROOT / "tasks" / task / "info" / f"{name}.md"
    return path.read_text().rstrip("\n") if path.is_file() else ""


def _folded_reasoning(event: dict) -> bool:
    """Whether this turn's reasoning was folded into the message history -- exact rather than
    matched on a prefix, since that fold is the only thing making the two channels differ."""
    return ((event.get("message") or {}).get("content") or "") != ((event.get("raw_output") or {}).get("content") or "")


def _manipulation_checks(events: list[dict], run_start: dict, run_end: dict) -> dict:
    """Whether each axis's manipulation reached the agent, e.g. that react actually kept reasoning.

    check_violations compares measured counts with the run's labels. Information is checked against
    the current info/ snippets.
    """
    llm_events = [e for e in events if e["event"] == "llm_response"]
    # reasoning/content both read from raw_output, the model's own reply, so folding the thought
    # into the persisted message can't inflate either count.
    raw_outputs = [e.get("raw_output") or {} for e in llm_events]
    persisted = [e for e in llm_events if _folded_reasoning(e)]
    turns, reasoning_turns = len(llm_events), sum(1 for r in raw_outputs if r.get("reasoning"))
    violations: list[str] = []

    harness = run_start.get("harness")
    if harness and turns:
        for mechanism, expected, actual in (
            ("generate reasoning", generates_reasoning(harness), reasoning_turns),
            ("persist reasoning into context", persists_reasoning(harness), len(persisted)),
        ):
            if expected and not actual:
                violations.append(f"harness={harness} should {mechanism}, no turn did")
            elif actual and not expected:
                violations.append(f"harness={harness} should not {mechanism}, {actual} turn(s) did")

    information, task = run_start.get("information"), run_start.get("task_name")
    seed_user = next(
        (m.get("content") or "" for m in reversed(run_start.get("seed_messages", [])) if m.get("role") == "user"), None
    )
    checked_info = bool(information and task and seed_user is not None)
    granted: tuple[str, ...] = ()
    found: list[str] = []
    if checked_info:
        # Which rungs' text is in the description at all, then split by whether it should be. Both
        # directions matter: a missing rung weakens the treatment, a leaked one contaminates the control.
        present = [name for name in INFORMATION_LEVELS[1:] if (text := _snippet_text(task, name)) and text in seed_user]
        granted = info_snippets_for(information)
        found = [name for name in present if name in granted]
        if missing := [name for name in granted if name not in found]:
            violations.append(f"information={information} grants {list(granted)}, missing from the description: {missing}")
        if leaked := [name for name in present if name not in granted]:
            violations.append(f"information={information} withholds the rungs above it, but the description carries: {leaked}")

    verification = run_start.get("verification")
    if verification and run_end.get("status") == "finished":
        reported = run_end.get("expected_score") is not None and bool(run_end.get("verification_evidence"))
        if is_verified(verification) and not reported:
            violations.append(f"verification={verification} requires expected_score/verification_evidence, finish() carried neither")
        elif reported and not is_verified(verification):
            violations.append(f"verification={verification} should not accept expected_score/verification_evidence, finish() carried them")

    return {
        "check_llm_turns": turns,
        "check_reasoning_turns": reasoning_turns,
        "check_reasoning_chars": sum(len(r.get("reasoning") or "") for r in raw_outputs),
        # The covariate for whether act-only just relocates its reasoning into the content channel,
        # which disabling the reasoning channel does not prevent.
        "check_content_chars": sum(len(r.get("content") or "") for r in raw_outputs),
        "check_persisted_turns": len(persisted),
        "check_info_snippets_found": len(found) if checked_info else None,
        "check_info_snippets_granted": len(granted) if checked_info else None,
        # Never a violation: a run that finished early was simply never given a budget treatment,
        # and a budget level nothing reaches is a level that measures nothing.
        "budget_bound": run_end.get("status") == "max_duration_reached" if run_end else None,
        "check_violations": "; ".join(violations) or None,
    }


def render(trace_path: Path, base_url: str | None = None, model: str | None = None) -> tuple[str, dict]:
    """Returns (trace_report.md, analysis_generated.json) -- the analysis holding the model-usage evidence and the judge's parsed fields."""
    events = [json.loads(line) for line in trace_path.read_text().splitlines() if line]
    run_start = next((e for e in events if e["event"] == "run_start"), {})
    run_end = next((e for e in events if e["event"] == "run_end"), {})

    task_name = run_start.get("task_name", "unknown")
    models = _candidate_models(task_name)
    self_reported = run_end.get("model_used")
    model_usage = _model_usage(events, models) if models else {}

    lines: list[str] = []
    lines.append(f"# Agent trajectory: {task_name}")
    lines.append("")
    if run_start:
        lines.append(f"- **Model:** {run_start.get('model_name', 'unknown')}")
        lines.append(f"- **GPU:** {run_start.get('gpu', run_start.get('gpu_uuid', 'unknown'))}")
        sampling = run_start.get("sampling")
        if sampling:
            lines.append("- **Sampling:** " + ", ".join(f"{k}={v}" for k, v in sampling.items()))
    if run_end:
        lines.append(f"- **Status:** `{run_end.get('status')}`")
        if run_end.get("submission_path"):
            lines.append(f"- **Submission:** `{run_end['submission_path']}`")
        if models:
            lines.append(f"- **Domain model used:** `{self_reported}`" if self_reported else "- **Domain model used:** (none reported)")

    eval_path = trace_path.parent / "score.json"
    result = json.loads(eval_path.read_text()) if eval_path.is_file() else None
    if result is not None:
        if result.get("valid"):
            lines.append(f"- **{result['metric']}:** {result['score']:.4f} ({result['n']} rows)")
            gap_closed = result.get("gap_closed")
            gap_part = f", gap_closed {gap_closed:.4f}" if gap_closed is not None else ""
            lines.append(
                f"- **Regime:** `{result.get('regime')}`{gap_part} "
                f"(below_both={result.get('below_both')}, beat_trivial={result.get('beat_trivial')})"
            )
        else:
            lines.append(f"- **Evaluation:** INVALID -- {'; '.join(result.get('errors', []))}")

    # The run's own totals, as the agent recorded them in run_end.
    if run_end.get("iterations") is not None:
        lines.append(f"- **Iterations:** {run_end['iterations']} (max {run_start.get('max_iterations', '?')})")
    if run_end.get("wallclock_s") is not None:
        max_duration_s = run_start.get("max_duration_s")
        budget_part = f" (max {_format_duration(max_duration_s)})" if max_duration_s is not None else ""
        lines.append(f"- **Duration:** {_format_duration(run_end['wallclock_s'])}{budget_part}")
    if run_end.get("completion_tokens") is not None:
        # prompt_tokens is cumulative input volume across calls (each iteration resends the whole conversation), not distinct tokens; completion_tokens has no such overlap.
        prompt_tokens, completion_tokens = run_end["prompt_tokens"], run_end["completion_tokens"]
        lines.append(
            f"- **Tokens:** {prompt_tokens:,} prompt (cumulative across calls, not unique) "
            f"+ {completion_tokens:,} completion = {prompt_tokens + completion_tokens:,} total"
        )

    checks = _manipulation_checks(events, run_start, run_end)
    if checks["check_llm_turns"]:
        lines.append(
            f"- **Manipulation checks:** reasoning generated on {checks['check_reasoning_turns']}"
            f"/{checks['check_llm_turns']} turns, persisted into context on {checks['check_persisted_turns']}"
            f"; info snippets {checks['check_info_snippets_found']}/{checks['check_info_snippets_granted']}"
            f"; budget bound: {checks['budget_bound']}"
            + (f" -- **VIOLATIONS: {checks['check_violations']}**" if checks["check_violations"] else " -- all as specified")
        )

    tool_stats = _tool_call_stats(events)
    if tool_stats:
        total = sum(ok + fail for _, ok, fail in tool_stats)
        breakdown = ", ".join(f"{name} {ok + fail} ({ok} ok, {fail} failed)" for name, ok, fail in tool_stats)
        lines.append(f"- **Tool calls:** {total} total -- {breakdown}")

    lines.append("")
    header_end = len(lines)  # insertion point for the LLM summary, if any -- after the mechanical
    # header stats above, before the trace detail below.

    web_fetches = _web_fetches(events)
    if web_fetches:
        lines.append("## Fetched URLs")
        lines.append("")
        lines.extend(f"- `{status}` {url}" for url, status in web_fetches)
        lines.append("")

    if models:
        lines.append("## Model usage")
        lines.append("")
        lines.append("Text-pattern evidence of engagement with each candidate model -- a diagnostic, not a precise measurement.")
        lines.append("")
        for candidate, evidence in model_usage.items():
            tiers = [tier for tier, seen in evidence.items() if seen]
            lines.append(f"- **{candidate}:** {', '.join(tiers) if tiers else '(no trace evidence)'}")
        if _self_report_unsupported(self_reported, model_usage):
            lines.append(
                f"- note: self-reported model `{self_reported}` has no 'executed_referencing' trace evidence"
            )
        lines.append("")

    resource_stats = _resource_stats(trace_path.parent / "resources.jsonl")
    if resource_stats:
        lines.append("## Node GPU Load")
        lines.append("")
        lines.append("Sampled ~5s at the physical-GPU level -- on MIG this is **all** tenants' load, not just this run's slice.")
        lines.append("")
        for s in resource_stats:
            if s["util_mean"] is not None:
                util_part = f"; utilization {s['util_min']:.0f}-{s['util_max']:.0f}% (avg {s['util_mean']:.0f}%)"
            else:
                util_part = "; utilization N/A (MIG-partitioned GPU)"
            lines.append(
                f"- **GPU {s['index']}:** memory {s['mem_min']:.0f}-{s['mem_max']:.0f} MiB "
                f"(avg {s['mem_mean']:.0f}, of {s['mem_total']:.0f} total); "
                f"power {s['power_min']:.0f}-{s['power_max']:.0f} W "
                f"(avg {s['power_mean']:.0f}){util_part} -- {s['n']} samples"
            )
        lines.append("")

    if run_start:
        for m in run_start.get("seed_messages", []):
            lines.extend(_block(f"## Seed message: {m.get('role', 'unknown')}", m.get("content") or ""))

    max_iteration = max((e["iteration"] for e in events if e["event"] == "llm_response"), default=-1)
    for i in range(max_iteration + 1):
        lines.append(f"## Iteration {i}")
        lines.append("")

        llm_event = next((e for e in events if e["event"] == "llm_response" and e["iteration"] == i), None)
        if llm_event:
            reasoning = llm_event.get("raw_output", {}).get("reasoning")
            if reasoning:
                lines.extend(_block("*Reasoning:*", reasoning, truncate=True))

            content = llm_event["message"].get("content")
            if content:
                lines.extend(_block("**Response:**", content))
            for tc in llm_event["message"].get("tool_calls", []):
                try:
                    args_pretty = json.dumps(json.loads(tc["function"]["arguments"]), indent=2)
                except (json.JSONDecodeError, TypeError):
                    args_pretty = tc["function"]["arguments"]
                lines.extend(_block(f"**Tool call: `{tc['function']['name']}`**", args_pretty, lang="json"))

        nudge = next((e for e in events if e["event"] == "nudge_message" and e["iteration"] == i), None)
        if nudge:
            lines.extend(_block("**Nudge:**", nudge["message"]["content"]))

        for e in events:
            if e["event"] == "tool_call" and e["iteration"] == i:
                lines.extend(_block("**Result:**", e["message"]["content"], truncate=True))

        status = next((e for e in events if e["event"] == "status_message" and e["iteration"] == i), None)
        if status:
            lines.extend(_block("**Status:**", status["message"]["content"]))

    if run_end and run_end.get("status") == "finished":
        lines.extend(_block("## Summary", run_end.get("summary", "")))
    elif run_end:
        lines.extend(
            _block(
                "## Result",
                f"Run ended with status `{run_end.get('status')}` after {run_end.get('iterations')} iteration(s).",
            )
        )

    # Snapshotted before either LLM call's output is spliced in, so neither sees the other's answer first.
    llm_input = "\n".join(lines)
    summary = assessment = None
    # Skipped for Claude runs, where it would be a second billed call per run.
    if base_url and model and provider_for(model) != "anthropic":
        reference_docs = _model_dev_readmes(models)
        summary_input = _truncate_for_summary(llm_input)
        if reference_docs:
            summary_input += (
                "\n\n---\n\nReference documentation for the candidate model(s) (human-only, never "
                f"shown to the agent):\n\n{_truncate_for_summary(reference_docs, limit=50_000)}"
            )
        summary = _llm_section(SUMMARY_PROMPT, summary_input, base_url, model, max_tokens=600)
        assessment = _llm_section(
            VERIFICATION_ASSESSMENT_PROMPT, _truncate_for_summary(llm_input), base_url, model, max_tokens=300
        )
        blocks: list[str] = []
        if summary:
            blocks += _block(f"## Trace Analysis  (Generated by `{model}`)", summary)
        if assessment:
            blocks += _block(f"## Verification Analysis  (Generated by `{model}`)", assessment)
        lines[header_end:header_end] = blocks

    analysis = {
        "task": task_name,
        "status": run_end.get("status") if run_end else None,
        # The finish(model_used) claim beside the trace evidence for it.
        "model_used_reported": self_reported,
        "model_usage": model_usage,
        "model_usage_unsupported": _self_report_unsupported(self_reported, model_usage),
        # Judged by the same backbone the run used: a reading aid, not an independent measurement.
        "assessed_by": model if assessment else None,
        "narrative_summary": summary,
        **parse_verification_assessment(assessment),
        # Measured, not judged: unlike the two fields above these are counted off the trace.
        **checks,
    }
    return "\n".join(lines) + "\n", analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a run's trace.jsonl as a readable markdown transcript")
    parser.add_argument("trace", type=Path, help="Path to a run's trace.jsonl, or its parent run directory")
    parser.add_argument("--out", type=Path, default=None, help="Write to this file instead of stdout")
    parser.add_argument(
        "--base-url", default="http://localhost:8000/v1",
        help="vLLM server for the two LLM sections (summary and assessment)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--no-summary", action="store_true",
        help="skip both LLM sections",
    )
    parser.add_argument(
        "--analysis-out", type=Path, default=None,
        help="also write the parsed analysis as JSON (orchestrate.py writes analysis_generated.json)",
    )
    args = parser.parse_args()

    trace_path = args.trace / "trace.jsonl" if args.trace.is_dir() else args.trace
    base_url, model = (None, None) if args.no_summary else (args.base_url, args.model)
    markdown, analysis = render(trace_path, base_url=base_url, model=model)

    if args.out:
        args.out.write_text(markdown)
        print(f"Wrote {args.out}")
    else:
        print(markdown)
    if args.analysis_out:
        args.analysis_out.write_text(json.dumps(analysis, indent=2))
        print(f"Wrote {args.analysis_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

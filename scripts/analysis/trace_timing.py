"""Per-run timing, oracle-call and specialist-execution facts from trace.jsonl, cached as a pickle
for the --trace-facts options of analysis_cross_task.py and analysis_cross_models.py.

The interval before each event is attributed to it (generation for llm_response, execution for
tool_call).

Usage: uv run python -m scripts.analysis.trace_timing runs/<grid> [...] --out analysis/trace_facts.pkl
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from scripts.analysis.trace_render import _model_usage
from scripts.analysis.utils import parse_run_name

_ORACLE_SCORE = re.compile(r"^\s*[a-zA-Z_0-9]+:\s*(-?[0-9.]+(?:e-?[0-9]+)?)")
# Commands that (re)run a model or a training/inference script, as opposed to editing a file.
_COMPUTE_COMMAND = re.compile(r"\b(python3?|uv run|torchrun|accelerate)\b")


def _content(e: dict) -> str:
    msg = e.get("message") or {}
    return (msg.get("content") if isinstance(msg, dict) else str(msg)) or ""


def _run_facts(run_dir: Path) -> dict:
    row = {"grid": run_dir.parent.name, "run_name": run_dir.name, **parse_run_name(run_dir.name)}
    trace = run_dir / "trace.jsonl"
    if not trace.is_file():
        return row
    events = []
    with trace.open() as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not events:
        return row
    start = next((e for e in events if e.get("event") == "run_start"), {})
    row["sandbox_gpu"] = start.get("gpu")

    llm_s = tool_s = 0.0
    n_llm = completion_tokens = 0
    prev_ts = float(events[0]["ts"])
    oracle_scores, compute_between, compute_since_last = [], [], False
    for e in events:
        ts = float(e["ts"])
        kind = e.get("event")
        if kind == "llm_response":
            llm_s += ts - prev_ts
            n_llm += 1
            usage = e.get("usage") or {}
            completion_tokens += int(usage.get("completion_tokens") or 0)
        elif kind == "tool_call":
            tool_s += ts - prev_ts
            name = e.get("name")
            if name == "oracle_check":
                m = _ORACLE_SCORE.match(_content(e))
                oracle_scores.append(float(m.group(1)) if m else None)
                compute_between.append(compute_since_last)
                compute_since_last = False
            elif name == "run_bash":
                cmd = json.dumps(e.get("arguments") or {})
                if _COMPUTE_COMMAND.search(cmd):
                    compute_since_last = True
        prev_ts = ts
    row |= {
        "trace_llm_s": llm_s,
        "trace_tool_s": tool_s,
        "trace_total_s": float(events[-1]["ts"]) - float(events[0]["ts"]),
        "trace_n_llm": n_llm,
        "trace_completion_tokens": completion_tokens,
        "oracle_calls": len(oracle_scores),
        "oracle_scores": oracle_scores,
        # For call i >= 1: whether any python/uv/torchrun/accelerate command ran between call i-1 and call i.
        "oracle_compute_between": compute_between[1:],
    }
    if row["task"] == "mmlu-astronomy":
        row["astrosage_executed"] = _model_usage(events, ["astrosage"])["astrosage"]["executed_referencing"]
        row["astrosage_script_executed"] = _script_executed(events, "/models/astrosage")
    return row


_SCRIPT_NAME = re.compile(r"([\w./-]+\.py)\b")


def _script_executed(events: list[dict], mount: str) -> bool:
    """Whether a python script whose source references `mount` ran successfully (missed by _model_usage)."""
    scripts = set()
    for e in events:
        if e.get("event") != "tool_call":
            continue
        args = e.get("arguments") or {}
        if e.get("name") == "write_file" and mount in str(args.get("content", "")):
            scripts.add(Path(str(args.get("path", ""))).name)
        elif e.get("name") == "run_bash":
            cmd = str(args.get("command", ""))
            if mount in cmd:
                scripts.update(Path(m).name for m in _SCRIPT_NAME.findall(cmd))
            elif _content(e).startswith("exit_code: 0") and "python" in cmd:
                if any(Path(m).name in scripts for m in _SCRIPT_NAME.findall(cmd)):
                    return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    run_paths = [d for g in args.run_dirs for d in sorted(g.iterdir()) if d.is_dir()]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(_run_facts, run_paths, chunksize=16))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_pickle(args.out)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

"""Per-run LLM latency and retry statistics from trace.jsonl timestamps.

A call's latency is the gap between its llm_response and the previous event (run_start or the
last status_message). latency_clean_* excludes calls whose gap contains retries.

Usage: uv run python -m scripts.analysis.trace_stats runs/ --out trace_stats.csv
"""

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ARCHIVE_DIR_NAME = "archive"

FIELDS = [
    "run_dir",
    "run_start_ts",  # when this run began -- the x-axis for spotting infra drift across a grid's runtime
    "status", "error_type", "error_text",
    "n_llm_calls", "n_llm_retries", "n_transient_retries", "n_nontransient_retries",
    "latency_mean_s", "latency_median_s", "latency_p95_s", "latency_max_s",
    "latency_clean_mean_s", "latency_clean_median_s", "latency_clean_p95_s", "n_clean_calls",
    "n_tool_calls", "n_tool_call_errors",
]


def _read_trace_events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile -- no interpolation, so it never invents a value outside the sample."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def _round_all(stats: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in stats.items()}


def _latency_stats(events: list[dict]) -> dict:
    """Latency over all calls and over calls without retries, plus retry counts by transience."""
    all_latencies: list[float] = []
    clean_latencies: list[float] = []
    n_retries = n_transient = n_nontransient = 0

    prev_ts = None
    pending_retries = 0  # llm_retry events seen since the last non-retry event
    for event in events:
        et = event["event"]
        if et == "llm_retry":
            pending_retries += 1
            n_retries += 1
            if event.get("transient"):
                n_transient += 1
            else:
                n_nontransient += 1
            continue
        if et == "llm_response" and prev_ts is not None:
            gap = event["ts"] - prev_ts
            all_latencies.append(gap)
            if pending_retries == 0:
                clean_latencies.append(gap)
        pending_retries = 0
        prev_ts = event["ts"]

    return {
        "n_llm_calls": len(all_latencies),
        "n_llm_retries": n_retries,
        "n_transient_retries": n_transient,
        "n_nontransient_retries": n_nontransient,
        "latency_mean_s": statistics.fmean(all_latencies) if all_latencies else None,
        "latency_median_s": statistics.median(all_latencies) if all_latencies else None,
        "latency_p95_s": _percentile(all_latencies, 95),
        "latency_max_s": max(all_latencies) if all_latencies else None,
        "latency_clean_mean_s": statistics.fmean(clean_latencies) if clean_latencies else None,
        "latency_clean_median_s": statistics.median(clean_latencies) if clean_latencies else None,
        "latency_clean_p95_s": _percentile(clean_latencies, 95),
        "n_clean_calls": len(clean_latencies),
    }


def _tool_call_ok(event: dict) -> bool:
    """Mirrors trace_render.py's _tool_call_ok: run_bash checks its own exit_code, every other tool an "error: " prefix."""
    content = event.get("message", {}).get("content", "")
    if event.get("name") == "run_bash":
        return content.startswith("exit_code: 0")
    return not content.startswith("error:")


def _tool_call_stats(events: list[dict]) -> dict:
    total = errors = 0
    for e in events:
        if e["event"] != "tool_call":
            continue
        total += 1
        if not _tool_call_ok(e):
            errors += 1
    return {"n_tool_calls": total, "n_tool_call_errors": errors}


def _error_detail(run_end: dict | None) -> dict:
    """Exception type and message from run_end's error field, or None."""
    error = (run_end or {}).get("error")
    if not error:
        return {"error_type": None, "error_text": None}
    error_type, _, message = error.partition(": ")
    return {"error_type": error_type, "error_text": message or error}


def _aggregate_run(run_dir: Path) -> dict:
    events = _read_trace_events(run_dir / "trace.jsonl")
    run_start = next((e for e in events if e["event"] == "run_start"), None)
    run_end = next((e for e in events if e["event"] == "run_end"), None)

    row = {
        "run_dir": str(run_dir),
        "run_start_ts": (run_start or {}).get("ts"),
        "status": (run_end or {}).get("status"),
    }
    row.update(_error_detail(run_end))
    row.update(_round_all(_latency_stats(events)))
    row.update(_tool_call_stats(events))
    return row


def aggregate(runs_dirs: list[Path]) -> list[dict]:
    rows = []
    for runs_dir in runs_dirs:
        for trace_path in sorted(runs_dir.rglob("trace.jsonl")):
            if ARCHIVE_DIR_NAME in trace_path.relative_to(runs_dir).parts:
                continue
            rows.append(_aggregate_run(trace_path.parent))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct per-run LLM-call latency and error/retry detail from trace.jsonl timestamps."
    )
    parser.add_argument("runs_dirs", nargs="+", type=Path, help="Directories to search recursively for trace.jsonl")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path")
    args = parser.parse_args()

    for runs_dir in args.runs_dirs:
        if not runs_dir.is_dir():
            print(f"error: not a directory: {runs_dir}")
            return 1

    rows = aggregate(args.runs_dirs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

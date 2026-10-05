"""Tool-call counts and error rates, specialist-model usage and web_fetch destinations per task,
summed over the given grids. Writes tool_calls.csv, model_usage.csv, web_fetch_summary.csv and
web_fetch_urls.csv.

Usage: uv run python -m scripts.analysis.analysis_tool_usage runs/<grid> [...] --out-dir analysis/tool_usage
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from eval.evaluate import load_task_config
from scripts.analysis.trace_render import (
    _candidate_models,
    _model_reference_urls,
    _model_usage,
    _same_resource,
    _tool_call_stats,
    _web_fetches,
)

def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.") or "(unparseable)"


def _fetch_status_ok(status: str) -> bool:
    return status.startswith("2")

ARCHIVE_DIR_NAME = "archive"


def _read_events(trace_path: Path) -> list[dict]:
    return [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]


def _runs_by_task(grid_dirs: list[Path]) -> dict[str, list[Path]]:
    """Run directories with a trace.jsonl, grouped by the task in their name; archive/ is skipped."""
    runs = defaultdict(list)
    for grid in grid_dirs:
        for trace_path in sorted(grid.rglob("trace.jsonl")):
            if ARCHIVE_DIR_NAME not in trace_path.relative_to(grid).parts:
                runs[trace_path.parent.name.split("__")[0]].append(trace_path.parent)
    return dict(sorted(runs.items()))


def _correct_model(task: str) -> str | None:
    """The task's DOMAIN_MODEL if it beats the backbone (gap_positive), else None."""
    cfg = load_task_config(task)
    gap = (cfg.REFERENCE - cfg.BACKBONE) if cfg.HIGHER_IS_BETTER else (cfg.BACKBONE - cfg.REFERENCE)
    return getattr(cfg, "DOMAIN_MODEL", None) if gap > cfg.MARGIN else None


def _classify_fetch(url: str, reference_urls: dict[str, list[str]], correct_model: str | None) -> str:
    """"correct", "decoy" or "off_list": which candidate model's README links a fetched URL matches."""
    for model, refs in reference_urls.items():
        if any(_same_resource(url, ref) for ref in refs):
            return "correct" if model == correct_model else "decoy"
    return "off_list"


def aggregate(grid_dirs: list[Path]) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Returns (tool_rows, model_usage_rows, web_fetch_summary_rows, web_fetch_url_rows)."""
    tool_rows: list[dict] = []
    model_usage_rows: list[dict] = []
    web_fetch_summary_rows: list[dict] = []
    web_fetch_url_rows: list[dict] = []

    for task, run_dirs in _runs_by_task(grid_dirs).items():
        candidates = _candidate_models(task)
        reference_urls = {m: _model_reference_urls(m) for m in candidates}
        correct_model = _correct_model(task)
        tool_totals: dict[str, list[int]] = {}  # name -> [ok, fail]
        usage_totals = {m: {"mentioned": 0, "read_docs": 0, "executed_referencing": 0, "researched_online": 0} for m in candidates}
        expected_model_hits = {m: 0 for m in candidates}
        # (host, classification) -> [ok, fail]; per-URL detail keyed on the exact string fetched.
        fetch_by_host: dict[tuple[str, str], list[int]] = {}
        fetch_by_url: dict[tuple[str, str], list[int]] = {}
        n_runs = 0

        for run_dir in run_dirs:
            trace_path = run_dir / "trace.jsonl"
            if not trace_path.is_file():
                continue
            events = _read_events(trace_path)
            if not events:
                continue
            n_runs += 1

            for name, ok, fail in _tool_call_stats(events):
                bucket = tool_totals.setdefault(name, [0, 0])
                bucket[0] += ok
                bucket[1] += fail

            usage = _model_usage(events, candidates)
            for model, tiers in usage.items():
                for tier, seen in tiers.items():
                    if seen:
                        usage_totals[model][tier] += 1

            for url, status in _web_fetches(events):
                cls = _classify_fetch(url, reference_urls, correct_model)
                idx = 0 if _fetch_status_ok(status) else 1  # bucket = [ok, fail], matches (ok, fail) unpacking below
                host_bucket = fetch_by_host.setdefault((_host(url), cls), [0, 0])
                host_bucket[idx] += 1
                url_bucket = fetch_by_url.setdefault((url, cls), [0, 0])
                url_bucket[idx] += 1

            score_path = run_dir / "score.json"
            if score_path.is_file():
                score = json.loads(score_path.read_text())
                expected = score.get("expected_model")
                if expected in expected_model_hits:
                    expected_model_hits[expected] += 1

        total_fetches = sum(sum(v) for v in fetch_by_host.values())
        by_class = {"correct": 0, "decoy": 0, "off_list": 0}
        for (_, cls), (ok, fail) in fetch_by_host.items():
            by_class[cls] += ok + fail
        web_fetch_summary_rows.append(
            {
                "task": task,
                "n_runs": n_runs,
                "total_web_fetches": total_fetches,
                "mean_per_run": round(total_fetches / n_runs, 3) if n_runs else 0.0,
                "correct_model_ref_pct": round(100 * by_class["correct"] / total_fetches, 2) if total_fetches else 0.0,
                "decoy_model_ref_pct": round(100 * by_class["decoy"] / total_fetches, 2) if total_fetches else 0.0,
                "off_list_pct": round(100 * by_class["off_list"] / total_fetches, 2) if total_fetches else 0.0,
            }
        )
        for (host, cls), (ok, fail) in sorted(fetch_by_host.items(), key=lambda kv: -sum(kv[1])):
            web_fetch_url_rows.append(
                {"task": task, "scope": "host", "url_or_host": host, "classification": cls, "ok": ok, "fail": fail, "total": ok + fail}
            )
        for (url, cls), (ok, fail) in sorted(fetch_by_url.items(), key=lambda kv: -sum(kv[1])):
            web_fetch_url_rows.append(
                {"task": task, "scope": "url", "url_or_host": url, "classification": cls, "ok": ok, "fail": fail, "total": ok + fail}
            )

        for name, (ok, fail) in sorted(tool_totals.items(), key=lambda kv: -(kv[1][0] + kv[1][1])):
            total = ok + fail
            tool_rows.append(
                {
                    "task": task,
                    "tool": name,
                    "n_runs": n_runs,
                    "total_calls": total,
                    "ok": ok,
                    "fail": fail,
                    "fail_rate_pct": round(100 * fail / total, 2) if total else 0.0,
                    "mean_per_run": round(total / n_runs, 3) if n_runs else 0.0,
                }
            )

        for model in candidates:
            t = usage_totals[model]
            model_usage_rows.append(
                {
                    "task": task,
                    "model": model,
                    "n_runs": n_runs,
                    "n_expected_correct": expected_model_hits[model],
                    "mentioned_pct": round(100 * t["mentioned"] / n_runs, 2) if n_runs else 0.0,
                    "read_docs_pct": round(100 * t["read_docs"] / n_runs, 2) if n_runs else 0.0,
                    "executed_referencing_pct": round(100 * t["executed_referencing"] / n_runs, 2) if n_runs else 0.0,
                    "researched_online_pct": round(100 * t["researched_online"] / n_runs, 2) if n_runs else 0.0,
                }
            )

    return tool_rows, model_usage_rows, web_fetch_summary_rows, web_fetch_url_rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="grid run directories")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/tool_usage"))
    args = parser.parse_args()

    tool_rows, model_usage_rows, web_fetch_summary_rows, web_fetch_url_rows = aggregate(args.run_dirs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "tool_calls.csv", tool_rows)
    _write_csv(args.out_dir / "model_usage.csv", model_usage_rows)
    _write_csv(args.out_dir / "web_fetch_summary.csv", web_fetch_summary_rows)
    _write_csv(args.out_dir / "web_fetch_urls.csv", web_fetch_url_rows)

    print(
        f"Wrote {len(tool_rows)} tool row(s), {len(model_usage_rows)} model-usage row(s), "
        f"{len(web_fetch_summary_rows)} web-fetch summary row(s), {len(web_fetch_url_rows)} web-fetch url/host row(s) to {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

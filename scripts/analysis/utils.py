"""Shared helpers for scripts/analysis: run tables, masks, effect sizes, reliability metrics and output naming."""

import contextlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from axes import AXIS_LEVELS, AXIS_NAMES, estimated_cost_usd

TOOL_NAMES = ("read_file", "write_file", "run_bash", "web_fetch", "finish")
CELL_AXES = ["task", *AXIS_NAMES]  # a "cell" = one config, its replicates are the only thing varying


def parse_run_name(name: str) -> dict:
    """"task__<AXIS_NAMES levels>__r<replicate>" (axes.RunSpec.run_name) -> its parts."""
    task, *levels, rep = name.split("__")
    if len(levels) != len(AXIS_NAMES):
        raise ValueError(f"run name {name!r} doesn't match task__{{{'__'.join(AXIS_NAMES)}}}__rN")
    return {"task": task, **dict(zip(AXIS_NAMES, levels)), "replicate": int(rep.removeprefix("r"))}


def _tool_call_ok(event: dict) -> bool:
    """Whether a tool_call succeeded: exit_code 0 for run_bash, no "error: " prefix otherwise. Keep in sync with trace_render.py and trace_stats.py."""
    content = event.get("message", {}).get("content", "")
    if event.get("name") == "run_bash":
        return content.startswith("exit_code: 0")
    return not content.startswith("error:")


def tool_call_stats(trace_path: Path) -> dict:
    """Per-tool call counts, tool-call errors, distinct tools used and whether finish was called."""
    counts = Counter()
    errors = 0
    if trace_path.is_file():
        for line in trace_path.read_text().splitlines():
            event = json.loads(line)
            if event.get("event") != "tool_call":
                continue
            name = event.get("name")
            counts[name] += 1
            if not _tool_call_ok(event):
                errors += 1
    stats = {f"n_calls_{tool}": counts.get(tool, 0) for tool in TOOL_NAMES}
    stats["n_tool_calls"] = sum(counts.values())
    stats["n_tool_call_errors"] = errors
    stats["n_distinct_tools"] = sum(1 for tool in TOOL_NAMES if counts.get(tool, 0) > 0)
    stats["called_finish"] = counts.get("finish", 0) > 0
    return stats


def _labels_by_run(assignments_path: Path) -> dict[tuple[str, str], str]:
    labels = {}
    for line in assignments_path.read_text().splitlines():
        row = json.loads(line)
        grid, run_name = row["run"].split("/", 1)
        labels[(grid, run_name)] = row["label"]
    return labels


def build_taxonomy_table(run_dirs: list[Path], stage2_dir: Path, dimensions: list[str] | None = None) -> pd.DataFrame:
    """build_run_table plus one column per Stage 2 dimension; runs without an assignment get NaN."""
    if dimensions is None:
        dimensions = sorted(p.name.removesuffix(".categories.json") for p in stage2_dir.glob("*.categories.json"))

    base = build_run_table(run_dirs).set_index(["grid", "run_name"])
    for dim in dimensions:
        labels = _labels_by_run(stage2_dir / f"{dim}.assignments.jsonl")
        base[dim] = pd.Series(labels)
    return base.reset_index()


def build_run_table(run_dirs: list[Path]) -> pd.DataFrame:
    """One row per run across the grid dirs: axis levels, score.json fields and tool-call counts.

    status is "missing_score" when score.json does not exist.
    """
    rows = []
    for run_dir in run_dirs:
        for d in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            row = {"grid": run_dir.name, "run_name": d.name, **parse_run_name(d.name)}
            score_path = d / "score.json"
            if score_path.is_file():
                row.update(json.loads(score_path.read_text()))
            else:
                row["status"] = "missing_score"
            row.update(tool_call_stats(d / "trace.jsonl"))
            rows.append(row)
    df = pd.DataFrame(rows)
    if "expected_score" in df:
        # Metrics cap at 1.0, so larger self-reports are malformed (e.g. 72.5 for 72.5%). Flagged, not changed.
        df["expected_score_valid"] = df["expected_score"].isna() | (df["expected_score"] <= 1.0)
    df["extreme_outlier"] = _extreme_outlier_mask(df)
    _generalize_gap_closed(df)
    _generalize_calibration_error(df)
    _add_cost(df)
    return df


def load_run_table(run_dirs: list[Path], cache: Path | None = None) -> pd.DataFrame:
    """build_run_table, cached as a pickle when `cache` is given. A cache built from other grids is
    rebuilt; derived columns are recomputed on load, so the cache holds only raw fields."""
    if cache is not None and cache.is_file():
        df = pd.read_pickle(cache)
        if set(df["grid"].unique()) == {d.name for d in run_dirs}:
            df["extreme_outlier"] = _extreme_outlier_mask(df)
            _generalize_gap_closed(df)
            _generalize_calibration_error(df)
            _add_cost(df)
            return df
        print(f"cache {cache} covers other grids, rebuilding")
    df = build_run_table(run_dirs)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def restrict_to_common_configs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose non-model axis levels are not shared by every model in df (e.g. step37 lacks act-only). Prints what it drops."""
    models = sorted(df["model"].unique())
    if len(models) < 2:
        print("  only one model present -- nothing to restrict")
        return df
    dropped_any = False
    keep = pd.Series(True, index=df.index)
    for axis in AXIS_NAMES:
        if axis == "model":
            continue
        levels_by_model = {m: set(df.loc[df["model"] == m, axis].unique()) for m in models}
        common = set.intersection(*levels_by_model.values())
        dropped_levels = set.union(*levels_by_model.values()) - common
        for level in sorted(dropped_levels):
            missing_from = [m for m in models if level not in levels_by_model[m]]
            n_rows = int((df[axis] == level).sum())
            print(f"  dropping {axis}={level!r} ({n_rows} row(s)) -- missing for {', '.join(missing_from)}")
            dropped_any = True
        keep &= df[axis].isin(common)
    if not dropped_any:
        print("  every model already covers the same configs -- nothing to restrict")
    return df[keep].reset_index(drop=True)


def _generalize_gap_closed(df: pd.DataFrame) -> None:
    """gap_closed for every regime: recovery from the worse to the better anchor (0 to 1).

    Equals score.json's value in gap_positive. In gap_negative (mmlu-astronomy) BACKBONE is the better
    anchor, so 1 means the agent declined the weaker specialist. NaN in gap_negligible.
    Analysis layer only; score.json is unchanged.
    """
    required = {"reference", "backbone", "score", "higher_is_better", "regime"}
    if not required <= set(df.columns):
        return
    have_inputs = df[["reference", "backbone", "score"]].notna().all(axis=1) & df["higher_is_better"].notna()
    sign = np.where(df["higher_is_better"].fillna(True) == True, 1.0, -1.0)  # noqa: E712
    ref_vs_backbone = sign * (df["reference"] - df["backbone"])  # >0 iff reference is the better anchor here
    good = np.where(ref_vs_backbone >= 0, df["reference"], df["backbone"])
    bad = np.where(ref_vs_backbone >= 0, df["backbone"], df["reference"])
    with np.errstate(invalid="ignore", divide="ignore"):
        gap_closed = sign * (df["score"] - bad) / (sign * (good - bad))
    df["gap_closed"] = np.where(have_inputs & (df["regime"] != "gap_negligible"), gap_closed, np.nan)


def _generalize_calibration_error(df: pd.DataFrame) -> None:
    """calibration_error divided by |reference - backbone|, comparable across tasks. Analysis layer only."""
    required = {"reference", "backbone", "calibration_error", "regime"}
    if not required <= set(df.columns):
        return
    anchor_gap = (df["reference"] - df["backbone"]).abs()
    have_inputs = df[["reference", "backbone", "calibration_error"]].notna().all(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = df["calibration_error"] / anchor_gap
    df["calibration_error_normalized"] = np.where(have_inputs & (df["regime"] != "gap_negligible"), normalized, np.nan)


def _add_cost(df: pd.DataFrame) -> None:
    """cost_usd from token counts via axes.estimated_cost_usd."""
    if not {"model", "prompt_tokens", "completion_tokens"} <= set(df.columns):
        return
    has_tokens = df["prompt_tokens"].notna() & df["completion_tokens"].notna()
    df["cost_usd"] = np.nan
    # NaN instead of None for unpriced models, to keep the column float.
    df.loc[has_tokens, "cost_usd"] = [
        cost if (cost := estimated_cost_usd(m, p, c)) is not None else np.nan
        for m, p, c in zip(df.loc[has_tokens, "model"], df.loc[has_tokens, "prompt_tokens"], df.loc[has_tokens, "completion_tokens"])
    ]


# Valid runs can still be extreme outliers (e.g. a collapsed constant prediction), which dominate
# r2 means. They are reported separately and excluded from summary statistics (see _valid_mask).
EXTREME_OUTLIER_K = 10


def _extreme_outlier_mask(df: pd.DataFrame) -> pd.Series:
    anchor_gap = (df["reference"] - df["trivial"]).abs()
    threshold = np.where(
        df["higher_is_better"] == True,  # noqa: E712
        df["trivial"] - EXTREME_OUTLIER_K * anchor_gap,
        df["trivial"] + EXTREME_OUTLIER_K * anchor_gap,
    )
    worse = np.where(df["higher_is_better"] == True, df["score"] < threshold, df["score"] > threshold)  # noqa: E712
    return (df["valid"] == True) & worse  # noqa: E712


def _valid_mask(df: pd.DataFrame) -> pd.Series:
    """valid runs, minus extreme outliers -- the population every replicate-variance
    and effect-size statistic downstream is computed over (see EXTREME_OUTLIER_K)."""
    return (df["valid"] == True) & ~df["extreme_outlier"]  # noqa: E712


# Score-derived metrics get outlier gating; cost and effort metrics are defined for every completed run.
OUTLIER_SENSITIVE_METRICS = {"score", "gap_closed", "calibration_error", "calibration_error_normalized", "expected_score"}

# Self-reported metrics also need expected_score_valid gating.
_EXPECTED_SCORE_GATED_METRICS = {"expected_score", "calibration_error", "calibration_error_normalized"}


def _metric_mask(df: pd.DataFrame, metric: str) -> pd.Series:
    """Rows usable for one metric: non-null, plus outlier gating for score-derived metrics and expected_score_valid gating for self-reports."""
    mask = df[metric].notna()
    if metric in OUTLIER_SENSITIVE_METRICS:
        mask &= _valid_mask(df)
    if metric in _EXPECTED_SCORE_GATED_METRICS and "expected_score_valid" in df:
        mask &= df["expected_score_valid"].fillna(True)
    return mask


def _noise_floor_metrics(df: pd.DataFrame, min_valid: int, metric: str = "score") -> dict:
    valid = df[_metric_mask(df, metric)]
    if valid["task"].nunique() > 1:
        # Cells never span tasks, so pooled raw values would count between-task mean differences as
        # between-cell signal (pooled ICC above every per-task ICC). Remove each task's mean first.
        valid = valid.assign(**{metric: valid[metric] - valid.groupby("task")[metric].transform("mean")})
    grouped = valid.groupby(CELL_AXES)[metric]
    n_per_cell = grouped.count()
    qualifying = n_per_cell[n_per_cell >= min_valid].index
    metrics = {"cells qualifying (>= min_valid valid reps)": len(qualifying)}
    if len(qualifying) < 2:
        return metrics

    means = grouped.mean().loc[qualifying]
    stds = grouped.std().loc[qualifying]
    ns = n_per_cell.loc[qualifying]
    k = len(qualifying)  # groups
    N = int(ns.sum())  # total replicates across those groups
    grand_mean = (means * ns).sum() / N

    ss_within = ((ns - 1) * stds**2).sum()
    df_within = N - k
    ms_within = ss_within / df_within  # = pooled within-cell variance = the noise floor

    ss_between = (ns * (means - grand_mean) ** 2).sum()
    df_between = k - 1
    ms_between = ss_between / df_between

    # Method-of-moments correction: observed cell means include within-cell sampling noise.
    n0 = (N - (ns**2).sum() / N) / df_between
    between_variance_raw = (ms_between - ms_within) / n0
    between_variance = max(0.0, between_variance_raw)

    metrics |= {
        "total valid replicates used (N)": N,
        "within-cell std (noise floor)": ms_within**0.5,
        "between-cell std (bias-corrected)": between_variance**0.5,
    }
    if between_variance_raw < 0:
        metrics["note"] = "raw correction went negative (truncated to 0) -- no detectable signal above noise"
    icc = between_variance / (between_variance + ms_within) if (between_variance + ms_within) > 0 else float("nan")
    metrics["ICC: share of variance that's real between-cell signal"] = f"{icc:.1%}"
    return metrics


# --- Outcome reliability (hurdle model) ---------------------------------------------------------
# Rare total failures inflate a continuous ICC. Following arXiv:2602.16666 and arXiv:2603.29231,
# success is binarized (beat_trivial) and magnitude is analyzed conditional on success.
# See docs/analysis-metrics.md.
_SUCCESS_INDICATOR = "beat_trivial"


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (Wilson, 1927), safe for small n."""
    if n == 0:
        return float("nan"), float("nan")
    p_hat = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    half_width = (z / denom) * ((p_hat * (1 - p_hat) / n + z2 / (4 * n**2)) ** 0.5)
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _success_series(df: pd.DataFrame, metric: str) -> pd.Series:
    """beat_trivial as float over _metric_mask's population (cast, since the column may be object dtype)."""
    return df.loc[_metric_mask(df, metric), _SUCCESS_INDICATOR].astype(float)


def _outcome_consistency_metrics(df: pd.DataFrame, min_valid: int, metric: str = "score") -> dict:
    """Outcome Consistency C_out (arXiv:2602.16666) with cells as tasks and beat_trivial as the outcome.

        C_out = mean over cells of (2*p_hat - 1)^2

    1 means every cell's replicates all succeed or all fail; an evenly split cell contributes 0.
    Read next to the continuous ICC: rare large misses lower the ICC but not C_out.
    """
    valid = df[_metric_mask(df, metric)]
    counts = valid.groupby(CELL_AXES)[_SUCCESS_INDICATOR].count()
    qualifying = counts[counts >= min_valid].index
    metrics = {"cells qualifying (>= min_valid valid reps)": len(qualifying)}
    if len(qualifying) < 1:
        return metrics

    y_by_cell = valid.assign(**{_SUCCESS_INDICATOR: valid[_SUCCESS_INDICATOR].astype(float)}).groupby(CELL_AXES)[_SUCCESS_INDICATOR]
    p_hat = y_by_cell.mean().loc[qualifying]
    c_out = (2 * p_hat - 1) ** 2
    metrics |= {
        "mean per-cell success rate (beat_trivial)": p_hat.mean(),
        "outcome consistency C_out (arXiv:2602.16666, mean over cells)": c_out.mean(),
    }
    return metrics


def _failure_rate_metrics(df: pd.DataFrame, metric: str = "score") -> dict:
    """Share of replicates that do not beat trivial (pass@1-style, arXiv:2603.29231), with a Wilson interval."""
    y = _success_series(df, metric)
    n = int(y.notna().sum())
    if n == 0:
        return {"n (valid, non-outlier replicates)": 0}
    failures = int((y == 0.0).sum())
    lo, hi = _wilson_interval(failures, n)
    return {
        "n (valid, non-outlier replicates)": n,
        "failures (score doesn't beat trivial by MARGIN)": failures,
        "failure rate": failures / n,
        "failure rate, 95% Wilson interval": f"[{lo:.1%}, {hi:.1%}]",
    }


def _conditional_noise_floor_metrics(df: pd.DataFrame, min_valid: int, metric: str = "score") -> dict:
    """_noise_floor_metrics restricted to replicates that beat trivial (score repeatability given success)."""
    conditional = df[df[_SUCCESS_INDICATOR] == True]  # noqa: E712
    return _noise_floor_metrics(conditional, min_valid, metric)


def _axis_failure_rates(
    df: pd.DataFrame, exclude: dict[str, tuple[str, ...]] | None = None, metric: str = "score",
) -> pd.DataFrame:
    """Per-level failure rate with Wilson interval for every axis; the hurdle-side counterpart to _axis_effect_sizes."""
    exclude = exclude or {}
    rows = []
    for axis in AXIS_NAMES:
        d = df[~df[axis].isin(exclude[axis])] if axis in exclude else df
        valid = d[_metric_mask(d, metric)]
        for level, g in valid.groupby(axis):
            y = g[_SUCCESS_INDICATOR].astype(float)
            n = int(y.notna().sum())
            if n == 0:
                continue
            failures = int((y == 0.0).sum())
            lo, hi = _wilson_interval(failures, n)
            rows.append({
                "axis": axis, "level": level, "n": n, "failures": failures,
                "failure rate": failures / n, "95% Wilson interval": f"[{lo:.1%}, {hi:.1%}]",
            })
    return pd.DataFrame(rows).set_index(["axis", "level"])


def metric_anchor_lines(df: pd.DataFrame, metric: str) -> list[tuple[float, str, str]]:
    """Reference lines (value, linestyle, label) for a metric: fixed 0/1 for gap_closed, else per-task anchors."""
    if metric == "gap_closed":
        return [(0.0, "--", "Worse anchor"), (1.0, "-.", "Better anchor")]
    return [
        (df[col].dropna().iloc[0], style, col.capitalize())
        for col, style in (("reference", "--"), ("backbone", "-."), ("trivial", ":"))
        if col in df and df[col].notna().any()
    ]


def metric_display_name(df: pd.DataFrame, metric: str) -> str:
    """Display name for a metric; "score" shows the task's own metric name."""
    if metric == "score":
        return df["metric"].dropna().iloc[0] if "metric" in df and df["metric"].notna().any() else "score"
    return metric


# protocol hands over a reference solution, so it is excluded from effect sizes.
EFFECT_SIZE_EXCLUDE = {"information": ("protocol",)}


# Marginal means condition on valid runs, so levels with low completion are biased samples.
# Flooring failures at trivial was rejected, since trivial's severity differs across tasks.
def _axis_effect_sizes(
    df: pd.DataFrame, min_valid: int, exclude: dict[str, tuple[str, ...]] | None = None, metric: str = "score",
) -> pd.DataFrame:
    exclude = exclude or {}
    rows = []
    for axis in AXIS_NAMES:
        # Per axis, so each noise floor uses the same filtered population.
        d = df[~df[axis].isin(exclude[axis])] if axis in exclude else df
        within_std = _noise_floor_metrics(d, min_valid, metric).get("within-cell std (noise floor)")
        valid = d[_metric_mask(d, metric)]
        level_means = valid.groupby(axis)[metric].mean()
        level_ns = valid.groupby(axis)[metric].count()
        rng = level_means.max() - level_means.min()
        rows.append({
            "axis": axis,
            "levels": len(level_means),
            "min valid n per level": int(level_ns.min()),
            "marginal range": rng,
            "effect size (range / within-cell std)": rng / within_std if within_std else float("nan"),
        })
    table = pd.DataFrame(rows).set_index("axis")
    table["rank"] = table["effect size (range / within-cell std)"].rank(ascending=False).astype(int)
    return table


def _section(title: str) -> None:
    print(f"\n{'-' * 80}\n{title}\n{'-' * 80}")


def _print_metrics(rows: dict[str, object]) -> None:
    """Print label -> value as an aligned two-column table."""
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.0f}" if x.is_integer() else f"{x:.4f}"
        return str(x)
    series = pd.Series({k: fmt(v) for k, v in rows.items()}, name="value")
    print(series.to_string())


def _print_metrics_by_task(df: pd.DataFrame, compute: Callable[[pd.DataFrame], dict]) -> None:
    """Print compute()'s metrics with one column per task, since metrics differ in units across tasks."""
    tasks = sorted(df["task"].unique())
    if len(tasks) == 1:
        _print_metrics(compute(df))
        return
    per_task = {task: compute(df[df["task"] == task]) for task in tasks}
    table = pd.DataFrame(per_task)

    def fmt(x):
        if isinstance(x, float):
            return f"{x:.0f}" if x.is_integer() else f"{x:.4f}"
        return "" if pd.isna(x) else str(x)
    print(table.map(fmt).to_string())


class DropWrotePrefix:
    """Stdout wrapper that drops lines starting with "wrote " (used when chaining plot scripts)."""

    def __init__(self, stream) -> None:
        self._stream = stream
        self._buf = ""

    def write(self, s: str) -> None:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not line.startswith("wrote "):
                self._stream.write(line + "\n")

    def flush(self) -> None:
        if self._buf and not self._buf.startswith("wrote "):
            self._stream.write(self._buf)
        self._buf = ""
        self._stream.flush()


class _Tee:
    """Stream that duplicates every write to a second stream."""

    def __init__(self, stream, file) -> None:
        self._stream = stream
        self._file = file

    def write(self, s: str) -> None:
        self._stream.write(s)
        self._file.write(s)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()


@contextlib.contextmanager
def tee_stdout_to_file(file_path: Path):
    """Also write everything printed inside the block to file_path; restores stdout on exit."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    with file_path.open("w") as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            yield
        finally:
            sys.stdout = original_stdout


# Model level -> tier code for grid_label.
_MODEL_TIER_LETTER = dict(zip(AXIS_LEVELS["model"][:3], "sml"))
_MODEL_TIER_LETTER["step37-198b-a11b-fp8"] = "f"

# Size word in grid directory names -> tier code.
_SIZE_WORD_ORDER = ("small", "medium", "large")
_SIZE_WORD_LETTER = dict(zip(_SIZE_WORD_ORDER, "sml"))


def _tier_codes_to_label(codes: list[str]) -> str:
    """Join tier codes: concatenated if all are single letters, else "+"-separated."""
    return "".join(codes) if all(len(c) == 1 for c in codes) else "+".join(codes)


def _join_grid_segments(task_to_letters: dict[str, str]) -> str:
    """"<task first word>-<codes>" per task, sorted and joined with "_"."""
    return "_".join(f"{task.split('-')[0]}-{letters}" for task, letters in sorted(task_to_letters.items()))


def grid_label(df: pd.DataFrame) -> str:
    """Short label for the grids in df, e.g. "redshift-sml" or "redshift-sm_rna-sm".

    Used for analysis/ output directories, so different grid sets never overwrite each other.
    """
    known = list(AXIS_LEVELS["model"])
    task_to_letters = {}
    for task in df["task"].unique():
        tiers_present = df.loc[df["task"] == task, "model"].unique()
        # Qwen3.5 tiers first in size order, then other families alphabetically.
        tiers_present = sorted(tiers_present, key=lambda m: (known.index(m) if m in known else len(known), m))
        codes = [_MODEL_TIER_LETTER.get(m, m) for m in tiers_present]
        task_to_letters[task] = _tier_codes_to_label(codes)
    return _join_grid_segments(task_to_letters)


def grid_label_from_dirs(dirs: list[Path | str]) -> str:
    """grid_label from grid directory names, for stage2_cluster.py."""
    task_to_sizes: dict[str, set[str]] = {}
    for d in dirs:
        tokens = re.sub(r"-\d{8}-\d{6}$", "", Path(d).name).split("-")
        task, size = tokens[0], tokens[tokens.index("full") + 1]
        task_to_sizes.setdefault(task, set()).add(size)
    task_to_letters = {}
    for task, sizes in task_to_sizes.items():
        sizes_sorted = sorted(sizes, key=lambda s: (_SIZE_WORD_ORDER.index(s) if s in _SIZE_WORD_ORDER else len(_SIZE_WORD_ORDER), s))
        codes = [_SIZE_WORD_LETTER.get(s, s) for s in sizes_sorted]
        task_to_letters[task] = _tier_codes_to_label(codes)
    return _join_grid_segments(task_to_letters)


ICC_KEY = "ICC: share of variance that's real between-cell signal"


def icc_value(metrics: dict) -> float:
    """The ICC from a _noise_floor_metrics dict as a fraction, NaN if absent."""
    value = metrics.get(ICC_KEY)
    return float(value.rstrip("%")) / 100 if isinstance(value, str) else float("nan")


def gap_positive_tasks(df: pd.DataFrame) -> list[str]:
    """Tasks whose regime is gap_positive."""
    return sorted(df.loc[df["regime"] == "gap_positive", "task"].unique())


def cluster_resample(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Resample runs with replacement within each cell (completed or not), for cell-cluster bootstraps."""
    codes = df.groupby(CELL_AXES, sort=False).ngroup().to_numpy()
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, sorted_codes, side="left")
    ends = np.searchsorted(sorted_codes, sorted_codes, side="right")
    picks = starts + np.floor(rng.random(len(df)) * (ends - starts)).astype(int)
    return df.iloc[order[picks]].reset_index(drop=True)

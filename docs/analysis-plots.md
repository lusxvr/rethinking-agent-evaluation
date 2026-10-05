# Analysis plots reference

How to read each figure from `scripts/analysis/plot_*.py`. Definitions are in
`docs/analysis-metrics.md`. All scripts read `build_run_table()`. Raw scores from different tasks are
never pooled; pooled cross-task figures use `gap_closed`. The `harness` axis is labeled
"reasoning" in figures.

Common elements:
- REFERENCE, BACKBONE and TRIVIAL appear as dashed reference lines where the metric is a raw score;
  `gap_closed` has fixed anchors at 0 and 1.
- Panels use usable runs (`docs/analysis-metrics.md` §1): score-based metrics exclude invalid runs
  and extreme outliers; cost and effort metrics use every run that has a value.
- `--merge` pools a task's grid directories (e.g. one per model tier), so `model` varies. In
  `plot_univariate.py` and `plot_variance.py` it also draws one row per task.
- Effect sizes are printed in the text reports, not on the figures.

## Outcome and score per axis level: `plot_univariate.py`

One figure per axis. Top: outcome shares per level (`finished, valid`, `finished, invalid`,
`max_duration_reached`, `missing_score`, `extreme_outlier`). Bottom: score box plot and points per
level, valid runs only, with n in the tick labels. Read the top panel first: a level with few valid
runs is a smaller, likely easier subset.

Example (redshift-estimation, information): `protocol` is 100% valid and close to REFERENCE;
`none` and `identity` are about 55% valid and scatter below TRIVIAL.

## Replicate variance: `plot_variance.py`

The CLI writes four panels per task; the paper uses two figures of two panels each
(`plot_variance_main`, `plot_variance_appendix`, written by `analysis_cross_task.py` as
`variance/<tasks>_overview.pdf` and `variance/<tasks>_completion-ci.pdf`).

1. **Completion**: cells by number of completed replicates (0-5), stacked by how many clear the
   hurdle (green all, orange some, red none).
2. **Score variance**: each qualifying cell's mean (x) against its std (y), colored by its hurdle
   success rate and shaped by task when pooled, with an OLS line and Spearman correlation. High-std
   cells tend to be orange: mixed success inflates continuous variance.
3. **CI stability**: mean 95% CI half-width at 3, 4 and 5 replicates, for all runs and for runs that
   beat trivial.
4. **Noise floor**: between-cell (signal) vs. within-cell (noise) variance share, for all runs and
   for runs that beat trivial; the share is the ICC.

## Two-axis interactions: `plot_interactions.py`

A bubble matrix pairing each top-ranked axis (`-k`, default 2) with every other non-model axis: color is the mean metric at each
combination of levels (other axes free), size is the number of runs. All pairs drawn in one call
share color and size scales, so panels can be compared. If the color pattern only shifts between
rows, the axes act independently; if it changes shape, they interact.

`plot_budget_bars` (the paper's budget figures) draws mean `gap_closed` with a 95% CI per budget,
grouped by Model or Information level; `analysis_cross_task.py` and `analysis_cross_models.py` write
it as `interactions/<axis>-budget_interaction.pdf`.

## Metric vs. metric: `plot_correlation.py`

Left: one bar per axis, how much conditioning on it changes the correlation (blue: the axis
inflates it, a likely confound; red: it suppresses it). Right: the scatter, shaped by the
top-ranked axis (or `--axis`) and colored by score (5th-95th percentile, gray if unusable), with
log axes at decade ticks for heavy-tailed metrics. `expected_score` vs. `score` also gets the
perfect-calibration line and an OLS fit.

Example (redshift-estimation, `wallclock_s` vs. `cost_usd`): Spearman 0.888. Conditioning on
information lowers it to 0.71 (runs given the protocol finish fast and cheaply); model explains
almost nothing (gap 0.001).

Without `--metric-x/--metric-y`, all pairs of `DEFAULT_METRICS` (`wallclock_s`, `cost_usd`,
`n_tool_calls`, `n_tool_call_errors`) plus the calibration pair are drawn.

## Axis vs. any metric: `plot_axis_effect.py`

Box plot of a metric per level of one axis; each box is filled with its mean score (or
`gap_closed` when pooled). `--show-points` overlays runs shaped by a second axis (the one with the
largest effect, or `--shape-axis`). Without `--axis/--metric`, every axis is drawn against
`DEFAULT_AXIS_EFFECT_METRICS` (the defaults plus `calibration_error`).

## Cross-task figures: `plot_cross_task.py`

Per-axis level lines per task plus the pooled line, a pooled bar chart over gap_positive tasks, a
bump chart of each axis's effect-size rank per task, a heatmap of the signed change from each
axis's first to last level, a single-axis drilldown and per-task `gap_closed` distributions.

## Taxonomy: `plot_categories.py`

Per dimension: category counts (colored by valence when available) and category shares per axis
level (`--hue-axis` splits each level, e.g. by model family or oracle). `--summary` adds the
Cramer's V table and one stacked bar per dimension.

## Which script for which question

| Question | Script |
|---|---|
| How does one axis affect completion and score? | `plot_univariate.py` |
| How reproducible is a configuration? | `plot_variance.py` |
| Does one axis's effect depend on another? | `plot_interactions.py` |
| Are two outcomes related, and which axis drives it? | `plot_correlation.py` |
| How does an axis affect time, cost or tool use? | `plot_axis_effect.py` |
| Which axes matter across tasks? | `plot_cross_task.py` |
| Which behaviors occur, and under which conditions? | `plot_categories.py` |

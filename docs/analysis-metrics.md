# Analysis metrics reference

Definitions of the metrics computed by `scripts/analysis/` (`analysis_single_task.py`,
`analysis_cross_task.py`, `analysis_cross_models.py`, shared code in `utils.py`). Section numbers
are referenced from code comments.

## 1. The run table

`build_run_table()` builds one row per run with:
- **Axis levels**, parsed from the run directory name.
- **`score.json` fields**: `status`, `valid`, `score`, the anchors, `gap_closed`,
  `calibration_error`, `wallclock_s`, token counts and so on. `status` is `"missing_score"` when
  the file does not exist.
- **Tool-call counts** from `trace.jsonl`: per tool, `n_tool_calls`, `n_tool_call_errors`
  (`run_bash` with a non-zero exit code, other tools with an `"error: "` result),
  `n_distinct_tools`, and whether `finish` was called.
- **`cost_usd`**: tokens priced with `axes.estimated_cost_usd` at first-party list prices.
- **Generalized columns**: `gap_closed` for every regime (§3) and `calibration_error_normalized`
  (calibration error divided by |REFERENCE - BACKBONE|), so tasks can be pooled.

A **cell** is one task plus one combination of the five axis levels, with up to 5 replicates.

A run is **usable** for a metric when the value is present. For score-derived metrics
(`OUTLIER_SENSITIVE_METRICS`) it must also be valid and not an extreme outlier (§4); for
self-reported metrics (`expected_score`, `calibration_error`) `expected_score` must be at most 1
(`_metric_mask`).

## 2. Task metrics

| Task | Metric | Range | TRIVIAL |
|---|---|---|---|
| mmlu-astronomy (n=152) | accuracy | [0, 1] | 0.25 (chance) |
| promoter-prediction (n=613) | MCC | [-1, 1] | 0 (constant predictor) |
| rna-folding (n=300) | relaxed base-pair F1 | [0, 1] | 0 |
| redshift-estimation (n=20) | R² | (-inf, 1] | 0 (mean predictor) |

R² has no lower bound, so one collapsed run can dominate a mean or variance. This is why extreme
outliers are excluded (§4), why medians and IQRs are read first, and why raw scores are never
pooled across tasks.

## 3. Per-run scoring (`eval/evaluate.py`)

Each task declares REFERENCE (specialist operated correctly), BACKBONE (backbone alone), TRIVIAL
(null predictor) and MARGIN (noise threshold) in `solutions/<task>/eval_config.py`.

- `signed_gap(a, b)` is `a - b` for higher-is-better metrics, else `b - a`; positive means a is
  better.
- `regime`: `gap_positive` if signed_gap(REFERENCE, BACKBONE) > MARGIN, `gap_negative` if it is
  below -MARGIN, otherwise `gap_negligible`.
- `gap_closed` = signed_gap(score, BACKBONE) / signed_gap(REFERENCE, BACKBONE), defined in
  `gap_positive` only. The analysis layer generalizes it to `gap_negative` by measuring recovery
  from the worse to the better anchor, so 1 means matching BACKBONE when the specialist is worse.
  Values can fall below 0 or above 1.
- `below_both`: worse than both anchors by more than MARGIN.
- `beat_trivial`: better than TRIVIAL by more than MARGIN.
- `model_choice_correct`: the domain model in `gap_positive`, no model in `gap_negative`, unscored
  in `gap_negligible`.
- `calibration_error` = signed_gap(expected_score, score): positive means overconfident. Only at
  `reported` and `binding`.

## 4. Extreme outliers

A valid run can still be a total failure (e.g. a near-constant prediction). Such runs are flagged
and excluded from every score statistic from §6 on:

```
threshold = TRIVIAL - K * |REFERENCE - TRIVIAL|   (higher-is-better; mirrored otherwise)
flagged   = valid and score worse than threshold,   K = EXTREME_OUTLIER_K = 10
```

K = 10 is a chosen constant, not derived from data. Flagged runs still count in completion statistics.

## 5. Completion

Status counts and status x valid (to catch finished runs without a scorable output).

## 6. Descriptive summaries

`describe()` for every numeric column, with out-of-range `expected_score` values and extreme
outliers blanked for the affected columns. Integer columns are rounded. For low-cardinality text
columns, the five most common values.

## 7. Completion variance

Cells are always valid, always invalid or mixed across their replicates, plus the mean and std of
the per-cell valid share.

## 8. Score variance

For cells with at least `min_valid` (default 3) valid replicates: mean of cell means and the
distribution of per-cell std, also as a percentage of |REFERENCE - TRIVIAL|. These cells skew
toward easier configurations.

## 9. Noise floor (one-way random-effects ANOVA)

Splits score variance over qualifying cells into within-cell noise and between-cell signal:

```
MS_within  = sum_i (n_i - 1) * s_i^2 / (N - k)
MS_between = sum_i n_i * (mean_i - grand_mean)^2 / (k - 1)
n_0        = (N - sum_i n_i^2 / N) / (k - 1)
between    = max(0, (MS_between - MS_within) / n_0)
ICC        = between / (between + MS_within)
```

The `n_0` correction removes the sampling noise in cell means (unbalanced design). A low ICC means
most cell-to-cell variation is noise. When several tasks are pooled, each task's mean is
subtracted first; otherwise differences between tasks would count as between-cell signal.

ICC describes the grid as a whole, not a test between two cells. Because ANOVA uses squared
deviations, rare severe misses can dominate MS_within (§10).

## 10. Outcome reliability (hurdle decomposition)

A few failed replicates inside otherwise tight cells can lower the ICC sharply. The analysis
therefore separates whether a replicate produces a real attempt (`beat_trivial`) from how
repeatable its score is given success (a two-part hurdle model, Cragg 1971), following
arXiv:2602.16666 and arXiv:2603.29231.

- **Outcome Consistency** (arXiv:2602.16666), with p_i a cell's `beat_trivial` rate:
  ```
  C_out = mean over cells of (2 * p_i - 1)^2  =  1 - p_i * (1 - p_i) / 0.25
  ```
  1 means every cell's replicates all succeed or all fail; an evenly split cell contributes 0.
- **Failure rate**: 1 - mean(`beat_trivial`) with a 95% Wilson interval (normal intervals
  undercover at n ~ 5 and rates near 0 or 1), pooled and per axis level.
- **Conditional noise floor**: §9 restricted to replicates that beat trivial. Compared with the
  unconditional ICC: a large rise means rare failures dominated it; little change means the noise
  was ordinary; a drop (rna-folding: 45.7% -> 35.2%) means part of the axis effect is which
  configurations clear the hurdle, so read it next to the failure rates.

`beat_trivial` is a hard MARGIN-based threshold; there is too little data per cell to fit a
mixture model. It is separate from the coarser extreme-outlier screen (§4).

## 11. Confidence-interval stability

Mean 95% CI half-width of a cell mean at n = 3, 4, 5, over cells with 5 valid replicates:

```
half_width(n) = t(0.975, n-1) * s / sqrt(n)     t = 4.303, 3.182, 2.776 for n = 3, 4, 5
```

For n < 5 the standard error is averaged over all n-sized subsets of each cell's 5 scores. Unlike
the raw std, the half-width shrinks steadily with n, so it answers how many replicates are needed.

## 12. Axis effect sizes

For each axis, the marginal mean per level pools all valid runs at that level, and:

```
effect_size = (max level mean - min level mean) / within-cell std
```

This is a signal-to-noise ratio, not a significance test. Caveats:
1. Range grows with the number of levels. A second ranking excludes `information=protocol`
   (`EFFECT_SIZE_EXCLUDE`), which hands over the reference solution.
2. Means pool only valid runs, so levels with different completion rates are biased samples.
   Scoring failures at TRIVIAL was rejected, since TRIVIAL's severity differs across tasks.

Effect sizes and the noise floor accept any numeric column (`--axis-effect`), e.g.
`wallclock_s` or `cost_usd`. Score-derived metrics (`OUTLIER_SENSITIVE_METRICS`) use valid,
non-outlier runs; cost and effort metrics use every run that has a value.

`analysis_cross_task.py` pools tasks on `gap_closed` with equal weight per task and reports each
axis's rank per task, so consistency across tasks is visible.

## 13. Correlation between two metrics

Spearman rank correlation by default (`--correlation-method pearson` for linear), over rows where
both metrics are usable. To see whether an axis explains a correlation, both metrics are demeaned
within each level of that axis and the correlation is recomputed. A much weaker within-level
correlation means the axis drives the raw association (Simpson's paradox). One axis at a time;
demeaning before ranking is an approximation for Spearman.

For calibration (`expected_score` vs. `score`) the plot adds the y = x line and an OLS fit.

## 14. Robustness checks (`analysis_cross_task.py --robustness`)

- **Effect uncertainty**, per gap_positive task: the share of 1,000 cell-cluster bootstrap draws
  (runs resampled within each cell, completed or not) in which each axis has the largest effect
  size; partial η² from a main-effects OLS on `gap_closed` of usable runs, with a 200-draw
  bootstrap interval; and type-II F-tests of all two-way interactions.
- **Budget robustness**: long minus short budget in mean and median `gap_closed` per Model and
  Information level, pooled over gap_positive tasks, for all runs and for matched configurations
  (all axes but budget with at least one usable run at every budget), with bootstrap intervals.
- **Margin sensitivity**: `beat_trivial` recomputed at 0, 0.5, 1 and 2 times MARGIN; hurdle failure
  rate and hurdle-clearing ICC per task and pooled.
- **gap_negative tasks** on raw score: completion and mean score per level, effect sizes, ICC and,
  with trace facts, how often the domain model was used.

Oracle hill-climbing (`analysis_cross_models.py --group-by oracle --trace-facts`): a run is flagged
if it calls `oracle_check` at least 5 times and at least half of its re-checks (counted over calls
that returned a score) follow only file edits, with no python/uv/torchrun command in between.

## 15. Limitations

- R² stays skewed after outlier exclusion; read medians and IQRs first.
- Variance and effect-size metrics condition on valid runs, which favors easier configurations.
- `EXTREME_OUTLIER_K = 10` and `beat_trivial`'s MARGIN are chosen thresholds.
- ICC describes the whole grid, not pairwise differences.
- CI estimates for 3 and 4 replicates are resampled from 5-replicate cells.
- Effect sizes are confounded by level count and completion rates.
- Correlation breakdowns are single-axis.
- Cost and effort metrics are not gated on validity, unlike score metrics.

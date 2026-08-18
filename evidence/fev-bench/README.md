# fev-bench

The 100-task fev-bench: our per-task result and the six published comparators, with the
seasonal-naive reference they are normalized to.

| file | what it is |
|---|---|
| `tinycast.json` | ours: SQL, MASE, WQL and WAPE on each of the 100 tasks, host profile, with the covariate and multivariate flags the benchmark assigns |
| `detector-firing.json` | ours: per-task detector firing rate beside SQL, MASE and WQL |
| `comparators/seasonal_naive.csv` | **the denominator.** Every ratio on this benchmark divides by it |
| `comparators/citras-fm.csv`, `comparators/flowstate.csv`, `comparators/toto-2.0-4m.csv` | the three released models |
| `comparators/autoarima.csv`, `comparators/autoets.csv`, `comparators/autotheta.csv` | the statistical baselines |

## The comparators are not ours

Each is that publisher's own per-task result, vendored byte for byte from the
benchmark's `benchmarks/fev_bench/results/`, and the export refuses to emit if one of
them differs from its source by a byte. `model_name` inside each file is the publisher's
own identifier, left as they wrote it.

They were published under three harness versions, recorded in each file's own
`fev_version` column: 0.6.1 for `citras-fm.csv`, 0.8.0 for `flowstate.csv`, the three
statistical baselines and the seasonal-naive reference, and 0.9.0 for
`toto-2.0-4m.csv`. A small difference on this benchmark can carry harness drift as well
as model difference, and the paper states that where it reports the comparison.

Coverage differs as well. `autoarima.csv` reports 96 of the 100 tasks and `autoets.csv`
97, and they miss different ones; an unreported task is dropped from that model's
geometric mean rather than imputed at the seasonal-naive score, so those two rows are
means over 96 and 97 ratios where every other row is over 100.

`flowstate.csv` carries FlowState's own `trained_on_this_dataset` column, which reads
true on 8 of the 100 tasks.

One row does not survive the construction below and the paper does not report it.
`autoets.csv` aggregates to 3.2652 relative WQL, which is a handful of tasks whose
seasonal-naive WQL denominator is near zero rather than a statement about AutoETS; its
relative MASE, 0.9767, is unaffected. The paper reports AutoARIMA and AutoTheta on this
benchmark and not AutoETS, for that reason.

## Aggregation

Per task, divide the `MASE` and `WQL` columns by the same column of
`comparators/seasonal_naive.csv`, then take the geometric mean over the tasks both
cover. That construction gives the relative MASE and relative WQL the paper reports on
this benchmark: ours are 0.8193 and 0.6581.

Two other constructions live in these files and are not that one. The `all`,
`univariate`, `no_covariates` and `multivariate` blocks of `tinycast.json` are
geometric means of the raw metric, unnormalized, which is the benchmark's own
convention; they divide by nothing and are not comparable with the ratios above. The
benchmark's headline skill score is one minus the geometric mean of the per-task ratio
to seasonal naive, so on the same task set and the same metric it is exactly one minus
the relative aggregate above.

The ratio arithmetic is the one the GIFT-Eval directories and `../chronos-zs/` use, over
a different task set and against a different seasonal-naive reference, so the levels are
not comparable across the three.

## `detector-firing.json`

The same evaluation re-run with the detector's periods returned unchanged, which is what
lets a firing rate be recorded per task. Its `aggregate` block agrees with
`tinycast.json` to four decimals, 1.0875 against 1.0876 on SQL and 1.2910 against 1.2913
on MASE; the residue is bf16 nondeterminism between two runs of one checkpoint.
`per_task` carries each task's firing rate beside its raw SQL, MASE and WQL and the
seasonality the benchmark declares for it; `bands_all_tasks` groups the tasks by firing
rate, `controlled_cyclic_only` repeats that comparison on the 71 tasks whose declared
seasonality is more than one step and carries its 95% interval, and
`spearman_firing_vs_sql` is the rank correlation between firing rate and score.

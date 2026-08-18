# Chronos-ZS

The 27-task Chronos zero-shot benchmark: our per-task result, the published field it is
placed against, and the seasonal-naive reference that field is normalized to. Every file
here covers 27 of 27 tasks.

| file | what it is |
|---|---|
| `tinycast.json` | ours: MASE and WQL on each of the 27 tasks, host profile |
| `detector-firing.json` | ours: per-task detector firing rate beside the normalized MASE |
| `comparators/seasonal_naive.csv` | **the denominator.** Every ratio on this benchmark divides by it |
| `comparators/*.csv` | the other 20 published per-task results |

## The comparators are not ours

Each is that publisher's own per-task result, vendored byte for byte from
`autogluon/fev` at `benchmarks/chronos_zeroshot/results/`, all of them under harness
version 0.6.0. We do not run another group's model to produce a number reported against
ourselves, and the export refuses to emit if one of these files differs from its source
by a byte. `model_name` inside each file is the publisher's own identifier, left as they
wrote it, which is why these filenames keep their spelling where ours does not.

## Aggregation

Per task, divide the `MASE` and `WQL` columns by the same column of
`comparators/seasonal_naive.csv`, then take the geometric mean over the 27 tasks. That
construction gives the relative MASE and relative WQL the paper reports on this
benchmark: ours are 0.8800 and 0.7218. Applied to a comparator file it gives that
comparator's row, and applied to `seasonal_naive.csv` it gives 1.0 by construction.

`tinycast.json` also carries `ngmase` and `gwql`. Those are geometric means of our own
raw MASE and WQL over the 27 tasks, divided by nothing, and they are not the figures the
paper reports.

The arithmetic is the one the GIFT-Eval directories use, over a different task set and
against a different seasonal-naive reference, so the levels are not comparable with
them.

## `detector-firing.json`

The same evaluation re-run with the detector's periods returned unchanged, which is what
lets a firing rate be recorded per task. Its `aggregate_raw` agrees with `tinycast.json`
to seven decimals; the residue is bf16 nondeterminism between two runs of one
checkpoint. `per_task` carries each task's firing rate, its MASE already normalized to
the seasonal-naive reference, and the seasonality the benchmark declares for it;
`bands_all_tasks` groups the tasks by firing rate, and `controlled_cyclic_only` repeats
that comparison on the 17 tasks whose declared seasonality is more than one step,
with its 95% interval.

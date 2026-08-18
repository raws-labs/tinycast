# Replacing the detector's output at inference time

Inference-time interventions on the deployed checkpoint, not retrains. Each replaces
the periods the detector supplies and changes nothing else, so the phase machinery
stays in place and only its input moves. That makes these arms answer a different
question from `../detector-retrain/`, and they are not comparable with it.

**Control: `control.csv`**, the deployed checkpoint with its own detected periods,
unmodified. It is byte-identical to `../deployed/host.csv`.

| file | what replaces the detected periods |
|---|---|
| `control.csv` | nothing: the detector's own output |
| `zeroed.csv` | the output suppressed entirely. The matched control for the two substitutes below |
| `fixed.csv` | a fixed, data-blind period set |
| `shuffled.csv` | the detector's own pooled output, applied to the wrong series |
| `canonical.csv` | the metadata-declared canonical period |

Read `fixed.csv` and `shuffled.csv` against `zeroed.csv`, not against `control.csv`:
suppression is what isolates the value of the correspondence between a period and the
series it was measured from, which is the question these arms were built to answer.

Inference profile, constant across this directory: sign symmetrization on,
canonical-period alignment on, bf16 compute. It is stated here once rather than
carried in every filename.

## Aggregation

Per configuration, divide the arm's metric column by the same column of
`../deployed/seasonal-naive.csv`, then take the geometric mean over the configurations both
cover. That single construction produces every normalized aggregate in the paper:
applied to `eval_metrics/MASE[0.5]` it gives nGMASE, and applied to
`eval_metrics/mean_weighted_sum_quantile_loss` it gives nWQL, or nMAD where the arm
emits one quantile and that column reduces to a point error.
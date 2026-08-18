# Component family

Which recipe and inference choices are worth their cost. A 394,441-parameter
nine-quantile line, the architecture family's best arm rebuilt with a nine-quantile
head and its recency gate retained, trained for 30,000 steps at about a fifth of the
deployed sample budget.

**Control: `control.csv`.**

| file | what it changes against the control |
|---|---|
| `control.csv` | the 394,441-parameter nine-quantile line |
| `future-conv.csv` | the horizon-evolving future-conv correction (adds 51K parameters) |
| `synthetic-family-blend.csv` | the synthetic-family blend |
| `synthetic-family-blend-tuned-dose.csv` | the synthetic-family blend at a tuned dose |
| `gated-committing-loss.csv` | the gated committing loss |
| `future-conv-synthetic-families.csv` | future-conv and the synthetic families together |
| `ar-chunks-8.csv` | eight training AR chunks instead of four |
| `backtest-period.csv` | a backtest-selected period instead of the detector's raw pick |
| `mase-weighted.csv` | a MASE-weighted loss |
| `kalman.csv` | a Kalman-smoothed decoder head |

`kalman.csv` is excluded from the paper's component table and from its multiplicity
correction: it trained at four times the sample budget of the rest of the family, so
its delta is confounded with the budget difference and cannot be read against the
shared control.

Arms carry the control's parameter count except `future-conv.csv`, which adds 51K.

Inference profile, constant across this directory: sign symmetrization on,
canonical-period alignment on, bf16 compute. It is stated here once rather than
carried in every filename.

## Aggregation

Per configuration, divide the arm's metric column by the same column of
`../../deployed/seasonal-naive.csv`, then take the geometric mean over the configurations both
cover. That single construction produces every normalized aggregate in the paper:
applied to `eval_metrics/MASE[0.5]` it gives nGMASE, and applied to
`eval_metrics/mean_weighted_sum_quantile_loss` it gives nWQL, or nMAD where the arm
emits one quantile and that column reduces to a point error.
Each family has its own control and its own training line, so deltas within a family
are meaningful and absolute scores are not comparable across families.

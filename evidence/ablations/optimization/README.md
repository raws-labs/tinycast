# Optimization family

Whether the optimization settings sit at their optimum. Learning-rate peak and sample
budget, varied one at a time on a single-quantile line at the deployed recipe's
effective batch of 4096.

**Control: `lr-3e-3.csv`**, peak learning rate 3e-3. It is both the control and
one point of the learning-rate sweep.

| file | what it changes against the control |
|---|---|
| `lr-3e-3.csv` | the control |
| `lr-1e-3.csv`, `lr-2e-3.csv`, `lr-4e-3.csv` | peak learning rate |
| `budget-50m.csv`, `budget-150m.csv` | training sample budget |

The budget arms vary the number of training samples, so this family has no single step
count; the learning-rate arms all run the control's budget.

These arms emit a single quantile, so their weighted quantile loss reduces exactly to
the median absolute deviation. The paper writes that column nMAD.

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

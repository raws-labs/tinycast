# Single-variable overrides

Seven settings varied one at a time against a single base, to record what was measured
and not adopted. The base is the architecture family's best arm, phase binning plus the
recency gate, 393,921 parameters, single-quantile head, 7,500 steps. All eight files
cover 97 of 97 configurations.

**Control: `control.csv`.** It is the same evaluation as
`../architecture/phasebin-recency.csv`; both copies ship so each directory reads on its
own.

| file | what it changes against the control |
|---|---|
| `phase-bins-32.csv` | `phase_bins` 16 to 32 |
| `horizon-kernel-5.csv` | `horizon_kernel` 0 to 5 |
| `kernel-5.csv` | `kernel_size` 3 to 5 |
| `gated-conv.csv` | `gated_conv` off to on |
| `harmonics-2.csv` | `n_harmonics` 1 to 2 |
| `top-k-8.csv` | `top_k_periods` 4 to 8 |
| `recency-tau.csv` | `phase_recency_tau` 0 to 0.1 |

Two caveats the paper states rather than hides. These runs postdate the freeze of the
deployed line, and `phase-bins-32.csv` scores better than the control and was not
carried forward for that reason. No bootstrap intervals were computed for these arms,
and they are not entered into the multiple-comparison family.

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

# How the deployed parameter count was arrived at

Two paired probes on the size-reduction path from the development flagship to the
146,505-parameter deployed configuration. Neither is an ablation family: each is one
matched pair.

**Width, at fixed recipe.** `control.csv` against `width-d76.csv`, model width 64 to 76
(+19%) with everything else held, on the 30,000-step nine-quantile line and all 97
configurations. `capacity_d76.json` carries the paired statistics. `control.csv` is the
same evaluation as `../ablations/component/control.csv`.

**Stacked reduction, 445,513 to 225,865 parameters.**
`shrink_control_s7000.csv` against `shrink_stacked_s7000.csv`, matched corpus, recipe
and budget, evaluated at step 7,000. `shrink_probe.json` carries the paired statistics
and the four config keys that separate the two arms. Two limits stated plainly: coverage
is partial, 43 and 44 of 97 configurations, and the comparison is paired on the 43 both
cover; and step 7,000 is a probe budget, so convergence of the delta is not
demonstrated.

The last leg, 225,865 to 146,505 parameters, is a separable-convolution change with no
isolated evaluated pair, and nothing here measures it.

## Aggregation

Per configuration, divide the arm's metric column by the same column of
`../deployed/seasonal-naive.csv`, then take the geometric mean over the configurations both
cover. That single construction produces every normalized aggregate in the paper:
applied to `eval_metrics/MASE[0.5]` it gives nGMASE, and applied to
`eval_metrics/mean_weighted_sum_quantile_loss` it gives nWQL, or nMAD where the arm
emits one quantile and that column reduces to a point error.
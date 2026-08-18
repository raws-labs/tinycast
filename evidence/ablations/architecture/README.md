# Architecture family

Which architectural component carries the accuracy. A dilated-convolution base at
340,545 parameters, single-quantile head, trained for 7,500 steps.

**Control: `control.csv`**, the dilated-convolution base with neither phase binning
nor the recency gate.

| file | what it changes against the control |
|---|---|
| `control.csv` | the base (340,545 parameters) |
| `causal.csv` | causal padding, which the streaming mode requires |
| `phasebin.csv` | phase binning (361,089) |
| `phasebin-causal.csv` | phase binning and causal padding (361,089) |
| `phasebin-recency.csv` | phase binning and the recency gate (393,921), the family's best arm |
| `detector-off.csv` | the periodicity detector off, with a phase-free recency path substituted (410,241) |

The arms are not parameter-matched, which is why each row gives its count.
`detector-off.csv` changes capacity at the same time as it removes the detector; the
capacity-neutral version of that question is `../../detector-retrain/`, which retrains
at the deployed size with the detector flag alone.

These arms emit a single quantile, so their weighted quantile loss reduces exactly to
the median absolute deviation. The paper writes that column nMAD and does not compare
it with the nWQL of a nine-quantile model.

`phasebin-recency.csv` is the same evaluation as `../overrides/control.csv`.

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

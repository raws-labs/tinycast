# Is the periodicity detector worth its structure?

Retrains, not inference-time interventions. Three arms trained from scratch to 36,621
steps at effective batch 4096 on the same corpus as the deployed model, each differing
from the shipped configuration by exactly one key, and each scored on all 97 GIFT-Eval
configurations. The detector-off flag is capacity-neutral, so every arm here
instantiates at 146,505 parameters, the deployed size.

**Control: `shipped-seed42-soup.csv`**, the deployed run itself. It joins the three
retrains as a fourth arm, giving a shipped-recipe seed band over three seeds.

Every arm is scored under two checkpoint conventions, and they disagree materially, so
neither stands in for the other:

| convention | files |
|---|---|
| final checkpoint, step 36,621 | `detector-off-seed42.csv`, `shipped-seed42-last.csv`, `shipped-seed43.csv`, `shipped-seed44.csv` |
| 8-checkpoint soup | `detector-off-seed42-soup.csv`, `shipped-seed42-soup.csv`, `shipped-seed43-soup.csv`, `shipped-seed44-soup.csv` |

**The manuscript quotes the soup**, because the published headline numbers are
8-checkpoint soups and that is the comparison matching how the deployed model is
reported. Each soup is the uniform average of that arm's own eight periodic saves
between steps 29,000 and 36,000, which excludes the final checkpoint.

The four paired-bootstrap files hold the pairwise contrasts, 10,000 resamples over
configurations, percentile intervals at 95%:

| file | what it contrasts |
|---|---|
| `arm1_vs_shipped_last_bootstrap.json` | detector-off against the deployed run, final checkpoint |
| `three_arm_last_bootstrap.json` | the three retrains, final checkpoint |
| `three_arm_soup_bootstrap.json` | the three retrains, soup |
| `four_arm_soup_bootstrap.json` | all four arms, soup. The file the per-seed contrasts are read off |

No contrast is duplicated across the four files.

A fifth file holds the contrast the manuscript reports, which is not a pairwise one.
`mean_of_three_cluster_bootstrap.json` contrasts the detector-off soup with the MEAN of
the three shipped-recipe soups, and resamples it two ways: over the 97 configurations,
and over the 28 base datasets they come from, which is the clustered construction every
comparator interval in the paper also carries. Its point estimate is the difference of
two levels in the table above, so it is checkable by subtraction; only the intervals need
the resampling. 20,000 replicates, percentile intervals at 95%.

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
# TinyCast evaluation evidence

The per-entry evaluation results behind the paper's ablation tables, its deployed
model, its deployment profiles and its two secondary benchmarks. Every file is a
frozen evaluation output, one entry per benchmark configuration or task, and every
aggregate the paper states from this tree is recomputable from the tree alone.

| directory | what it holds |
|---|---|
| `ablations/architecture/` | which architectural component carries the accuracy |
| `ablations/component/` | which recipe and inference choices are worth their cost |
| `ablations/optimization/` | whether the optimization settings sit at their optimum |
| `ablations/overrides/` | seven settings varied one at a time on a single base |
| `deployed/` | the deployed model, one file per inference and quantization profile |
| `detector-retrain/` | the periodicity detector removed and retrained from scratch |
| `detector-substitution/` | the detector's output replaced at inference time |
| `scale-path/` | how the deployed parameter count was arrived at |
| `chronos-zs/` | the 27-task Chronos zero-shot benchmark: ours and the published field |
| `fev-bench/` | the 100-task fev-bench: ours and the published comparators |
| `census-sweep/` | the leaderboard sweep behind the comparator census, one line per entry |
| `calibration/` | the static-W8A8 calibration record: which series set the scales, and how they were chosen |
| `configurations.csv` | the two per-configuration variables the paper's subgroup splits use |
| `seasonality-split.json` | the head-to-head against the next-smallest comparator, split by declared seasonality |
| `corpus-resolution.json` | how the two corpus exclusion lists resolve against the pretraining release |

Every directory above `chronos-zs/` is GIFT-Eval, over the same 97 configurations, and
`deployed/seasonal-naive.csv` is their normalizer: every ratio in them divides by it.
The two benchmark directories run over different task sets and each divides by the
published seasonal-naive reference shipped inside it. The three constructions are the
same arithmetic and their levels are not comparable with each other.

Each family has its own control and its own training line, so deltas within a family
are meaningful and absolute scores are not comparable across families.

`ablations/` is a container and holds no files of its own. Each family directory has
a README stating its control, its training line and its inference profile.

## `configurations.csv`

Two variables per configuration, and the paper splits the 97 on both.

| column | what it is |
|---|---|
| `dataset` | the configuration id every file in this tree is keyed by |
| `declared_seasonality` | what GIFT-Eval assigns the configuration's frequency: the gluonts default seasonality reduced by the offset multiple, 1 where it does not divide |
| `detector_firing_rate` | the fraction of the configuration's evaluation windows on which the periodicity detector accepted at least one period, at the deployed 2048-sample context, over at most 200 windows |

The 24 configurations at `declared_seasonality` 1 are the paper's least-seasonal group.
Seasonality 1 makes seasonal naive identical to naive, so that set can be checked against
the benchmark's own published files: it is exactly the configurations whose `naive` and
`seasonal_naive` rows agree on every eval-metric column in `../census/snapshot/`.

The paper's other seasonality group, the most seasonal quarter, is a different variable
and is not a column here because the census already carries what it is computed from: it
is the 25 configurations with the largest
`MASE(naive) / MASE(seasonal_naive)`, both read from `../census/snapshot/`. Its lower
boundary is 1.6785 against 1.5371 for the next configuration down, so the group has no
tied members.

The firing rate is measured with the predictor's own context preparation: truncate to the
last 2048 samples, left-pad with the series' first value, impute missing values, then run
the detector. A measurement that strips missing values instead reads lower on the 16
configurations that carry them, and is not what this column holds.

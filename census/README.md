# Comparator census

Every comparator number in the paper is that model's own published per-configuration
result, taken from the benchmark's public results repository and re-aggregated under
the leaderboard's rule so that every entry is normalized identically. No comparator was
run by us.

| file | what it holds |
|---|---|
| `comparator_bootstrap.json` | how much of each comparator delta survives resampling |

## `comparator_bootstrap.json`

An aggregate is a point estimate, and the 97 GIFT-Eval configurations come from 28 base
datasets, seven of which carry over half the weight. This file is the resampling that
says which differences resolve: 28 models against TinyCast, three metrics each, 84
deltas, every one paired on all 97 configurations.

The delta is TinyCast's score minus the comparator's, so a positive number means the
comparator leads. Each carries a 95% percentile interval from 20,000 resamples under
two resampling units. `config` resamples the 97 configurations. `cluster` resamples the
base datasets, which is the honest unit: short, medium and long of one
dataset-frequency are the same series at three horizons and are not independent.
`spans_zero` says whether the interval beside it contains zero; 15 of the 84 do under
clustering, and those are the deltas the paper reports as unresolved.

The pinned per-configuration snapshot these deltas are computed from, the git blob
digest of every file in it, and the script that re-aggregates it accompany the paper as
supplementary material.

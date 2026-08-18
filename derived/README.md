# Derived records

Four records the paper prints that are computed from the per-configuration results in
`../evidence/`, or from the implementation itself, rather than read off one file. Each
is regenerated from its source when this tree is assembled, so none of them is a
transcription.

| file | what it holds |
|---|---|
| `coverage_curves_gift_eval.json` | empirical coverage of the nine central prediction intervals |
| `deployment_cost_table.json` | the core call's multiply-accumulate breakdown, and the INT8 weight size of nine models against a 2 MiB flash budget |
| `parameter_counts.json` | the parameter budget of the deployed model and of every ablation arm |
| `fig_qualitative_selection.json` | which window each panel of the qualitative figure shows |

## `coverage_curves_gift_eval.json`

The deployed model at the host profile: sign symmetrization on, canonical-period
alignment on, bf16 compute. `summary` carries the pooled curve and the splits by term
and by frequency; `per_config` carries all 97 configurations, each with the same nine
nominal levels. A missing actual is masked out of the count rather than scored as not
covered, so the denominator is the number of steps a forecast could be checked
against.

## `deployment_cost_table.json`

`core_call` is the cost of re-encoding a 2048-step context and decoding one 48-step
block, as multiply-accumulates, split into the encoder, the decoder and the future-conv
stack. `census` is nine models at one byte per learned scalar, which is the comparison
the 2 MiB flash budget is read against; each row records the context length, the patch
stride and where its shape was taken from.

## `parameter_counts.json`

Counts by instantiation, which counts a weight-tied tensor once. A sum over a loaded
checkpoint does not: both feed-forward stacks are tied, and loading materializes the
shared storage under each alias. `parameter_counts.py` is what produced the record and
what re-checks it against the implementation:

```bash
python3 -B derived/parameter_counts.py --check derived/parameter_counts.json
```

It finds the implementation in this tree on its own; `--package-root` points it at
another copy. `-B` keeps the import from leaving compiled bytecode behind, which is
worth having in a tree that is read as a record.

## `fig_qualitative_selection.json`

Ten panels over seven domains: slots are allocated in proportion to each domain's
configuration count with a floor of one, and within a domain the configurations at
evenly spaced order statistics of its nGMASE are taken, so the figure shows typical
behaviour rather than selected behaviour. `window` is the window whose median absolute
error at the 0.5 quantile is the median over that configuration's windows.
`display_rule` records the truncation, which is display only: scoring uses the full
horizon.

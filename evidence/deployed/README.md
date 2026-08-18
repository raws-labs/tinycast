# The deployed model

The 146,505-parameter model the paper reports, evaluated on all 97 GIFT-Eval
configurations. There is no control here and no ablation: every file is the same
trained model under a different inference or quantization profile, which is why the
profile is spelled out in each name instead of being stated once for the directory.

| file | profile |
|---|---|
| `host.csv` | sign symmetrization on, canonical-period alignment on, bf16. The host headline |
| `host-no-symmetrization.csv` | alignment on, symmetrization off, bf16 |
| `host-no-alignment.csv` | symmetrization on, alignment off, bf16 |
| `single-pass.csv` | neither strategy, bf16. The profile a device runs |
| `single-pass-fp32.csv` | neither strategy, fp32-strict. The dtype-matched unquantized reference for the single-pass profile |
| `reference-fp32.csv` | both strategies, fp32-strict, no quantization. The denominator of the host quantization cost |
| `int8-weights-only.csv` | INT8 weights, activations in floating point |
| `w8a8-host.csv` | exact static W8A8 at the host profile: both strategies on |
| `w8a8-firmware.csv` | exact static W8A8 at the single-pass profile: the configuration the firmware executes |
| `w8a8-dynamic-DIAGNOSTIC.csv` | dynamic fake-quant W8A8. **A diagnostic, not a path the paper reports** |
| `seasonal-naive.csv` | the normalizer, not a model |
| `w8a8-host-manifest.json` | what `w8a8-host.csv` was produced under, not a result |

Five of these are the manuscript's profile table, in its order: `host.csv`,
`single-pass.csv`, `reference-fp32.csv` (the quantization reference), `w8a8-host.csv`
(the quantized host) and `w8a8-firmware.csv`. Every row of that table is scored here, on
the host, over the same 97 configurations; the board runs the fidelity chain of the
deployment appendix rather than the benchmark. Dividing `w8a8-host.csv` by
`reference-fp32.csv` gives the quantization cost the paper reports, which is why those
two are the pair to read together: same checkpoint, same strategies, same arithmetic for
the floating-point islands, quantization the only difference.

`w8a8-dynamic-DIAGNOSTIC.csv` carries the suffix because it aggregates to within 0.003
nGMASE of `w8a8-firmware.csv` while measuring something else, and the two have been
confused once already. Match the file to the profile you mean; closeness between them
is coincidence and is not a check on anything.

`host.csv` is also `../detector-substitution/control.csv`: the substitution arms are
inference-time interventions on this same checkpoint, and their control is this
evaluation unmodified.

`w8a8-host-manifest.json` is the manifest the exact static-W8A8 run was frozen under,
and `score_blind_freeze` records that the freeze happened before any score was read. It
names the baseline that run is paired against, the 97 configuration keys, the inference
recipe and the per-configuration alignment factors, and pins every input it consumed by
digest. Three of those inputs are files in this directory and their entries give the
path. Their recorded digest is of the file as the run read it, which is `seasonal-naive.csv`
and `reference-fp32.bins.json` byte for byte; `reference-fp32.csv` differs from it in the
`model` column and nowhere else, that column being the one value the export renames. The
rest of the inputs are the checkpoint, the compiled plans, the integer runtime and its
driver sources, none of which is released, and their entries say so while keeping the
digest that pins them.

## Aggregation

Per configuration, divide the arm's metric column by the same column of
`seasonal-naive.csv`, then take the geometric mean over the configurations both
cover. That single construction produces every normalized aggregate in the paper:
applied to `eval_metrics/MASE[0.5]` it gives nGMASE, and applied to
`eval_metrics/mean_weighted_sum_quantile_loss` it gives nWQL, or nMAD where the arm
emits one quantile and that column reduces to a point error.
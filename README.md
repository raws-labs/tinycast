# TinyCast

**An attention-free, 146,505-parameter dilated-convolution time-series foundation model.**

TinyCast forecasts unseen series zero-shot and emits a predictive distribution rather than a
point estimate. It replaces self-attention with a stack of dilated causal convolutions plus a
zero-parameter normalized-periodogram phase prior, so periodicity is computed from the context
instead of learned. Every learned operation is a convolution, a matrix multiplication or a
normalization, so the model runs in constant per-step memory and quantizes to INT8.

- **Parameters:** 146,505 (fp32 weights, 0.6 MB)
- **Architecture:** ten dilated causal Conv1d blocks (receptive field 2047 over a context of
  2048), depthwise-separable convolutions, a weight-tied FFN, a normalized-periodogram phase
  encoding, a phase-gather plus future-conv decoder, and nine decile quantile outputs.
- **Streaming:** left-only convolution padding, so the model streams per step in constant memory.
- **Weights:** [raws-labs/tinycast](https://huggingface.co/raws-labs/tinycast) on the Hugging
  Face Hub.
- **Paper:** [TinyCast: Probabilistic Zero-Shot Forecasting with Computed Periodicity](https://arxiv.org/abs/2608.15767)
- **License:** Apache-2.0.

This repository holds the weights loader, the predictor, the training recipe, the checkpoint
export path, the synthetic-corpus generator and the evaluation artifacts.

## Results (GIFT-Eval, zero-shot)

| Metric | Value |
|--------|-------|
| Overall **nGMASE** (point accuracy) | **0.7738** |
| Overall **nWQL** (probabilistic accuracy; the leaderboard's CRPS column) | **0.5454** |
| Overall **nMSIS** (interval score) | **0.5541** |

All three are geometric means, over the 97 [GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
configurations, of the ratio between the model's metric and the seasonal-naive reference's. They
were produced with flip-invariance symmetrization, bf16 autocast at compute, and the eval-time
period-alignment downsample rule.

**Point accuracy.** TinyCast matches the nearest family in the comparator census and is smaller
than every member of it. Reverso-Nano scores 0.760 nGMASE at 200 K parameters against our 0.7738,
and that difference does not survive resampling by base dataset; the family runs from 200 K to
2.6 M.

**Probabilistic accuracy.** TinyCast defines the frontier. It is the only model in the census
below 1.4 M parameters that emits a predictive distribution at all, and the nearest member of that
same family to carry a quantile head carries eighteen times its parameters. On GIFT-Eval it is
also the smallest zero-shot forecaster with a public per-configuration result and no declared
test-data leakage. The census criteria and the source of every parameter count are in the paper.

The per-configuration results behind the table are pinned in the package, and re-deriving the
aggregates from them needs no GPU and no benchmark data:

```python
from pathlib import Path

import tinycast
from tinycast import summarize_by_freq_bin

pinned = Path(tinycast.__file__).parent / "reference" / "gift_eval_tinycast.csv"
s = summarize_by_freq_bin(pinned)
s["overall"]["ngmase"], s["overall"]["ncrps"], s["overall"]["nmsis"]
# (0.7737836631366714, 0.5454197284419108, 0.5541246851433915)
```

It runs in about a millisecond and reproduces the table exactly.

## Install

There is no PyPI package. Install from this repository:

```bash
pip install git+https://github.com/raws-labs/tinycast.git
```

or clone it to work on it:

```bash
git clone https://github.com/raws-labs/tinycast.git
cd tinycast
pip install -e ".[dev]"        # [dev] adds the notebook test runner
```

One dependency set covers inference, evaluation, training, export and corpus generation. The
GIFT-Eval data loader is the exception: only the benchmark driver needs it, and it has its own
command because it has no PyPI release. Note the two names, which differ: the distribution is
`salesforce-gift-eval` and the import is `gift_eval`.

```bash
pip install "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git"
```

Building the synthetic corpus additionally needs a CUDA device, which no dependency can express.

## Load the weights

The release is a `model.safetensors` (0.6 MB) plus a `config.json`. Point `load_model` at the
weights file and the config beside it is picked up automatically; pass a second path to override
that.

```python
from huggingface_hub import hf_hub_download
from tinycast import load_model

hf_hub_download("raws-labs/tinycast", "config.json")     # lands beside the weights
st = hf_hub_download("raws-labs/tinycast", "model.safetensors")
model, config = load_model(st)                           # TinyCastForPrediction, 146,505 params
```

The notebooks resolve the weights the same way, or from a local copy when `TINYCAST_WEIGHTS`
names one, which is how they run with no network:

```bash
export TINYCAST_WEIGHTS=/path/to/model.safetensors   # or the directory holding it
```

When neither resolves, the weight-dependent cells raise. `TINYCAST_SKIP_WEIGHTS=1` downgrades that
to a skip and says, at each step it skips, that the weight-dependent claims went unverified. The
training notebook needs no weights at all.

The file stores 146,505 values across 121 tensors, while a `state_dict()` of the loaded model sums
to 321,225: both FFN stacks are weight-tied and loading materializes the shared storage under each
alias. Take the parameter count from a model instantiated from its config, never from a sum of
checkpoint tensors.

## Forecast

```python
from tinycast import TinyCastPredictor

predictor = TinyCastPredictor(prediction_length=48, checkpoint_path=st,
                              freq="H", domain="Energy", device="cpu")
forecasts = predictor.predict(gift_eval_test_input)   # gluonts QuantileForecasts
```

`TinyCastPredictor` is the gluonts predictor the published numbers were produced with. AR-rollout
decoding in 48-step chunks and NaN imputation are unconditional; flip-invariance symmetrization
needs `force_flip_invariance=True`, and period-alignment downsampling needs a `downsample_factor`,
which the benchmark driver derives per configuration. The call above leaves those two at their
defaults and so does not reproduce the table; `python -m tinycast.eval` sets both.

## Reproduce the GIFT-Eval numbers

```bash
export GIFT_EVAL=/path/to/gift-eval          # the GIFT-Eval data directory
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

# all 97 configs -> leaderboard-format all_results.csv (+ .bins.json summary)
python -m tinycast.eval --ckpt model.safetensors --flip \
    --device cuda --output gift_eval_submission/all_results.csv
```

`evaluate()` returns the summary it writes, so a caller can assert on the aggregates without
re-reading the CSV. Any configuration a reference lookup or a missing value would have dropped
from an aggregate raises instead; `--no-strict-summary` downgrades that to a warning and reports
the counts under `summary["dropped"]`.

**What changes the numbers.** The `OMP_*` caps are not optional: without them per-config
evaluation is roughly 40 times slower on many-core machines. bf16 autocast engages only on CUDA;
on CPU the driver runs fp32, which shifts results by about 1e-3. Two environment variables read by
the predictor change what it emits, and neither is on by default: `TINYCAST_INT8` (`w8` or `w8a8`
post-training fake quantization) and `TINYCAST_TILT_K` (quantile tilting, with
`TINYCAST_TILT_MODE` read only while the tilt is on). Leave them unset to reproduce the table
above. A named device must exist: `TinyCastPredictor(device="cuda")` on a host with no CUDA device
raises rather than substituting CPU. `device=None` is the one request that picks a device, and
`predictor.device` is the one that ran.

See `notebooks/tinycast.ipynb` for the end-to-end load, evaluate and summarize walkthrough, and
`notebooks/training.ipynb` for the objective, the rollout and the release path. Both double as the
test suite: `pip install -e ".[dev]"` then `pytest --nbmake notebooks/` runs every cell and its
assertions. `TINYCAST_NB_SCALE` sets how much work each does: `smoke` is one benchmark
configuration and a dozen training steps, `full` is all 97 and a few hundred. The inference
notebook defaults to `full` and the training notebook to `smoke`, and either setting runs the same
lines.

## Train

```python
from tinycast import TinyCastConfig, train, training_window_width

config = TinyCastConfig()
width  = training_window_width(config)        # 2240 = 2048 + 4 * 48
result = train(config, data=windows, max_steps=1000, output_dir="run/",
               batch_size=32, device="cuda", checkpoint_every=100)
```

`train` runs the shipped recipe single-process: a four-block autoregressive rollout under
scheduled sampling, the nine-quantile pinball loss plus the gated committing term of
`tinycast.losses`, AdamW, and a warmup-stable-decay learning rate schedule. `data` is a `Dataset`,
an `IterableDataset`, a `DataLoader` or a plain iterable of samples; a sample carries a `window` of
`L + K * p` values and optionally a `mask` and a `scale_factor`. CPU is a first-class device, and a
request for an accelerator that is not present raises rather than falling back.

**What a short run reproduces, and what it does not.** The released checkpoint is 36,621 optimizer
steps at an effective batch of 4096 (128 per GPU, eight GPUs, four gradient accumulation steps),
bf16-mixed, seed 42, peak learning rate 3e-3 with 5% warmup and 35% decay, over GIFT-Eval-Pretrain
and Chronos KernelSynth from their publishers plus the four synthetic shards below. The released
weights are the uniform mean of the last eight periodic checkpoints. A short run here reproduces
the mechanics: the rollout, the objective, the schedule shape and the optimizer. It does not
reproduce the checkpoint or its numbers. This entry point also trains in the model's own precision
rather than bf16-mixed, and writes periodic checkpoints without averaging them.

`max_steps` reshapes a run rather than truncating one. Warmup, stable and decay are fractions of
it and the scheduled-sampling probability ramps over its first half, so a 200-step run is not the
prefix of a 36,621-step run.

## Export a release

```bash
python -m tinycast.export --out-dir release/ \
    --average ckpt1 ... ckpt8 --expect-parameters 146505
```

`average_checkpoints` is the uniform mean the release was made with, and `export_safetensors`
writes the `model.safetensors` plus `config.json` pair, storing each shared FFN storage once and
recording the aliases in the file's metadata. `check_export_roundtrip` reloads the result and
verifies that the forecasts are bitwise identical, that the tie is restored as object identity
rather than as equal copies, and that the file sums to the instantiated parameter count.

Pass the checkpoints in training order: an eight-term float32 sum is not associative, so reordering
them moves the result by about one ulp. If you byte-diff an export against the published artifact,
the `config.json`, the tensor payload and the tensor index match while the safetensors header's
metadata block does not, because the writer emits those keys in an order of its own; the mapping
they encode is verified against the model's own sharing groups.

## Rebuild the synthetic corpus

```bash
python -m tinycast.corpus build --shard 0            # needs a CUDA GPU
python -m tinycast.corpus verify synth4096_0 --shard 0
```

`verify` also takes an `https://` prefix or an `hf://<owner>/<repo>[/<subdir>]` source, and
range-reads only the rows it compares, so checking a remote shard pulls about 160 KB rather than
the half gigabyte the payload occupies.

`tinycast.synth` holds the three generator families (Gaussian process, spike trains,
trend-seasonal-impulse) and `tinycast.corpus` holds the recipe that turns them into shards: four
shards of 62,500 series at length 4,096, mixed 70/15/15, about 0.5 GB of float16 payload each.
Everything that changes the generated bytes is a default in `CANONICAL_RECIPE` rather than an
argument, so `build_shard()` with no arguments reproduces published shard 0. The per-shard family
salts are carried there as data because they cannot be recomputed.

**Reproducibility.** Regeneration reproduces the published float16 payload exactly on the stack it
was generated on: 0 of 327,680 values differ across four shards by twenty rows on an RTX 3090 with
torch 2.10.0+cu128 and driver 580.159.03. Other GPUs, CUDA versions and torch builds are untested,
and the Gaussian-process family factorizes through cuSOLVER. `verify_shard` is how you find out for
a given stack: it regenerates a prefix, range-reads only the rows it compares, and reports how many
float16 values differ. Compare against `series.f16`, the payload, and not against
`series_mean.f32`, whose last bit moves with the numpy build.

CUDA is required and is enforced rather than detected. Off CUDA the Gaussian-process draws come
from a different generator, so the same seed produces different series; a build with no GPU
therefore fails instead of quietly producing a different corpus. The `gp_batch` of 32 is part of
the data for the same reason, not a memory knob.

The generator families follow the recipe published by Reverso (arXiv 2602.17634, Appendix A). That
paper leaves several ranges and probabilities symbolic, and the work it defers to does not publish
them either; the choices made here are recorded as assumptions A1 to A4 in the `tinycast.synth`
module docstring.

## What is in here

```
tinycast/
  periodogram.py       # zero-parameter normalized-periodogram period detector
  encoding.py          # phase / bounded-recency positional encodings
  backbone.py          # DilatedConvBackbone (dilated-conv encoder + decoder)
  normalization.py     # per-window min-max normalizer
  scale.py             # frequency to seasonal scale factor
  model.py             # TinyCastForPrediction / TinyCastBackbone assembly
  config.py            # TinyCastConfig
  checkpoint.py        # load_model (safetensors + config.json)
  predictor.py         # TinyCastPredictor (AR-rollout gluonts predictor)
  downsample.py        # eval-time period-alignment rule
  quant.py             # INT8 post-training fake-quant (deployment study)
  losses.py            # pinball loss, gated committing term, seasonal copy
  train.py             # the shipped training recipe
  export.py            # checkpoint averaging -> released artifact
  eval.py              # GIFT-Eval driver and the normalized aggregates
  synth.py             # the three synthetic generator families
  corpus.py            # the canonical corpus recipe, builder and verifier
  reference/           # dataset properties, seasonal-naive denominators,
                       # and the pinned per-config results behind the table
gift_eval_submission/  # GIFT-Eval leaderboard submission metadata
notebooks/             # the two walkthroughs, which are also the test suite
```

Distributed training orchestration is not part of this package: `train` is single-process, and the
eight-GPU launcher the released checkpoint was produced with is not included.

## Citation

```bibtex
@misc{tinycast2026,
  title         = {TinyCast: Probabilistic Zero-Shot Forecasting with Computed Periodicity},
  author        = {Armin Steinhauser},
  year          = {2026},
  eprint        = {2608.15767},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2608.15767}
}
```

## License and attribution

Apache-2.0 (see `LICENSE`). Third-party attributions are in `NOTICE`.

**Model:** [raws-labs/tinycast](https://huggingface.co/raws-labs/tinycast)

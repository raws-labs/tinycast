# GIFT-Eval leaderboard submission

A leaderboard PR to [`SalesforceAIResearch/gift-eval`](https://github.com/SalesforceAIResearch/gift-eval)
adds `results/TinyCast/` containing two files:

- **`config.json`**: the model metadata (in this folder).
- **`all_results.csv`**: the 97-config results in leaderboard column order.

`all_results.csv` is not checked in here (it is regenerated, and the shippable
numbers require CUDA/bf16). Produce it with:

```bash
export GIFT_EVAL=/path/to/gift-eval
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python -m tinycast.eval --ckpt model.safetensors --flip \
    --device cuda --output gift_eval_submission/all_results.csv
```

or run [`../notebooks/tinycast.ipynb`](../notebooks/tinycast.ipynb) (the `code_link`).

Expected overall: **nGMASE 0.7738 / nWQL 0.5454** (flip-invariance + bf16 +
period-alignment).

## Labeling rationale

`model_type` is **`pretrained`**. The board reserves `zero-shot` for models whose
training corpus overlaps neither GIFT-Eval nor GIFT-Eval-Pretrain; NX-AI submits
TiRex-2 under both types, and its zero-shot checkpoint is documented as trained
on datasets that do not overlap with either. TinyCast trains on
GIFT-Eval-Pretrain, so `pretrained` is the honest declaration under that field.
The label records where the training data came from, not how the model forecasts:
TinyCast predicts unseen series without parameter updates, which is also how
Chronos-ZS and fev-bench evaluate it.

`testdata_leakage` is **`No`**. The subsets of GIFT-Eval-Pretrain that overlap the
GIFT-Eval test split are excluded before caching, so no evaluation data is
trained on. `model_dtype` is `bfloat16`, the compute dtype that produced the
numbers.

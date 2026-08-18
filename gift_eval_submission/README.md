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

`model_type` is **`zero-shot`**. The benchmark states two requirements for that
tag: do not leak test data, and do not train on the train split of any GIFT-Eval
dataset. Using GIFT-Eval-Pretrain is not disqualifying, and the board's own
entries settle it. IBM ships two models on the same corpus family and they are
classified differently on exactly this line: `ibm-research/flowstate` trains on
"a subset of Gift-Eval Pretrain, and a subset of the Chronos Pretraining Data
Corpus" and is accepted as `zero-shot`, while `ibm-research/ttm-r3` trains on
"GiftEvalPretrain, and Train" and is `pretrained`. Salesforce's own Moirai2
trains on Pretrain plus Train and is `pretrained`.

TinyCast trains on GIFT-Eval-Pretrain, Chronos KernelSynth and four synthetic
shards, and never on `Salesforce/GiftEval`, so it meets both requirements. Its
corpus is the same in kind as FlowState's.

`testdata_leakage` is **`No`**. The subsets of GIFT-Eval-Pretrain that overlap the
GIFT-Eval test split are excluded before caching, so no evaluation data is
trained on. `model_dtype` is `bfloat16`, the compute dtype that produced the
numbers.

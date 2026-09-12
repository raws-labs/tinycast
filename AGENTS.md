# tinycast

The released TinyCast model: an attention-free, 146,505-parameter time-series foundation
model, with the training, export, evaluation and synthetic-corpus code behind it. Python
package (setuptools; torch, gluonts, safetensors). Public, github.com/raws-labs/tinycast.

## Build, test, run
- `pip install -e ".[dev]"`: editable install plus the test runner (pytest, nbmake, ipykernel)
- `pytest --nbmake notebooks/`: the test suite. The two notebooks are the tests; every cell ends
  in assertions. `TINYCAST_NB_SCALE=smoke` evaluates one GIFT-Eval configuration instead of 97,
  `TINYCAST_WEIGHTS=<dir>` resolves `model.safetensors` offline, `TINYCAST_SKIP_WEIGHTS=1` opts
  out of the weight-dependent sections (announced loudly, never a silent pass)
- `python -m tinycast.eval --ckpt model.safetensors --flip --device cuda --output all_results.csv`:
  the GIFT-Eval driver. Needs `GIFT_EVAL=/path/to/gift-eval` and the loader, which is not on PyPI:
  `pip install "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval.git"`
  (distribution `salesforce-gift-eval`, import name `gift_eval`)
- `python -m tinycast.export --average <ckpt...> --out-dir <dir> --expect-parameters 146505`:
  checkpoint averaging into the released `model.safetensors` + `config.json` pair, with a
  reload round trip unless `--no-check`
- `python -m tinycast.corpus build --shard N` and `verify <source>`: the synthetic corpus;
  refuses to run without CUDA rather than fall back to a CPU path that yields different data
- No CI workflow. Weights are not in this repo: `hf_hub_download("raws-labs/tinycast", ...)` fetches
  `model.safetensors` and `config.json`; `load_model(path)` expects them as siblings

## Layout
- `tinycast/reference/`: dataset properties, seasonal-naive denominators and the pinned
  per-configuration results (`gift_eval_tinycast.csv`) the published aggregates re-derive from
- `evidence/`: frozen per-configuration evaluation outputs behind the paper's tables, one README
  per family stating its control and inference profile; `derived/` and `census/` are computed from it
- `gift_eval_submission/`: leaderboard `config.json`; its `all_results.csv` is regenerated, not committed
- `examples/load/`: zero-shot electricity-load example and a year-long backtest on Elia open data

## Conventions
- Installed from git; there is no PyPI release. Version lives in `pyproject.toml`
- `tinycast[train]` is an extra that installs nothing; it names the path for docs and CI

## Gotchas
- Take the parameter count from a model instantiated from its config. Summing `state_dict()`
  gives 321,225 because the weight-tied FFN is materialized under both aliases
- `TinyCastPredictor` defaults do not reproduce the table: flip-invariance symmetrization and
  period alignment are off there; `tinycast.eval` turns both on. The published profile is bf16
  autocast, `--flip`, period-alignment downsample
- Export `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4`
  before torch is imported; without the caps evaluation is about 40x slower on many-core hosts
- `TINYCAST_INT8` (`w8`, `w8a8`) and `TINYCAST_TILT_K` change what the predictor emits; both must
  be unset for the published numbers, and the notebooks assert that
- `/all_results.csv`, `/all_results.bins.json` and their `notebooks/` copies are gitignored eval
  output; the committed per-configuration results are `tinycast/reference/gift_eval_tinycast.csv`
- `*.ckpt` is never a release artifact; the release is `model.safetensors` from `tinycast.export`
- Corpus shards (`/synth4096_*/`) are tens of GB each and gitignored

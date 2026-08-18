"""TinyCast GIFT-Eval driver.

Runs the deployed TinyCast predictor against the GIFT-Eval benchmark and writes
a results CSV in the leaderboard column order (98 lines incl. header, 15
columns), plus a per-frequency-bin summary sidecar carrying the three
normalized aggregates (nGMASE, nCRPS, nMSIS).

``summarize_by_freq_bin`` also runs standalone on any leaderboard-format CSV, so
the published aggregates can be re-derived offline from the pinned per-config
results in ``reference/gift_eval_tinycast.csv`` without a GPU or benchmark data.
Called that way it only returns and prints the summary; pass
``write_sidecar=True`` to also write it beside the CSV.

Usage:
    python -m tinycast.eval --ckpt model.safetensors --flip \\
        --output all_results.csv                       # full 97 configs
    python -m tinycast.eval --ckpt model.safetensors --flip \\
        --configs "m4_hourly/H/short" --output smoke.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

MODEL_NAME_DEFAULT = "TinyCast"

# ---------------------------------------------------------------------------
# Dataset constants: the GIFT-Eval protocol's dataset / term layout.
# ---------------------------------------------------------------------------
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W "
    "hospital covid_deaths "
    "us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
    "SZ_TAXI/15T SZ_TAXI/H "
    "M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W ett2/W ett2/D "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)

MED_LONG_DATASETS = (
    "electricity/15T electricity/H "
    "solar/10T solar/H "
    "kdd_cup_2018_with_missing/H "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T M_DENSE/H "
    "ett1/15T ett1/H ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)

# Leaderboard CSV column order (byte-identical for leaderboard submission).
LEADERBOARD_COLUMNS = (
    "dataset", "model",
    "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]",
    "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS", "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]", "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "domain", "num_variates",
)

REFERENCE_DIR = Path(__file__).parent / "reference"


def _build_metrics():
    from gluonts.ev.metrics import (
        MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
        MeanWeightedSumQuantileLoss,
    )
    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]


def _load_dataset_properties() -> dict:
    with open(REFERENCE_DIR / "dataset_properties.json") as f:
        return json.load(f)


def iter_configs(
    configs_filter: Optional[Sequence[str]] = None,
    term_filter: Optional[str] = None,
) -> Iterable[tuple]:
    """Yield ``(ds_name, term, ds_config_key, ds_key, ds_freq)`` per (dataset, term).

    ``configs_filter`` matches by full ``ds_config_key`` (e.g.
    ``ett1/15T/long``); if ``None`` (default), all 97 configs are yielded.
    """
    dataset_properties = _load_dataset_properties()
    all_datasets = sorted(set(SHORT_DATASETS.split() + MED_LONG_DATASETS.split()))
    med_long_set = set(MED_LONG_DATASETS.split())
    all_terms = ["short", "medium", "long"] if term_filter is None else [term_filter]
    wanted = set(configs_filter) if configs_filter is not None else None

    for ds_name in all_datasets:
        if "/" in ds_name:
            ds_key = PRETTY_NAMES.get(ds_name.split("/")[0].lower(), ds_name.split("/")[0].lower())
            ds_freq = ds_name.split("/")[1]
        else:
            ds_key = PRETTY_NAMES.get(ds_name.lower(), ds_name.lower())
            ds_freq = dataset_properties[ds_key]["frequency"]

        for term in all_terms:
            if term in ("medium", "long") and ds_name not in med_long_set:
                continue
            ds_config_key = f"{ds_key}/{ds_freq}/{term}"
            if wanted is not None and ds_config_key not in wanted:
                continue
            yield ds_name, term, ds_config_key, ds_key, ds_freq


def evaluate(
    ckpt_path: str,
    output_csv: os.PathLike,
    *,
    model_name: str = MODEL_NAME_DEFAULT,
    configs_filter: Optional[Sequence[str]] = None,
    term_filter: Optional[str] = None,
    force_flip_invariance: bool = True,
    device: Optional[str] = None,
    use_amp: int = 1,
    period_align: bool = True,
    batch_size: int = 64,
    predictor_batch_size: Optional[int] = None,
    strict_summary: bool = True,
) -> dict:
    """Run GIFT-Eval for TinyCast; stream rows to ``output_csv``.

    ``device`` defaults to ``None``, which is the predictor's request to choose:
    CUDA when it is present, CPU otherwise. Naming a device instead makes it a
    requirement, and an absent one raises rather than being substituted.

    Returns the ``summarize_by_freq_bin`` summary for the run it just wrote, so
    a caller can assert on the aggregates without re-reading the CSV. The row
    CSV is complete on disk before the summary runs, so a ``strict_summary``
    failure costs no evaluation work: fix the reference and re-summarize.
    """
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    from gift_eval.data import Dataset

    from .predictor import TinyCastPredictor

    metrics = _build_metrics()
    dataset_properties = _load_dataset_properties()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        csv.writer(f).writerow(LEADERBOARD_COLUMNS)

    configs = list(iter_configs(configs_filter, term_filter))
    print(f"Evaluating {model_name} on {len(configs)} configs.")
    print(f"Output: {output_csv}")

    for i, (ds_name, term, ds_config, ds_key, ds_freq) in enumerate(configs):
        print(f"[{i + 1}/{len(configs)}] {ds_config}", flush=True)

        probe = Dataset(name=ds_name, term=term, to_univariate=False)
        to_uni = probe.target_dim != 1
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_uni)
        season_length = get_seasonality(dataset.freq)

        per_cfg_kwargs = dict(
            device=device,
            force_flip_invariance=bool(force_flip_invariance),
            use_amp=use_amp,
        )
        if predictor_batch_size is not None:
            per_cfg_kwargs["batch_size"] = predictor_batch_size
        # bizitobs_l2c has no daily cycle, so scale_factor /= 7.
        if "l2c" in ds_name.lower():
            per_cfg_kwargs["no_daily"] = True
        # Period-alignment downsampling: derive the aliasing factor from the
        # test INPUT contexts only (fires k=7 on bizitobs_l2c/5T med+long).
        if period_align:
            from .downsample import freq_to_seconds, period_alignment_factor
            fs = freq_to_seconds(ds_freq)
            if fs is not None:
                k_align = period_alignment_factor(
                    (np.asarray(e["target"], dtype=np.float64)
                     for e in dataset.test_data.input),
                    fs,
                    horizon=dataset.prediction_length,
                )
                if k_align > 1:
                    per_cfg_kwargs["downsample_factor"] = k_align
                    print(f"  period-align: downsample_factor={k_align}", flush=True)

        predictor = TinyCastPredictor(
            prediction_length=dataset.prediction_length,
            checkpoint_path=ckpt_path,
            freq=ds_freq,
            domain=dataset_properties[ds_key]["domain"],
            **per_cfg_kwargs,
        )

        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=batch_size,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )

        with open(output_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                ds_config, model_name,
                res["MSE[mean]"].iloc[0], res["MSE[0.5]"].iloc[0],
                res["MAE[0.5]"].iloc[0], res["MASE[0.5]"].iloc[0],
                res["MAPE[0.5]"].iloc[0], res["sMAPE[0.5]"].iloc[0],
                res["MSIS"].iloc[0], res["RMSE[mean]"].iloc[0],
                res["NRMSE[mean]"].iloc[0], res["ND[0.5]"].iloc[0],
                res["mean_weighted_sum_quantile_loss"].iloc[0],
                dataset_properties[ds_key]["domain"],
                dataset_properties[ds_key]["num_variates"],
            ])

        import gc
        del predictor, res, dataset, probe
        gc.collect()

    print(f"Done. Results: {output_csv}")
    # A partial run (--configs / --term) covers a subset of the 97 leaderboard
    # configurations, but every configuration it does cover must be scored, so
    # the strict reference check stays on.
    return summarize_by_freq_bin(
        output_csv, strict=strict_summary, write_sidecar=True,
    )


# ---------------------------------------------------------------------------
# Per-frequency-bin normalized summary (sidecar to the leaderboard CSV)
# ---------------------------------------------------------------------------
_FREQ_BIN_ORDER = ("sub_hourly", "hourly", "daily_or_coarser")

# The three reported GIFT-Eval aggregates, each the geometric mean of the
# per-configuration ratio of the column below to the same column of the
# seasonal-naive reference. One construction, three columns.
_NORMALIZED_METRICS = (
    ("ngmase", "eval_metrics/MASE[0.5]"),
    ("ncrps", "eval_metrics/mean_weighted_sum_quantile_loss"),
    ("nmsis", "eval_metrics/MSIS"),
)

# The GIFT-Eval leaderboard calls the quantile-loss aggregate nWQL; the paper
# calls it nCRPS. Same column, so the summary carries both spellings.
_METRIC_ALIASES = (("nwql", "ncrps"),)


def _freq_bin(freq: str) -> str:
    f = freq.strip().upper()
    if f == "H":
        return "hourly"
    if f.endswith("S") or f.endswith("T") or f == "MIN":
        return "sub_hourly"
    return "daily_or_coarser"


def _positive_float(value) -> Optional[float]:
    """Parse ``value``, returning ``None`` unless it is finite and positive.

    Every aggregate here is a geometric mean, so a zero, a negative or a NaN is
    not a small contribution: it is undefined.
    """
    try:
        out = float(value)
    except (ValueError, TypeError):
        return None
    return out if math.isfinite(out) and out > 0 else None


def _read_reference_metrics(reference_csv: os.PathLike) -> dict:
    """Map ``dataset`` to its seasonal-naive denominator per normalized metric."""
    out: dict = {}
    with open(reference_csv, newline="") as f:
        for row in csv.DictReader(f):
            ds = row.get("dataset")
            if not ds:
                continue
            denominators = {}
            for _, column in _NORMALIZED_METRICS:
                den = _positive_float(row.get(column))
                if den is not None:
                    denominators[column] = den
            out[ds] = denominators
    return out


def _geometric_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summarize_by_freq_bin(
    output_csv: os.PathLike,
    *,
    seasonal_naive_csv: "os.PathLike | None" = None,
    strict: bool = True,
    write_sidecar: bool = False,
) -> dict:
    """Read ``output_csv``, print the per-frequency-bin summary and return it.

    Three aggregates are reported, all built the same way: the geometric mean
    over configurations of the ratio between a column and the same column of the
    seasonal-naive reference. ``ngmase`` normalizes MASE (point accuracy),
    ``ncrps`` normalizes the mean weighted sum quantile loss (probabilistic
    accuracy, the leaderboard's CRPS column, carried under both keys),
    and ``nmsis`` normalizes MSIS (interval score).

    ``write_sidecar`` (default off) additionally writes the returned summary to
    ``<stem>.bins.json`` beside ``output_csv``. It is off by default because the
    common call re-derives the published aggregates from the pinned results
    inside the installed package, where the sidecar path lands in
    ``site-packages``: reading a file must not write one. ``evaluate`` turns it
    on, since there the CSV is the caller's own output path.

    ``strict`` (default on) raises when any configuration in ``output_csv`` is
    left out of an aggregate, whether because its dataset key is unparseable,
    because the reference file does not cover it, or because a value is missing
    or non-positive. A geometric mean says nothing about how many terms it has,
    so a reference file that has drifted away from the run would otherwise
    quietly shrink the sample and return a number that looks publishable. The
    dropped configurations are recorded under ``summary["dropped"]`` either way,
    so a caller running with ``strict=False`` can still assert on the count.
    """
    output_csv = Path(output_csv)
    if seasonal_naive_csv is None:
        seasonal_naive_csv = REFERENCE_DIR / "seasonal_naive.csv"
    seasonal_naive_csv = Path(seasonal_naive_csv)
    reference = (
        _read_reference_metrics(seasonal_naive_csv)
        if seasonal_naive_csv.exists()
        else {}
    )

    mase_per_bin = {b: [] for b in _FREQ_BIN_ORDER}
    ratios = {key: {b: [] for b in _FREQ_BIN_ORDER}
              for key, _ in _NORMALIZED_METRICS}
    rows_per_bin = {b: 0 for b in _FREQ_BIN_ORDER}
    dropped_reasons: dict = {}
    n_rows = 0

    def drop(dataset: str, reason: str) -> None:
        dropped_reasons.setdefault(reason, []).append(dataset)

    with open(output_csv, newline="") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            ds = row.get("dataset") or ""
            parts = ds.split("/")
            if len(parts) < 3:
                drop(ds, "unparseable_dataset_key")
                continue
            bin_name = _freq_bin(parts[1])
            rows_per_bin[bin_name] += 1

            mase = _positive_float(row.get("eval_metrics/MASE[0.5]"))
            if mase is not None:
                mase_per_bin[bin_name].append(mase)

            denominators = reference.get(ds)
            if denominators is None:
                drop(ds, "no_seasonal_naive_reference")
                continue
            for key, column in _NORMALIZED_METRICS:
                numerator = _positive_float(row.get(column))
                if numerator is None:
                    drop(ds, f"no_value_{key}")
                elif column not in denominators:
                    drop(ds, f"no_reference_value_{key}")
                else:
                    ratios[key][bin_name].append(numerator / denominators[column])

    summary: dict = {}
    for bin_name in list(_FREQ_BIN_ORDER) + ["overall"]:
        bins = _FREQ_BIN_ORDER if bin_name == "overall" else (bin_name,)
        mase_vals = [v for b in bins for v in mase_per_bin[b]]
        entry = {
            "n": sum(rows_per_bin[b] for b in bins),
            "gmean_mase": _geometric_mean(mase_vals),
        }
        for key, _ in _NORMALIZED_METRICS:
            vals = [v for b in bins for v in ratios[key][b]]
            entry[key] = _geometric_mean(vals)
            entry[f"n_{key}"] = len(vals)
        for alias, key in _METRIC_ALIASES:
            entry[alias] = entry[key]
            entry[f"n_{alias}"] = entry[f"n_{key}"]
        summary[bin_name] = entry

    n_dropped = sum(len(v) for v in dropped_reasons.values())
    summary["dropped"] = {
        "n_rows": n_rows,
        "n_dropped": n_dropped,
        "by_reason": {k: sorted(set(v)) for k, v in sorted(dropped_reasons.items())},
    }
    summary["reference"] = str(seasonal_naive_csv)

    if write_sidecar:
        sidecar = output_csv.with_suffix(".bins.json")
        sidecar.write_text(json.dumps(summary, indent=2))

    def fmt(value) -> str:
        return f"{value:.4f}" if value is not None else "n/a"

    header = "Per-frequency-bin GIFT-Eval aggregates"
    if write_sidecar:
        header += f"  (sidecar: {sidecar.name})"
    print(header)
    print(f"  {'bin':<18} {'n':>3}  {'nGMASE':>8} {'nCRPS':>8} {'nMSIS':>8}"
          f" {'gmean MASE':>11}")
    for bin_name in list(_FREQ_BIN_ORDER) + ["overall"]:
        s = summary[bin_name]
        print(f"  {bin_name:<18} {s['n']:>3}  {fmt(s['ngmase']):>8}"
              f" {fmt(s['ncrps']):>8} {fmt(s['nmsis']):>8}"
              f" {fmt(s['gmean_mase']):>11}")

    if n_dropped:
        detail = ", ".join(f"{reason}: {len(set(datasets))}"
                           for reason, datasets in sorted(dropped_reasons.items()))
        message = (
            f"{n_dropped} of {n_rows} configurations in {output_csv} were left "
            f"out of an aggregate ({detail}). The reference "
            f"{seasonal_naive_csv} does not match this run, so the aggregates "
            f"above are geometric means over an incomplete sample."
        )
        if strict:
            raise ValueError(message)
        print(f"  WARNING: {message}")

    return summary


def _parse_configs_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [c.strip() for c in value.split(",") if c.strip()]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run GIFT-Eval against the deployed TinyCast model."
    )
    parser.add_argument("--ckpt", required=True,
                        help="Path to model.safetensors (config.json sibling).")
    parser.add_argument("--configs", default=None,
                        help="Comma-separated ds_config_keys (e.g. "
                             "'ett1/15T/long,m4_hourly/H/short'). Default: all 97.")
    parser.add_argument("--term", default=None, choices=["short", "medium", "long"])
    parser.add_argument("--flip", dest="force_flip_invariance",
                        action="store_true", default=True,
                        help="Flip-invariance symmetrization (deployed default ON).")
    parser.add_argument("--no-flip", dest="force_flip_invariance",
                        action="store_false")
    parser.add_argument("--no-period-align", action="store_true",
                        help="Disable period-alignment downsampling (default ON).")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32-strict"],
                        help="bf16 autocast at compute (CUDA only) or strict fp32.")
    parser.add_argument("--device", default=None,
                        help="cuda | cpu. Default: whichever is present "
                             "(CUDA when available, else CPU). Naming one "
                             "makes it a requirement: an absent device is an "
                             "error, never a silent substitution.")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="gluonts.evaluate_model aggregation batch size.")
    parser.add_argument("--predictor-batch-size", type=int, default=None,
                        help="Internal predictor batch size (default 256).")
    parser.add_argument("--model-name", default=MODEL_NAME_DEFAULT,
                        help="Value written to the 'model' CSV column.")
    parser.add_argument("--output", default="all_results.csv",
                        help="Output CSV path (leaderboard format).")
    parser.add_argument("--no-strict-summary", action="store_true",
                        help="Warn instead of failing when the seasonal-naive "
                             "reference does not cover every evaluated config.")
    args = parser.parse_args(argv)

    evaluate(
        ckpt_path=args.ckpt,
        output_csv=args.output,
        model_name=args.model_name,
        configs_filter=_parse_configs_arg(args.configs),
        term_filter=args.term,
        force_flip_invariance=bool(args.force_flip_invariance),
        device=args.device,
        use_amp=0 if args.dtype == "fp32-strict" else 1,
        period_align=not args.no_period_align,
        batch_size=args.batch_size,
        predictor_batch_size=args.predictor_batch_size,
        strict_summary=not args.no_strict_summary,
    )


if __name__ == "__main__":
    main()

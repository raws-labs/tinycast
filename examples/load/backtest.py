"""Year-long day-ahead backtest of Belgian electricity load.

Reproduces the comparison against the transmission system operator's own
day-ahead forecast under matched conditions: both forecasts are issued at the
same moment and cover the same 24 hours.

The operator publishes its day-ahead forecast at 08:45 local time for the whole
of the following calendar day. Each TinyCast forecast is therefore issued from
history up to 09:00 local and scored on hours 15 to 38 ahead, which is the next
calendar day. Neither forecaster sees the target day when it commits.

    python backtest.py --checkpoint model.safetensors \
        --start 2025-08-01 --end 2026-07-31 --out backtest.csv
"""
from __future__ import annotations

import argparse
import pickle
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

from tinycast import TinyCastPredictor

API = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/records"
CONTEXT, HORIZON = 2048, 48
GATE_HOUR = 9              # local time the operator's day-ahead forecast is out
OFFSET, LENGTH = 15, 24    # hours 15..38 ahead = the next calendar day
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def fetch(start: str, end: str) -> pd.DataFrame:
    """Measured load and day-ahead forecast for Belgium, hourly, in MW.

    The API caps paging at 9,900 rows, so the range is walked in 25-day chunks.
    """
    rows = []
    for chunk in pd.date_range(start, end, freq="25D"):
        stop = min(chunk + pd.Timedelta(days=25), pd.Timestamp(end))
        offset = 0
        while True:
            query = urllib.parse.urlencode({
                "where": f"datetime >= date'{chunk:%Y-%m-%dT%H:%M:%S}' "
                         f"AND datetime < date'{stop:%Y-%m-%dT%H:%M:%S}'",
                "select": "datetime,totalload,dayaheadforecast",
                "order_by": "datetime asc",
                "limit": 100,
                "offset": offset,
            })
            with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:
                page = pd.read_json(response).get("results", pd.Series(dtype=object))
            if len(page) == 0:
                break
            rows.extend(page.tolist())
            offset += 100
            if offset >= 9900:
                break

    frame = pd.DataFrame(rows).drop_duplicates(subset="datetime")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame.set_index("datetime").sort_index()
    return frame.resample("1h").mean().dropna(subset=["totalload"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--start", required=True, help="first target day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="last target day, YYYY-MM-DD")
    parser.add_argument("--out", default="backtest.csv")
    parser.add_argument("--raw", default="backtest.pkl", help="per-day series, for figures")
    parser.add_argument("--cache", help="pickled frame from a previous fetch")
    args = parser.parse_args()

    if args.cache:
        frame = pickle.load(open(args.cache, "rb"))
    else:
        first_gate = pd.Timestamp(args.start) - pd.Timedelta(hours=CONTEXT + 24)
        frame = fetch(first_gate.strftime("%Y-%m-%d"),
                      (pd.Timestamp(args.end) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))

    local = frame.tz_convert("Europe/Brussels")
    load = local["totalload"].to_numpy(np.float32)
    baseline = local["dayaheadforecast"].to_numpy(np.float32)
    index = local.index

    predictor = TinyCastPredictor(prediction_length=HORIZON, checkpoint_path=args.checkpoint,
                                  device="cpu", freq="H", domain="Energy")

    gates = [i for i in range(CONTEXT, len(frame) - HORIZON)
             if index[i].hour == GATE_HOUR
             and args.start <= str(index[i + OFFSET].date()) <= args.end]
    print(f"{len(gates)} gates, {args.start} to {args.end}", flush=True)

    window = slice(OFFSET, OFFSET + LENGTH)
    days = []
    for n, gate in enumerate(gates):
        series = {"target": load[gate - CONTEXT:gate],
                  "start": pd.Period(index[gate - CONTEXT], freq="h")}
        forecast = next(iter(predictor.predict([series])))
        quantiles = {q: forecast.quantile(q)[window] for q in QUANTILES}

        actual = load[gate + OFFSET:gate + OFFSET + LENGTH]
        operator = baseline[gate + OFFSET:gate + OFFSET + LENGTH]
        if len(actual) < LENGTH or not np.isfinite(operator).all():
            continue

        median, low, high = quantiles[0.5], quantiles[0.1], quantiles[0.9]
        days.append(dict(
            issued=str(index[gate]), target_day=str(index[gate + OFFSET].date()),
            tinycast_mae=float(np.mean(np.abs(median - actual))),
            operator_mae=float(np.mean(np.abs(operator - actual))),
            mean_load=float(np.mean(actual)),
            coverage=float(np.mean((actual >= low) & (actual <= high)) * 100),
            width=float(np.mean(high - low)),
            context=load[gate - 72:gate].tolist(),
            unseen=load[gate:gate + OFFSET].tolist(),
            actual=actual.tolist(), operator=operator.tolist(),
            **{f"q{int(q * 100)}": quantiles[q].tolist() for q in QUANTILES}))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(gates)}", flush=True)

    pickle.dump(days, open(args.raw, "wb"))
    table = pd.DataFrame([{k: v for k, v in day.items() if not isinstance(v, list)}
                          for day in days])
    table["ratio"] = table.tinycast_mae / table.operator_mae
    table.to_csv(args.out, index=False)

    print(f"\ndays                {len(table)}")
    print(f"TinyCast median MAE {table.tinycast_mae.median():7.0f} MW")
    print(f"operator median MAE {table.operator_mae.median():7.0f} MW")
    print(f"TinyCast closer on  {100 * (table.ratio < 1).mean():7.0f} % of days")
    print(f"80% band coverage   {table.coverage.median():7.0f} % (median day)")
    print("wrote", args.out, "and", args.raw)


if __name__ == "__main__":
    main()

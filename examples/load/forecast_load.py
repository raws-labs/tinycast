"""Zero-shot 48-hour forecast of Belgian electricity load.

Downloads recent measured load from the Elia open data platform, forecasts the
next 48 hours from history alone, and compares against the transmission system
operator's own day-ahead forecast. TinyCast sees only the past of the series and
uses no weather data, no calendar and no covariates.

    python forecast_load.py --checkpoint model.safetensors --out load.png
"""
from __future__ import annotations

import argparse
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tinycast import TinyCastPredictor

API = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods001/records"
CONTEXT, HORIZON = 2048, 48


def fetch(days: int) -> pd.DataFrame:
    """Measured load and day-ahead forecast for Belgium, hourly, in MW."""
    start = (pd.Timestamp.utcnow().floor("h") - pd.Timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    rows, offset = [], 0
    while True:
        query = urllib.parse.urlencode({
            "where": f"datetime >= date'{start}'",
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
        if offset >= 9900:          # API paging ceiling
            break

    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame = frame.set_index("datetime").sort_index()
    return frame.resample("1h").mean().dropna(subset=["totalload"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="solar.png")
    parser.add_argument("--days", type=int, default=100)
    args = parser.parse_args()

    frame = fetch(args.days)
    if len(frame) < CONTEXT + HORIZON:
        raise SystemExit(f"need {CONTEXT + HORIZON} hours, have {len(frame)}")

    split = len(frame) - HORIZON
    context = frame["totalload"].to_numpy(np.float32)[split - CONTEXT:split]
    actual = frame["totalload"].to_numpy(np.float32)[split:]
    baseline = frame["dayaheadforecast"].to_numpy(np.float32)[split:]
    issued = frame.index[split]

    predictor = TinyCastPredictor(prediction_length=HORIZON, checkpoint_path=args.checkpoint,
                                  device="cpu", freq="H", domain="Energy")
    start = pd.Period(frame.index[split - CONTEXT], freq="h")
    forecast = next(iter(predictor.predict([{"target": context, "start": start}])))
    low, median, high = (forecast.quantile(q) for q in (0.1, 0.5, 0.9))

    mae = np.mean(np.abs(median - actual))
    inside = np.mean((actual >= low) & (actual <= high)) * 100
    print(f"issued           {issued:%Y-%m-%d %H:%M} UTC")
    print(f"TinyCast MAE     {mae:8.1f} MW")
    print(f"inside 10-90     {inside:8.0f} %")
    if np.isfinite(baseline).all():
        print(f"day-ahead MAE    {np.mean(np.abs(baseline - actual)):8.1f} MW")

    history = 96
    past = np.arange(-history, 0)
    ahead = np.arange(HORIZON)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(past, frame["totalload"].to_numpy()[split - history:split], color="#8b97a1", lw=1.3, label="measured load")
    axis.fill_between(ahead, low, high, color="#1a9c66", alpha=0.20, label="TinyCast 10-90%")
    axis.plot(ahead, median, color="#1a9c66", lw=2.2, label="TinyCast median")
    if np.isfinite(baseline).all():
        axis.plot(ahead, baseline, color="#c2703d", lw=1.6, label="grid operator day-ahead")
    axis.plot(ahead, actual, color="#1a2531", lw=1.8, ls="--", label="what happened")
    axis.axvline(0, color="#c8ced3", lw=1)
    axis.set_xlabel("hours from forecast time")
    axis.set_ylabel("electricity load  MW")
    axis.set_title(f"Belgian electricity load, 48 hours ahead, issued {issued:%d %B %Y %H:%M} UTC",
                   loc="left", fontsize=13, fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper left", fontsize=9, framealpha=0.9)
    figure.tight_layout()
    figure.savefig(args.out, dpi=170, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

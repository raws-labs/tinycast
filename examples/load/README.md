# Electricity load, zero-shot

Forecasts Belgian electricity load 48 hours ahead and compares against the
transmission system operator's own day-ahead forecast.

TinyCast sees only the past of the series: no weather data, no calendar, no
covariates, and no fitting to this series. The operator's forecast is an
operational product built with weather models and load-specific methods.

## Run

```bash
pip install tinycast matplotlib pandas
python forecast_load.py --checkpoint model.safetensors --out load.png
```

Data comes from the [Elia open data platform](https://opendata.elia.be)
(dataset `ods001`, measured total load and day-ahead forecast for Belgium,
resampled to hourly). No account or key is required. Each run downloads the
most recent data, so the forecast window moves with the calendar and the
numbers below will not reproduce exactly.

## Result

`forecast_load.py` makes a single forecast from the most recent data, so its numbers
move with the calendar. `backtest.py` runs the matched year-long comparison behind the
figures below.

```bash
python backtest.py --checkpoint model.safetensors \
    --start 2025-08-01 --end 2026-07-31 --out backtest.csv
```

The operator publishes its day-ahead forecast at 08:45 local time, covering the whole
following calendar day. Each TinyCast forecast is issued from history up to 09:00 local
and scored on the same 24 hours, so neither forecaster sees the target day.

Over the 365 days from 1 August 2025 to 31 July 2026:

| | median MAE | share of days closer |
|---|---|---|
| TinyCast | 256 MW | 45% |
| Grid operator, day-ahead | 245 MW | 55% |

That is a median gap of 4%, or 2.83% of daily mean demand against 2.66%. The 10-90%
interval contained the outcome 84% of the time across all 8,760 hours, and every decile
falls within 5 percentage points of its nominal rate.

The largest single-day gap runs the other way: on 15 August 2025, a public holiday,
TinyCast was 1,154 MW out against the operator's 170 MW. A yearly cycle sits outside the
2,048-step context, which at hourly resolution reaches back about 85 days.

Load is a favourable case for a univariate model: its structure is daily, weekly and
seasonal, and therefore present in the history the model reads. Solar generation is not.
Repeating this against Elia's solar dataset (`ods032`) gives a median error of 536 MW
against the operator's 191 MW, because the driver there is cloud cover, which no amount
of past output reveals.

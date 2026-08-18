"""Seasonal discretization scale factor.

Maps a pandas frequency string to the model's seasonal scale factor
``s = base_seasonality / (samples per natural period)``, i.e. how many
context samples fall in one canonical seasonal cycle at that sampling rate.
"""
from __future__ import annotations

from typing import Optional

# Cycles per canonical day: the unit every scale factor is expressed against.
# ``tinycast.losses`` reads this so the seasonal lag the committing term folds
# at and the scale factor the model is conditioned on cannot drift apart.
BASE_SEASONALITY = 24.0


def seasonal_scale_factor(freq: str, domain: Optional[str] = None) -> float:
    """Seasonal scale factor for a pandas frequency string."""
    has_weekly = domain in ["Transport", "Healthcare", "Sales"]

    if freq == "4S":
        factor = BASE_SEASONALITY / (3600.0 / 4)
    elif freq == "10S":
        factor = BASE_SEASONALITY / 360
    elif freq == "T":
        factor = BASE_SEASONALITY / (24.0 * 60)
    elif freq[-1] == "T":
        n_min = int(freq[:-1])
        factor = BASE_SEASONALITY / (24 * 60 / n_min)
    elif freq == "H":
        factor = BASE_SEASONALITY / 24
    elif freq == "6H":
        factor = BASE_SEASONALITY / 4
    elif freq == "D":
        factor = BASE_SEASONALITY / 7 if has_weekly else BASE_SEASONALITY / 365
    elif freq[-1] == "D" and "WED" not in freq:
        n = int(freq[:-1])
        factor = BASE_SEASONALITY / 7 if has_weekly else BASE_SEASONALITY / 365
        factor *= n
    elif freq == "W" or "W-" in freq:
        factor = BASE_SEASONALITY / (365.0 / 7)
    elif freq == "M" or "M-" in freq or freq == "MS":
        factor = BASE_SEASONALITY / 12
    elif "Q" in freq:
        factor = BASE_SEASONALITY / 4.0
    elif "A" in freq:
        factor = BASE_SEASONALITY / 4.0
    else:
        raise NotImplementedError(
            f"{freq} not implemented. Add {freq} option to seasonal_scale_factor."
        )
    return factor

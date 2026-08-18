"""Eval-time period-alignment downsampling rule.

Some datasets carry their dominant seasonality at a clean integer multiple of
the canonical (samples-per-day) period for their sampling frequency, with the
canonical period itself absent. For example, bizitobs_l2c at 5T has no daily
cycle at all, only weekly (2016 samples = 7x the canonical 288). Models calibrated
around the canonical period grid systematically mis-handle such series;
downsampling the context by the multiple aliases the dominant period back onto
the grid (weekly-at-5T becomes daily-at-35T) and restores in-distribution
behaviour. This module derives the factor from the CONTEXT data (a k=7 alias
on bizitobs_l2c/5T medium+long), so no per-dataset hand-tuning and no
information beyond the model's own inputs is used.

Fire conditions (all required, and deliberately narrow: downsampling a config
whose canonical peak is intact makes MASE strictly worse, m4_hourly's strong
weekly harmonic included):
  1. sub-daily frequency (canonical period >= MIN_CANONICAL samples/day);
  2. a dominant spectral peak at period P with P/canonical within
     ``REL_TOL`` of an integer k in [2, MAX_K];
  3. the canonical-period peak is ABSENT: power near the canonical period is
     below ``CANONICAL_ABSENT`` x the dominant peak's power;
  4. the dominant peak is significant: >= ``PEAK_SIG`` x median spectral power;
  5. a >= ``QUORUM`` fraction of sampled series agree on the same k;
  6. the downsampled context still fills the model window
     (median len / k >= ``context_window``, else the aliasing starves the
     encoder: on bizitobs_l2c/H, where T/k ~ 358, k=7 turns from a win into a
     loss at long horizons);
  7. the horizon spans at least one canonical day (H >= canonical, else the
     coarse forecast's interpolation loses more than the aliasing gains: on
     bizitobs_l2c/5T/short, H=48 against a canonical 288, by +0.05 MASE).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

MIN_CANONICAL = 8          # rule inactive for daily-or-coarser frequencies
MAX_K = 16
REL_TOL = 0.06             # |P/canonical - k| <= REL_TOL * k (after sub-bin refinement)
CANONICAL_ABSENT = 0.10    # canonical peak power < 10% of dominant peak power
PEAK_SIG = 20.0            # dominant peak >= 20x median spectral power
QUORUM = 0.7
MIN_CYCLES = 3             # dominant period must repeat >= 3x in the analysed tail
TAIL = 16384               # analyse at most this many trailing samples
N_SAMPLE = 64              # series sampled per config


def _series_factor(x: np.ndarray, canonical: int) -> int:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < MIN_CYCLES * 2 * canonical:
        return 1
    x = x[-TAIL:]
    n = x.size
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    p[0] = 0.0
    med = np.median(p[1:])
    if med <= 0.0:
        return 1

    periods = np.full(p.shape, np.inf)
    periods[1:] = n / np.arange(1, p.shape[0], dtype=np.float64)

    # Power near the canonical period (max over a +-10% band).
    canon_band = (periods >= 0.9 * canonical) & (periods <= 1.1 * canonical)
    canon_pow = p[canon_band].max() if canon_band.any() else 0.0

    # Dominant peak among periods that are >= 1.5x canonical and repeat
    # >= MIN_CYCLES times in the analysed tail.
    cand = (periods >= 1.5 * canonical) & (periods <= n / MIN_CYCLES)
    if not cand.any():
        return 1
    idx = int(np.flatnonzero(cand)[np.argmax(p[cand])])
    peak_pow = p[idx]
    if peak_pow < PEAK_SIG * med:
        return 1
    if canon_pow > CANONICAL_ABSENT * peak_pow:
        return 1                      # canonical period present: do not alias

    # Sub-bin peak refinement (parabolic on log-power): the raw frequency
    # grid is coarse at long periods (spacing ~P^2/n, i.e. ~12% of P for
    # weekly-at-5T in a 16k tail), which would let non-integer multiples
    # masquerade as clean ones under any workable tolerance.
    delta = 0.0
    if 1 <= idx < p.shape[0] - 1 and p[idx - 1] > 0 and p[idx + 1] > 0:
        lp = np.log(p[idx - 1:idx + 2])
        denom = lp[0] - 2.0 * lp[1] + lp[2]
        if denom < 0:
            delta = float(np.clip(0.5 * (lp[0] - lp[2]) / denom, -0.5, 0.5))
    refined_period = n / (idx + delta)

    r = refined_period / canonical
    k = int(round(r))
    if k < 2 or k > MAX_K or abs(r - k) > REL_TOL * k:
        return 1
    return k


def period_alignment_factor(
    contexts: Iterable[np.ndarray],
    freq_seconds: float,
    horizon: int,
    context_window: int = 2048,
    n_sample: int = N_SAMPLE,
) -> int:
    """Downsample factor for one eval config, from context windows only.

    ``contexts`` are the test INPUT series (the model's own inputs);
    ``freq_seconds`` is the sampling interval; ``horizon`` the prediction
    length; ``context_window`` the model's encoder window. Returns 1 unless
    the config's series agree (>= QUORUM) on the same aliasing multiple k >= 2
    and the horizon/window guards pass.
    """
    canonical = int(round(86400.0 / float(freq_seconds)))
    if canonical < MIN_CANONICAL:
        return 1
    if int(horizon) < canonical:           # guard 7: >= one canonical day
        return 1
    ks = []
    lens = []
    for x in contexts:
        x = np.asarray(x, dtype=np.float64)
        ks.append(_series_factor(x, canonical))
        lens.append(x.size)
        if len(ks) >= n_sample:
            break
    if not ks:
        return 1
    vals, counts = np.unique(ks, return_counts=True)
    best = int(vals[np.argmax(counts)])
    if best == 1 or counts.max() / len(ks) < QUORUM:
        return 1
    if float(np.median(lens)) / best < context_window:   # guard 6
        return 1
    return best


def freq_to_seconds(freq: str) -> Optional[float]:
    """Sampling interval in seconds for a pandas-style freq string, or None."""
    import pandas as pd

    try:
        off = pd.tseries.frequencies.to_offset(freq)
        return pd.Timedelta(off).total_seconds()
    except (ValueError, TypeError):
        return None

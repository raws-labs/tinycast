"""Synthetic series generators: Gaussian process, spike trains, TSI.

These are the three families in TinyCast's synthetic pretraining shards. They
follow the recipe published by Reverso (arXiv 2602.17634, Appendix A),
implemented from its Algorithms 1 to 3 and Table 8:

* :func:`generate_gp` is Algorithm 1: a Gaussian process over a 38-entry kernel
  bank (Constant; Linear with sigma in {0, 1, 10}; RBF with l in {0.1, 1, 10};
  RationalQuadratic with alpha in {0.1, 1, 10}; Matern with nu in
  {0.5, 1.5, 2.5} crossed with l in {0.1, 1, 10}; Periodic over a 19-period set
  normalized by series length), with J ~ U{1,5} kernels composed by random sum
  or product, and a mean function that is a linear trend with probability 1/2.
* :func:`generate_spikes` is Algorithm 2: trapezoid pulse trains (a quarter of
  the pulse ramping up, half flat, the remainder ramping down) tiled at a fixed
  period over a baseline, in an upward or downward variant, plus white noise.
* :func:`generate_tsi` is Algorithm 3, the trend-seasonal-impulse process: an
  optional trend, K ~ U{1,3} seasonal components (sine, sawtooth or square),
  noise, sparse outliers and level shifts.

All three return float32 with no normalization applied. The corpus builder
z-normalizes per series on write, which is why the absolute scale of any
parameter below does not matter and the shape does.

DOCUMENTED ASSUMPTIONS. The source paper leaves the following symbolic, and
neither Kairos (arXiv 2509.25826) nor Chronos-2 (arXiv 2510.15821), which the
respective algorithms defer to, publishes them. We record the choices we made
rather than presenting the families as an exact replication:

  A1  Algorithm 1 trend units. The slope m ~ U[-0.01, 0.01] is applied per
      index step (mu_t = m*t + c with t = 0..L-1), not per unit of the [0,1]
      kernel grid. On the [0,1] grid the sampled trend would be at most 1% of
      the GP's unit scale, which makes the probability-1/2 branch pointless.
      Index units give trend-dominated series about half the time, which is a
      realistic class, and the per-series z-normalization on write removes the
      magnitude difference.
  A2  Algorithm 2 numeric ranges: baseline U[-1, 1], period U{16, L//8}, pulse
      width U{4, p}, amplitude U[0.5, 3], noise sigma U[0.01, 0.3]. Under
      z-normalization the levels are immaterial; shape, sparsity and period are
      what the model sees.
  A3  Algorithm 3 probabilities and ranges: P_trend = 0.5, P_seasonal = 0.8,
      P_noise = 0.8, P_outlier = 0.2, P_shift = 0.2; trend types linear,
      exponential, quadratic and piecewise linear; periods taken from the
      Algorithm 1 period set in index steps, filtered to at most L/2; amplitude
      U[0.5, 2]; noise sigma U[0.05, 0.5], normal or Laplace; U{1, L//100}
      outliers at +/- U[3, 8] standard deviations; U{1,3} level shifts of
      magnitude +/- U[0.5, 3].
  A4  The family mix. The paper gives a total series count but not the
      proportions, and describes spikes and TSI as additions to a GP majority.
      :mod:`tinycast.corpus` uses 70/15/15.

DEVICE. :func:`generate_gp` draws from numpy on the CPU path and from torch on
the CUDA path, so the same seed produces different series on the two branches.
The published shards came from the CUDA branch, and :mod:`tinycast.corpus`
refuses anything else for that reason. The CPU branch is kept here because it
is useful at small scale and because spikes and TSI are numpy either way.
"""
from __future__ import annotations

import math

import numpy as np

# Algorithm 1 / Table 8 period set. Table 8 specifies the periodic kernel's p
# as a fraction of the series length, so these enter the bank divided by L.
PERIODS = (24, 48, 96, 168, 336, 672, 7, 14, 30, 60,
           365, 730, 4, 26, 52, 6, 12, 40, 10)

# Jitter added to the diagonal before factorization, and the larger retry value
# the CPU path falls back to. The CUDA path does not retry: see generate_gp.
GP_JITTER = 1e-6
GP_JITTER_RETRY = 1e-4


def kernel_bank(length: int) -> list[tuple[str, tuple]]:
    """The 38 Table-8 kernels as (tag, params) pairs, for a series length.

    Only the periodic entries depend on ``length``, and they depend on it
    because Table 8 normalizes the period by the series length.
    """
    bank: list[tuple[str, tuple]] = [("const", (1.0,))]
    bank += [("linear", (s,)) for s in (0.0, 1.0, 10.0)]
    bank += [("rbf", (l,)) for l in (0.1, 1.0, 10.0)]
    bank += [("rq", (a,)) for a in (0.1, 1.0, 10.0)]
    bank += [("matern", (nu, l)) for nu in (0.5, 1.5, 2.5) for l in (0.1, 1.0, 10.0)]
    bank += [("periodic", (p / length,)) for p in PERIODS]
    return bank


def _matern(d: np.ndarray, nu: float, l: float) -> np.ndarray:
    """Matern closed forms for nu in {0.5, 1.5, 2.5}, so no Bessel call."""
    if nu == 0.5:
        return np.exp(-d / l)
    if nu == 1.5:
        a = math.sqrt(3.0) * d / l
        return (1.0 + a) * np.exp(-a)
    if nu == 2.5:
        a = math.sqrt(5.0) * d / l
        return (1.0 + a + a * a / 3.0) * np.exp(-a)
    raise ValueError(f"unsupported Matern nu={nu}")


def _dense_kernel(tag: str, params: tuple, t: np.ndarray) -> np.ndarray:
    d = np.abs(t[:, None] - t[None, :])
    if tag == "const":
        return np.full_like(d, params[0])
    if tag == "linear":            # sigma^2 + x.x'
        return params[0] ** 2 + np.outer(t, t)
    if tag == "rbf":
        return np.exp(-(d ** 2) / (2.0 * params[0] ** 2))
    if tag == "rq":                # Table 8's form carries no lengthscale (l=1)
        return (1.0 + d ** 2 / (2.0 * params[0])) ** (-params[0])
    if tag == "matern":
        return _matern(d, *params)
    if tag == "periodic":          # Table 8: exp(-2 sin^2(pi d / p)), l=1
        return np.exp(-2.0 * np.sin(np.pi * d / params[0]) ** 2)
    raise ValueError(tag)


def generate_gp(
    num_series: int,
    length: int,
    seed: int | None = None,
    max_kernels: int = 5,
    device: str | None = None,
    batch: int = 32,
) -> np.ndarray:
    """Algorithm 1: GP samples with composed Table-8 kernels and a trend mean.

    Sampling is by dense jittered Cholesky. A circulant or FFT sampler is not
    an option here because the composed bank is not stationary: the linear and
    periodic entries break both stationarity and circularity.

    ``device='cuda'`` batches the covariance construction and the factorization
    on the GPU, which is what production volume at length 4096 needs. Both
    paths work in float64; float32 loses the high-frequency modes of the
    periodic entries outright. The CUDA path uses ``cholesky_ex`` and writes
    NaN for any row whose covariance failed to factorize, leaving the caller to
    drop it, while the CPU path retries once at a larger jitter and raises if
    that also fails.

    ``device=None`` selects the numpy path; any device string selects the torch
    path on that device, so ``'cpu'`` runs the batched code on the CPU and is a
    third result rather than a cheap stand-in for either. The paths do not agree
    row by row. ``batch`` is part of the data on the torch path rather than a
    memory knob: see :mod:`tinycast.corpus`.

    Returns float32 of shape ``(num_series, length)``, unnormalized.
    """
    rng = np.random.default_rng(seed)
    bank = kernel_bank(length)
    t = np.linspace(0.0, 1.0, length)
    idx = np.arange(length, dtype=np.float64)
    out = np.empty((num_series, length), dtype=np.float32)

    def compose_indices():
        n_k = int(rng.integers(1, max_kernels + 1))
        picks = [int(rng.integers(0, len(bank))) for _ in range(n_k)]
        ops = [int(rng.integers(0, 2)) for _ in range(n_k - 1)]
        return picks, ops

    def mean_fn():
        # ASSUMPTION A1: the trend is in index units.
        if rng.uniform() < 0.5:
            m = rng.uniform(-0.01, 0.01)
            c = rng.uniform(-0.1, 0.1)
            return m * idx + c
        return np.zeros(length)

    if device is None:
        for i in range(num_series):
            picks, ops = compose_indices()
            K = _dense_kernel(*bank[picks[0]], t)
            for op, pi in zip(ops, picks[1:]):
                Kn = _dense_kernel(*bank[pi], t)
                K = K + Kn if op == 0 else K * Kn
            K = K + GP_JITTER * np.eye(length)
            try:
                Lc = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += (GP_JITTER_RETRY - GP_JITTER) * np.eye(length)
                Lc = np.linalg.cholesky(K)
            out[i] = (Lc @ rng.standard_normal(length) + mean_fn()).astype(np.float32)
        return out

    import torch

    dev = torch.device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(int(seed) if seed is not None else 0)
    tt = torch.linspace(0.0, 1.0, length, device=dev, dtype=torch.float64)
    dd = (tt[:, None] - tt[None, :]).abs()
    d2 = dd * dd
    outer = tt[:, None] * tt[None, :]
    eye = torch.eye(length, device=dev, dtype=torch.float64)

    def _k(tag: str, params: tuple) -> "torch.Tensor":
        if tag == "const":
            return torch.full_like(dd, params[0])
        if tag == "linear":
            return params[0] ** 2 + outer
        if tag == "rbf":
            return torch.exp(-d2 / (2.0 * params[0] ** 2))
        if tag == "rq":
            return (1.0 + d2 / (2.0 * params[0])).pow(-params[0])
        if tag == "matern":
            nu, l = params
            if nu == 0.5:
                return torch.exp(-dd / l)
            if nu == 1.5:
                a = math.sqrt(3.0) * dd / l
                return (1.0 + a) * torch.exp(-a)
            a = math.sqrt(5.0) * dd / l
            return (1.0 + a + a * a / 3.0) * torch.exp(-a)
        return torch.exp(-2.0 * torch.sin(math.pi * dd / params[0]) ** 2)

    Ks = torch.empty((batch, length, length), device=dev, dtype=torch.float64)
    done = 0
    while done < num_series:
        b = min(batch, num_series - done)
        means = np.empty((b, length), dtype=np.float64)
        for i in range(b):
            picks, ops = compose_indices()
            K = _k(*bank[picks[0]])
            for op, pi in zip(ops, picks[1:]):
                Kn = _k(*bank[pi])
                K = K + Kn if op == 0 else K * Kn
            Ks[i] = K
            means[i] = mean_fn()
        Kb = Ks[:b] + GP_JITTER * eye
        Lc, info = torch.linalg.cholesky_ex(Kb)
        z = torch.randn(b, length, 1, generator=g, device=dev, dtype=torch.float64)
        s = torch.matmul(Lc, z).squeeze(-1)
        bad = info > 0
        if bool(bad.any()):
            s[bad] = float("nan")
        out[done:done + b] = (s.cpu().numpy() + means).astype(np.float32)
        done += b
    return out


def generate_spikes(
    num_series: int, length: int, seed: int | None = None,
) -> np.ndarray:
    """Algorithm 2: trapezoid pulse trains over a baseline, plus white noise.

    The numeric ranges are ASSUMPTION A2. Levels wash out under the per-series
    z-normalization the corpus applies on write, so what this family
    contributes is the pulse shape and its spacing.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((num_series, length), dtype=np.float32)
    for i in range(num_series):
        b = rng.uniform(-1.0, 1.0)
        p = int(rng.integers(16, max(17, length // 8) + 1))
        w = int(rng.integers(4, p + 1))
        a = rng.uniform(0.5, 3.0)
        sigma = rng.uniform(0.01, 0.3)
        sign = -1.0 if rng.integers(0, 2) == 0 else 1.0   # downward or upward
        up = w // 4
        flat = w // 2
        down = w - up - flat
        pulse = np.concatenate([
            np.linspace(0.0, a, max(up, 1)),
            np.full(max(flat, 1), a),
            np.linspace(a, 0.0, max(down, 1)),
        ])[:w]
        x = np.full(length, b)
        for start in range(0, length, p):
            seg = min(w, length - start)
            x[start:start + seg] += sign * pulse[:seg]
        x += rng.normal(0.0, sigma, length)
        out[i] = x.astype(np.float32)
    return out


def generate_tsi(
    num_series: int, length: int, seed: int | None = None,
    p_trend: float = 0.5, p_seas: float = 0.8, p_noise: float = 0.8,
    p_out: float = 0.2, p_shift: float = 0.2,
) -> np.ndarray:
    """Algorithm 3: trend, seasonal components, noise, outliers, level shifts.

    The probabilities and ranges are ASSUMPTION A3: the source paper states the
    structure and defers the numbers, and the work it defers to does not
    publish them either.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64)
    tn = t / max(length - 1, 1)
    periods = [p for p in PERIODS if p <= length // 2]
    out = np.empty((num_series, length), dtype=np.float32)
    for i in range(num_series):
        x = np.zeros(length)
        if rng.uniform() < p_trend:
            kind = rng.integers(0, 4)
            if kind == 0:                                  # linear
                x += rng.uniform(-2.0, 2.0) * tn
            elif kind == 1:                                # exponential
                x += np.exp(rng.uniform(0.5, 2.0) * tn) - 1.0
            elif kind == 2:                                # quadratic
                x += rng.uniform(-2.0, 2.0) * tn ** 2
            else:                                          # piecewise linear
                cp = int(rng.integers(1, length - 1))
                s1, s2 = rng.uniform(-2.0, 2.0, 2)
                x[:cp] += s1 * tn[:cp]
                x[cp:] += s1 * tn[cp] + s2 * (tn[cp:] - tn[cp])
        if rng.uniform() < p_seas and periods:
            n_comp = int(rng.integers(1, 4))
            chosen = rng.choice(periods, size=min(n_comp, len(periods)), replace=False)
            for p in chosen:
                amp = rng.uniform(0.5, 2.0)
                phi = rng.uniform(0.0, 2.0 * np.pi)
                arg = 2.0 * np.pi * t / p + phi
                wave = rng.integers(0, 3)
                if wave == 0:
                    x += amp * np.sin(arg)
                elif wave == 1:                            # sawtooth
                    x += amp * (2.0 * ((arg / (2.0 * np.pi)) % 1.0) - 1.0)
                else:                                      # square
                    x += amp * np.sign(np.sin(arg))
        if rng.uniform() < p_noise:
            sigma = rng.uniform(0.05, 0.5)
            if rng.integers(0, 2) == 0:
                x += rng.normal(0.0, sigma, length)
            else:
                x += rng.laplace(0.0, sigma / math.sqrt(2.0), length)
        if rng.uniform() < p_out:
            n = int(rng.integers(1, max(2, length // 100)))
            pos = rng.integers(0, length, n)
            mag = rng.uniform(3.0, 8.0, n) * max(x.std(), 0.1)
            x[pos] += mag * rng.choice([-1.0, 1.0], n)
        if rng.uniform() < p_shift:
            n = int(rng.integers(1, 4))
            for _ in range(n):
                pos = int(rng.integers(1, length))
                x[pos:] += rng.uniform(0.5, 3.0) * (1.0 if rng.integers(0, 2) else -1.0)
        out[i] = x.astype(np.float32)
    return out

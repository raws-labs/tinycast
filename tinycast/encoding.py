"""Shared structural positional-encoding helpers (phase + bounded recency).

A normalized-periodogram detector supplies top-K periods per sample. Those
periods drive a structural positional encoding (sin/cos of phase) shared
between context tokens and decoder horizon queries, plus a bounded
recency/trend basis. Also holds the fp32 RMSNorm helper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _phase_encoding(
    positions: torch.Tensor, periods: torch.Tensor,
    n_harmonics: int = 1,
) -> torch.Tensor:
    """Compute sin/cos phase encoding for each position under each period.

    For each detected period ``p_k``, emit Fourier-series channels up to
    ``n_harmonics``: at harmonic m, the channels are sin(2π·m·t/p_k) and
    cos(2π·m·t/p_k). The fundamental (m=1) is the base behavior; m=2,3,...
    let the model represent non-sinusoidal periodic shapes (square-wave
    traffic, sawtooth load) that the fundamental alone cannot.

    Output channel ordering, per (period, harmonic):
        [sin(1·φ_1), cos(1·φ_1), ..., sin(H·φ_1), cos(H·φ_1),
         sin(1·φ_2), cos(1·φ_2), ..., sin(H·φ_K), cos(H·φ_K)]

    Args:
        positions:   (B, T) int tensor of absolute positions.
        periods:     (B, K) int tensor of per-sample detected periods. Zero
                     means "rejected by significance test" → all harmonics
                     of that period come back zeroed.
        n_harmonics: number of Fourier harmonics per period (default 1).
    Returns:
        (B, T, 2·K·n_harmonics) fp32 tensor.
    """
    B, T = positions.shape
    K = periods.shape[1]
    H = int(n_harmonics)
    if H < 1:
        raise ValueError(f"n_harmonics must be >= 1; got {H}")
    valid = (periods > 0).view(B, 1, K).float()                   # (B,1,K)
    p_safe = periods.clamp(min=1).view(B, 1, K).float()
    pos = positions.view(B, T, 1).float()
    phase_base = 2.0 * math.pi * pos / p_safe                     # (B,T,K)
    # Build (B,T,K,H,2): for each (period k, harmonic m), [sin(m·φ_k), cos(m·φ_k)]
    multipliers = torch.arange(1, H + 1, device=positions.device, dtype=phase_base.dtype)
    phase_m = phase_base.unsqueeze(-1) * multipliers              # (B,T,K,H)
    sin_m = torch.sin(phase_m) * valid.unsqueeze(-1)              # (B,T,K,H)
    cos_m = torch.cos(phase_m) * valid.unsqueeze(-1)
    pair = torch.stack([sin_m, cos_m], dim=-1)                    # (B,T,K,H,2)
    return pair.reshape(B, T, K * H * 2)


# Number of recency/trend channels added by the bounded-basis extension.
N_RECENCY_CHANNELS = 5


def _recency_encoding(
    positions: torch.Tensor, L: int,
) -> torch.Tensor:
    """Bounded recency/trend channels for the shared positional encoding.

    "Now" is anchored at position ``L-1`` (end of context). ``Δ = (t - (L-1)) / L``
    is signed: negative for past, zero at "now", positive for future. All
    channels are bounded so they're safe to evaluate at arbitrary future
    horizons (the parameterized-query path goes well beyond training H).

    Channels (5):
        rec_lin:  Δ                            (signed linear distance from now)
        rec_log:  sign(Δ) · log1p(|Δ|)/log(2)  (signed log-compressed distance)
        rec_e05:  exp(-0.5 · |Δ|)              (long-memory decay)
        rec_e2:   exp(-2.0 · |Δ|)              (medium-memory decay)
        rec_e8:   exp(-8.0 · |Δ|)              (short-memory / locality kernel)
    """
    B, T = positions.shape
    delta = (positions.float() - float(L - 1)) / float(L)         # (B, T)
    abs_d = delta.abs()
    rec_lin = delta
    rec_log = torch.sign(delta) * torch.log1p(abs_d) / math.log(2.0)
    rec_e05 = torch.exp(-0.5 * abs_d)
    rec_e2 = torch.exp(-2.0 * abs_d)
    rec_e8 = torch.exp(-8.0 * abs_d)
    return torch.stack([rec_lin, rec_log, rec_e05, rec_e2, rec_e8], dim=-1)


def _positional_encoding(
    positions: torch.Tensor, periods: torch.Tensor, L: int,
    n_harmonics: int = 1,
) -> torch.Tensor:
    """Full shared positional encoding (phase + bounded recency basis).

    Returns (B, T, 2·K·n_harmonics + 5).
    """
    return torch.cat(
        [
            _phase_encoding(positions, periods, n_harmonics=n_harmonics),
            _recency_encoding(positions, L),
        ],
        dim=-1,
    )


def _norm_fp32(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply RMSNorm in fp32 and cast the result back to the input's dtype.

    Under bf16-mixed AMP, RMSNorm receives bf16 input but holds fp32 weights.
    PyTorch's fused RMSNorm kernel falls back to a slow non-fused path on
    dtype mismatch. Explicit fp32 promotion matches the weight dtype and lets
    the fused kernel engage. This matters most under compile, where the unfused
    dispatch breaks the graph and prevents downstream fusions.
    """
    if x.dtype == torch.float32:
        return norm(x)
    with torch.amp.autocast(
        device_type=x.device.type if x.is_cuda else "cpu", enabled=False,
    ):
        return norm(x.float()).to(x.dtype)

"""Normalized-periodogram period detector.

A zero-parameter structural detector: it identifies the dominant seasonal
periods of each series via a significance-filtered normalized periodogram.
The dilated-conv encoder uses it to build its phase positional encoding.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def significant_periods(
    x: torch.Tensor,
    *,
    min_period: int = 2,
    max_period: int | None = None,
    top_k: int = 16,
    significance_alpha: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Identify candidate periods via the normalized periodogram.

    Score per frequency bin k:
        I_norm[k] = |X[k]|² / sum_k' |X[k']|²

    Under H_0 (white Gaussian noise), max(I_norm) follows an extreme-value
    distribution. Peaks are filtered by a Bonferroni-corrected significance
    threshold:

        t_α = ln(N_bins / α) / N_bins

    where N_bins is the number of valid frequency bins (data-determined,
    not a knob) and α is the significance level (default 0.05). Peaks below
    t_α are excluded by setting their score to -inf, so ``n_valid`` reflects
    only periods that pass.

    α is exposed for completeness but should normally stay at 0.05; smaller
    (e.g. 0.01) is stricter, larger (0.10) laxer. Set α=1.0 to disable
    filtering entirely.

    Returns ``(periods, scores, n_valid)``: integer periods (0 = rejected),
    per-slot scores, and the count of significant periods per sample.
    """
    B, L = x.shape
    device = x.device
    if max_period is None:
        max_period = L // 2

    with torch.amp.autocast(device_type=x.device.type if x.is_cuda else "cpu",
                            enabled=False):
        x_f = x.float()
        x_c = x_f - x_f.mean(dim=1, keepdim=True)
        n_fft = 1 << int(math.ceil(math.log2(max(2, L))))
        X = torch.fft.rfft(x_c, n=n_fft)
        power = (X * X.conj()).real
        power = power[:, 1:]   # skip DC
        n_bins = power.shape[1]
        total = power.sum(dim=1, keepdim=True).clamp(min=1e-12)
        I_norm = power / total

        k_lo = max(0, (n_fft // max_period) - 1)
        k_hi = min(n_bins - 1, max(0, (n_fft // max(2, min_period)) - 1))

        I_left = I_norm[:, :-2]
        I_mid = I_norm[:, 1:-1]
        I_right = I_norm[:, 2:]
        is_local_max = (I_mid > I_left) & (I_mid > I_right)
        is_local_max_padded = F.pad(is_local_max, (1, 1), value=False)

        valid = torch.zeros(n_bins, device=device, dtype=torch.bool)
        if k_hi > k_lo:
            valid[k_lo:k_hi + 1] = True
        is_peak = is_local_max_padded & valid.unsqueeze(0)

        neg_inf = torch.full_like(I_norm, float("-inf"))
        scored = torch.where(is_peak, I_norm, neg_inf)

        # Bonferroni-corrected significance threshold. N_bins = number of
        # frequency bins eligible to compete; approximated by the full
        # periodogram length (the local-max filter doesn't change the
        # multiple-testing count substantively).
        N_bins_eff = max(2, int(n_bins))
        alpha = max(min(float(significance_alpha), 1.0), 1e-12)
        sig_threshold = math.log(N_bins_eff / alpha) / N_bins_eff
        scored = torch.where(
            scored >= sig_threshold, scored, neg_inf,
        )

        scores, k_top = scored.topk(k=min(top_k, n_bins), dim=1)
        freq_bins = k_top + 1   # undo the DC skip
        # Round, don't truncate. The peak bin k for true period S has
        # k_exact = n_fft / S. Truncating ``n_fft // k`` flips period 7 <-> 6
        # for S=7 at L=2048 (depending on which side of the half-integer the
        # peak lands), and that 14% period error becomes a full-cycle phase
        # drift over a 48-step horizon.
        periods_raw = torch.round(n_fft / freq_bins.clamp(min=1).float()).long()
        finite = torch.isfinite(scores)
        n_valid = finite.sum(dim=1).long()
        periods = torch.where(finite, periods_raw, torch.zeros_like(periods_raw))

    return periods.long(), scores, n_valid

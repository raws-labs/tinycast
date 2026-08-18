"""Training objectives for TinyCast.

Two objectives supervise the model. The nine-quantile pinball loss trains the
whole predictive distribution; the gated committing term acts on the median
alone and is the paper's own addition. ``seasonal_copy_baseline`` builds the
reference forecast the committing term measures the median against.

All three work in the model's normalized output space. ``TinyCastBackbone.encode``
returns ``(y_norm, x_min, x_range)``; normalize the target with those same
statistics before calling anything here, so predictions, targets and the
seasonal copy sit on one scale.

Each function is self-contained and takes plain tensors, so it can be exercised
without a model or a training loop.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

# Cycles per canonical day, the constant the scale factor is expressed against:
# a sample's seasonal lag is BASE_SEASONALITY divided by its scale factor. It is
# defined once, in tinycast.scale, and re-exported here so the lag this module
# folds at and the factor the model is conditioned on cannot drift apart.
from .scale import BASE_SEASONALITY

# Weight of the committing term in the shipped recipe.
COMMIT_WEIGHT = 0.3

_Reduction = str


def _reduce(per_sample: torch.Tensor, reduction: _Reduction) -> torch.Tensor:
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(
        f"Unknown reduction: {reduction!r}; expected 'mean', 'sum' or 'none'."
    )


def _as_bh(target: torch.Tensor, name: str) -> torch.Tensor:
    """Accept ``(B, H)`` or ``(B, H, 1)`` and return ``(B, H)``."""
    if target.dim() == 3 and target.shape[-1] == 1:
        return target.squeeze(-1)
    if target.dim() != 2:
        raise ValueError(
            f"{name} must be (B, H) or (B, H, 1), got {tuple(target.shape)}."
        )
    return target


def _mask_like(mask: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(ref)
    mask = _as_bh(mask, "mask").to(ref.dtype)
    if mask.shape != ref.shape:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match "
            f"{tuple(ref.shape)}."
        )
    return mask


def pinball_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    quantiles: Union[Sequence[float], torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    *,
    reduction: _Reduction = "mean",
) -> torch.Tensor:
    """Quantile (pinball) loss over all quantile levels.

    Args:
        pred: ``(B, H, Q)`` normalized quantile forecasts, ordered to match
            ``quantiles``.
        target: ``(B, H)`` or ``(B, H, 1)`` normalized targets.
        quantiles: the ``Q`` levels, e.g. ``config.quantiles``.
        mask: ``(B, H)`` observedness, 1 for an observed target position and 0
            otherwise. ``None`` treats every position as observed.
        reduction: ``"mean"`` (default) averages over the batch, ``"sum"`` adds,
            ``"none"`` returns the ``(B,)`` per-sample losses.

    The per-sample loss divides by the observed count times ``Q``, so a sample
    with few observed positions is not down-weighted against a full one, and a
    sample with none contributes zero rather than a division by zero.

    A non-finite target position is replaced by the detached median forecast, so
    it contributes no loss and no gradient. Mask such positions out as well: the
    substitution is a guard, not the mechanism.
    """
    if pred.dim() != 3:
        raise ValueError(f"pred must be (B, H, Q), got {tuple(pred.shape)}.")
    target = _as_bh(target, "target")
    if pred.shape[:2] != target.shape:
        raise ValueError(
            f"pred {tuple(pred.shape)} and target {tuple(target.shape)} "
            "disagree on batch or horizon."
        )

    q = torch.as_tensor(quantiles, dtype=pred.dtype, device=pred.device)
    q = q.reshape(-1)
    n_q = pred.shape[-1]
    if q.numel() != n_q:
        raise ValueError(
            f"pred carries {n_q} quantile channels but {q.numel()} levels "
            "were given."
        )

    q_mid = n_q // 2
    target_b = target.unsqueeze(-1)
    fill = pred[..., q_mid: q_mid + 1].detach()
    target_safe = torch.where(torch.isfinite(target_b), target_b, fill)

    err = target_safe - pred
    q = q.view(1, 1, -1)
    loss = torch.maximum(q * err, (q - 1.0) * err)              # (B, H, Q)

    obs = _mask_like(mask, target)
    denom = obs.sum(dim=1).clamp(min=1.0) * float(n_q)
    per_sample = (loss * obs.unsqueeze(-1)).sum(dim=(1, 2)) / denom
    return _reduce(per_sample, reduction)


def seasonal_copy_baseline(
    context: torch.Tensor,
    horizon: int,
    scale_factor: Union[torch.Tensor, float],
    *,
    base_seasonality: float = BASE_SEASONALITY,
) -> torch.Tensor:
    """Repeat the last seasonal cycle of the context over the horizon.

    Args:
        context: ``(B, L)`` or ``(B, L, 1)`` context values, in whatever space
            the caller wants the copy in (raw or normalized).
        horizon: ``H``, how many steps to emit.
        scale_factor: per-sample or scalar seasonal scale factor ``s``, the
            value ``tinycast.scale.seasonal_scale_factor`` returns for the
            sample's frequency.
        base_seasonality: numerator of the lag, 24 by convention.

    The lag is ``round(base_seasonality / s)`` clipped to ``[2, L // 2]``, and
    position ``h`` of the copy is context position ``L - lag + (h mod lag)``.
    Returns ``(B, H)``.
    """
    if context.dim() == 3 and context.shape[-1] == 1:
        context = context.squeeze(-1)
    if context.dim() != 2:
        raise ValueError(
            f"context must be (B, L) or (B, L, 1), got {tuple(context.shape)}."
        )
    b, length = context.shape
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}.")
    if length < 4:
        raise ValueError(f"context is too short to fold a cycle: L={length}.")

    sf = torch.as_tensor(scale_factor, dtype=torch.float32, device=context.device)
    sf = sf.reshape(-1)
    if sf.numel() == 1:
        sf = sf.expand(b)
    elif sf.numel() != b:
        raise ValueError(
            f"scale_factor carries {sf.numel()} entries for a batch of {b}."
        )

    lag = (base_seasonality / sf.clamp(min=1e-3)).round().long()
    lag = lag.clamp(2, max(2, length // 2)).view(-1, 1)         # (B, 1)

    h = torch.arange(horizon, device=context.device).view(1, horizon)
    src = (length - lag + (h % lag)).clamp(0, length - 1)       # (B, H)
    return torch.gather(context, 1, src)


def committing_loss(
    median: torch.Tensor,
    target: torch.Tensor,
    seasonal_copy: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    weight: float = COMMIT_WEIGHT,
    gated: bool = True,
    reduction: _Reduction = "mean",
) -> torch.Tensor:
    """Gated committing term: penalize a median that hedges below the copy.

    Args:
        median: ``(B, H)`` normalized median forecast.
        target: ``(B, H)`` normalized targets.
        seasonal_copy: ``(B, H)`` the reference from
            ``seasonal_copy_baseline``, in the same space as ``median``.
        mask: ``(B, H)`` observedness; ``None`` treats every position as
            observed.
        weight: the multiplier the trainer applies, 0.3 in the shipped recipe.
        gated: apply the window-level gate. With ``False`` the hinge is scored
            on every window.
        reduction: as in :func:`pinball_loss`.

    The hinge ``relu(|median - target| - |copy - target|)`` is zero wherever the
    median is already at least as close as the copy, so the term stops as soon
    as the median reaches it. The gate multiplies the whole window by zero
    unless the copy beats the median summed over the observed positions, which
    keeps the term silent on windows the model already wins.

    Positions where any of the three inputs is non-finite contribute nothing,
    to the hinge and to the gate alike. The average is still taken over the
    observed count, so masking is what sets the denominator.
    """
    median = _as_bh(median, "median")
    target = _as_bh(target, "target")
    seasonal_copy = _as_bh(seasonal_copy, "seasonal_copy")
    if median.shape != target.shape or median.shape != seasonal_copy.shape:
        raise ValueError(
            f"median {tuple(median.shape)}, target {tuple(target.shape)} and "
            f"seasonal_copy {tuple(seasonal_copy.shape)} must agree."
        )

    obs = _mask_like(mask, target)
    med_err = (median - target).abs()
    copy_err = (seasonal_copy - target).abs()

    finite = (
        torch.isfinite(med_err) & torch.isfinite(copy_err) & torch.isfinite(target)
    ).to(obs.dtype)
    scored = obs * finite

    hinge = torch.relu(med_err - copy_err)
    hinge = torch.where(torch.isfinite(hinge), hinge, torch.zeros_like(hinge))
    denom = obs.sum(dim=1).clamp(min=1.0)
    per_sample = (hinge * scored).sum(dim=1) / denom

    if gated:
        med_total = (torch.nan_to_num(med_err, 0.0, 0.0, 0.0) * scored).sum(dim=1)
        copy_total = (torch.nan_to_num(copy_err, 0.0, 0.0, 0.0) * scored).sum(dim=1)
        per_sample = per_sample * (copy_total < med_total).to(per_sample.dtype)

    return _reduce(per_sample * float(weight), reduction)


__all__ = [
    "BASE_SEASONALITY",
    "COMMIT_WEIGHT",
    "pinball_loss",
    "seasonal_copy_baseline",
    "committing_loss",
]

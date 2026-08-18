"""Per-context-window min-max normalization used by the encoder path."""
from __future__ import annotations

import torch
import torch.nn as nn


class WindowMinMax(nn.Module):
    """Per-context-window min-max normalization for the encoder path.

    Per-window statistics:

        x_min   = x.min(1, keepdim=True)[0].detach()
        x_max   = x.max(1, keepdim=True)[0].detach()
        x_range = (x_max - x_min).clamp(min=1e-5).detach()
        x_norm  = (x - x_min) / x_range            # → [0, 1]

    Stats are detached: gradients do NOT flow through normalization.
    Uses explicit ``transform`` / ``inverse_transform`` so the backbone can
    hold the (x_min, x_range) tuple across encoder + decoder and apply the
    inverse at loss / forecast time.
    """

    def __init__(self, eps_clamp: float = 1e-5) -> None:
        super().__init__()
        self.eps_clamp = eps_clamp

    def transform(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize ``x: (B, L, 1)`` to [0, 1] per context window.

        Returns ``(x_normalized, x_min, x_range)`` with stats detached and
        ``x_range >= eps_clamp`` to avoid div-by-zero on constant series.

        Robust to NaN / +inf / -inf in ``x``: those positions are filled
        with 0 BEFORE computing min/max.
        """
        x_filled = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_min = x_filled.min(dim=1, keepdim=True).values.detach()
        x_max = x_filled.max(dim=1, keepdim=True).values.detach()
        x_range = (x_max - x_min).clamp(min=self.eps_clamp).detach()
        x_normalized = (x_filled - x_min) / x_range
        return x_normalized, x_min, x_range

    @staticmethod
    def inverse_transform(
        y_pred_normalized: torch.Tensor,
        x_min: torch.Tensor,
        x_range: torch.Tensor,
    ) -> torch.Tensor:
        """Un-normalize ``(B, p)`` predictions back to raw magnitude.

        ``x_min`` / ``x_range`` come from a prior ``transform`` call and are
        ``(B, 1, 1)``; the trailing dim is squeezed for broadcasting against
        ``(B, p)``.
        """
        return y_pred_normalized * x_range.squeeze(-1) + x_min.squeeze(-1)

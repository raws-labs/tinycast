"""TinyCast model assembly.

    model(past_values, scale_factor, prediction_length, batch_first).quantile_outputs
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import TinyCastConfig


@dataclass
class PredictionOutput:
    quantile_outputs: torch.Tensor
    prediction_outputs: torch.Tensor = None


class TinyCastBackbone(nn.Module):
    """Inner model: per-window min-max norm -> dilated-conv core -> denorm."""

    def __init__(self, config: TinyCastConfig):
        super().__init__()
        self.config = config
        from .backbone import DilatedConvBackbone
        from .normalization import WindowMinMax

        self.core = DilatedConvBackbone(
            seq_len=int(config.seq_len),
            p_out=int(config.output_token_len),
            n_quantiles=int(config.num_quantiles),
            d=int(config.conv_dim),
            n_layers=int(config.n_layers),
            kernel=int(config.kernel_size),
            ffn_mult=float(config.ffn_mult),
            top_k_periods=int(config.top_k_periods),
            significance_alpha=float(config.significance_alpha),
            n_harmonics=int(config.n_harmonics),
            pool_kind=str(config.pool_kind),
            causal=bool(config.causal),
            phase_bins=int(config.phase_bins),
            decoder_depth=int(config.decoder_depth),
            separable_conv=bool(config.separable_conv),
            share_ffn=bool(config.share_ffn),
            future_conv=bool(config.future_conv),
            future_conv_layers=int(config.future_conv_layers),
            future_conv_seed=int(config.future_conv_seed),
        )
        self.norm = WindowMinMax(eps_clamp=1e-5)

    def encode(
        self,
        past_values: torch.Tensor,
        batch_first: bool = True,
        scale_factor: "torch.Tensor | float | None" = None,
        horizon: "int | None" = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """per-window min-max norm -> dilated-conv core -> normalized y."""
        x = past_values
        if not batch_first:
            x = x.transpose(0, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        x_normed, x_min, x_range = self.norm.transform(x)
        nan_mask = (~torch.isnan(x)).to(x.dtype)
        y_norm = self.core(
            x_normed, nan_mask=nan_mask, scale_factor=scale_factor,
            horizon=horizon,
        )
        return y_norm, x_min, x_range


class TinyCastForPrediction(nn.Module):
    """Top-level model container.

    Usage:
        model = TinyCastForPrediction(TinyCastConfig())
        out = model(past_values=x, scale_factor=sf, prediction_length=pl,
                    batch_first=False)
        quantiles = out.quantile_outputs  # (B, Q, pred_len, 1)

    NOTE: this ``forward`` is the SINGLE-SHOT arbitrary-horizon path. The
    deployed GIFT-Eval inference uses the AR-rollout predictor
    (``tinycast.predictor.TinyCastPredictor``), which calls ``self.model.encode``
    in 48-step chunks. For horizons <= 48 the two paths coincide.
    """

    def __init__(self, config: TinyCastConfig):
        super().__init__()
        self.config = config
        self.model = TinyCastBackbone(config)

    def forward(self, past_values, scale_factor=None, prediction_length=None,
                batch_first=None):
        if scale_factor is None:
            scale_factor = 1.0
        if batch_first is None:
            batch_first = True
        ctx = past_values
        if not batch_first:
            ctx = ctx.transpose(0, 1)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(-1)
        ctx_len_max = int(self.config.seq_len)
        ctx_in = ctx[:, -ctx_len_max:, :]
        p_out_native = int(self.model.core.p_out)
        pred_len = int(prediction_length) if prediction_length else p_out_native
        y_norm, x_min, x_range = self.model.encode(
            ctx_in, batch_first=True, scale_factor=scale_factor,
            horizon=pred_len,
        )
        if y_norm.dim() == 2:
            y_norm = y_norm.unsqueeze(-1)
        y_pred = y_norm * x_range + x_min
        quantile_outputs = y_pred.permute(0, 2, 1).unsqueeze(-1)
        return PredictionOutput(quantile_outputs=quantile_outputs)

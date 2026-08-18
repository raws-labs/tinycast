"""GIFT-Eval gluonts predictor for TinyCast.

``TinyCastPredictor`` is the deployed predictor. It wraps the model in the
gluonts Predictor protocol and drives autoregressive-rollout decoding
(48-step chunks), flip-invariance symmetrization, NaN-imputation and
optional period-alignment downsampling.

Anything that changes what the predictor emits says so:

``device``          a named device must exist. ``device=None`` is the only
                    request that selects one for you (CUDA when present, else
                    CPU); the resolved device is ``predictor.device``.
``TINYCAST_INT8``   ``w8`` or ``w8a8`` post-training fake quantization, off by
                    default. Any other non-empty value is an error, not an
                    inert setting.
``TINYCAST_TILT_K`` an eval-only probe that moves the emitted median off the
                    quantile grid, off (``0``) by default, with
                    ``TINYCAST_TILT_MODE`` in {``adaptive``, ``fixed``}.

The environment variables are read once, when the predictor is constructed, and
an engaged one prints a line naming itself. Reproducing the published numbers
means leaving them unset.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .checkpoint import load_checkpoint
from .normalization import WindowMinMax
from .scale import seasonal_scale_factor

try:
    from torch.amp import autocast as _autocast_fp
except Exception:  # pragma: no cover
    _autocast_fp = None


def _resolve_device(requested: Optional[str]) -> torch.device:
    """Return the requested device, or say why it is unavailable.

    ``None`` is the request to choose: CUDA when it is available, CPU otherwise.
    Every other value is a requirement. Falling back to CPU behind the caller's
    back drops bf16 autocast for fp32 and moves every forecast by roughly 1e-3,
    which is small enough to read as a regression and large enough to fail an
    assertion against the published aggregates, so an absent accelerator is an
    error rather than a substitution.
    """
    if requested is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={requested!r} was requested but "
                "torch.cuda.is_available() is False. Pass device='cpu' to run "
                "on CPU (fp32, and about 1e-3 off the published numbers), or "
                "device=None to take whichever device is present."
            )
        if dev.index is not None and dev.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"device={requested!r} was requested but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            f"device={requested!r} was requested but MPS is unavailable. "
            "Pass device='cpu' to run on CPU."
        )
    return dev


_INT8_MODES = ("w8", "w8a8")
_INT8_OFF = ("", "0", "off", "no", "false", "none")
_TILT_MODES = ("adaptive", "fixed")


def _resolve_int8() -> Optional[str]:
    """Read ``TINYCAST_INT8``: the quantization mode, or ``None`` when off.

    Only ``w8`` and ``w8a8`` name a scheme. A truthy value such as ``1`` names
    no scheme, and picking one for the caller would quantize a run that asked
    for something else, so an unrecognized value raises. The alternative is what
    this used to do: leave the run in fp32 while the caller believes they are
    measuring INT8.
    """
    raw = os.environ.get("TINYCAST_INT8", "").strip().lower()
    if raw in _INT8_OFF:
        return None
    if raw not in _INT8_MODES:
        raise ValueError(
            f"TINYCAST_INT8={raw!r} is not a quantization mode. Use 'w8' "
            "(per-channel INT8 weights) or 'w8a8' (+ per-tensor dynamic INT8 "
            "activations), or unset it to run in floating point."
        )
    return raw


def _resolve_tilt() -> Tuple[float, str]:
    """Read the tilt probe's variables, announcing an engaged one.

    ``TINYCAST_INT8`` prints a line when it engages; this does the same, so
    neither switch can invalidate a golden array without appearing in the log.
    ``TINYCAST_TILT_MODE`` is checked only when the tilt is on, which keeps it
    inert at ``K=0`` as documented.
    """
    raw = os.environ.get("TINYCAST_TILT_K", "").strip()
    try:
        k = float(raw) if raw else 0.0
    except ValueError:
        raise ValueError(
            f"TINYCAST_TILT_K={raw!r} is not a number. Unset it, or set 0, to "
            "emit the model's own median."
        ) from None
    mode = os.environ.get("TINYCAST_TILT_MODE", "adaptive").strip().lower()
    if k != 0.0:
        if mode not in _TILT_MODES:
            raise ValueError(
                f"TINYCAST_TILT_MODE={mode!r} is not a tilt rule. Use "
                "'adaptive' (skew-following) or 'fixed'."
            )
        print(
            f"[tilt] TINYCAST_TILT_K={k:g} {mode}: the emitted median is "
            "re-interpolated off the quantile grid, so this is not the "
            "published model's forecast (inert below 3 quantiles)",
            flush=True,
        )
    return k, mode


def _numpy_fill(arr: np.ndarray) -> np.ndarray:
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return arr[np.arange(idx.shape[0])[:, None], idx]


class ARRolloutPredictor:
    """Base predictor: AR rollout + flip + NaN-imputation + period downsample.

    Subclasses install ``self.model`` (a callable ``(x, x_mark, y_mark)``).
    ``device=None`` selects one; a named device must exist. Either way the
    device that ran is ``self.device``.

    Every setting that changes a forecast is a named parameter, so a keyword
    that is not one raises. Absorbing unknown keywords would let a misspelling
    such as ``force_flip_invarience=True`` construct a predictor with flip
    symmetrization off and return a forecast that looks right and cannot be
    reproduced.
    """

    def __init__(
        self,
        prediction_length: int,
        device: Optional[str] = None,
        seq_len: int = 2048,
        input_token_len: int = 2048,
        output_token_len: int = 48,
        num_samples: int = 100,
        batch_size: int = 256,
        use_amp: int = 1,
        downsample_factor: int = 1,
        force_flip_invariance: bool = False,
        adaptive_ar_rollout: bool = False,
    ):
        self.device = _resolve_device(device)
        self.prediction_length = int(prediction_length)
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.input_token_len = int(input_token_len)
        self.output_token_len = int(output_token_len)
        self.use_amp = int(use_amp)
        self.downsample_factor = int(downsample_factor)
        self.force_flip_invariance = bool(force_flip_invariance)
        self.adaptive_ar_rollout = bool(adaptive_ar_rollout)
        # Quantile levels emitted by the model (set by subclasses from cfg).
        self.quantiles = [0.5]
        self.model = None

    def _downsample_if_needed(
        self, series: torch.Tensor
    ) -> Tuple[torch.Tensor, int]:
        cur = series
        if self.downsample_factor > 1:
            cur = cur[::self.downsample_factor]
        return cur, self.downsample_factor

    def _left_pad_to_len(
        self, arr: np.ndarray, target_len: int
    ) -> Tuple[np.ndarray, int]:
        if arr.shape[0] >= target_len:
            return arr[-target_len:], 0
        pad_len = target_len - arr.shape[0]
        fill_value = arr[0] if arr.shape[0] > 0 else 0.0
        padding = np.full((pad_len,), fill_value, dtype=arr.dtype)
        return np.concatenate([padding, arr], axis=0), pad_len

    def _prepare_context_matrix(
        self, context: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[int]]:
        xs = []
        downsample_factors = []
        for c in context:
            cur, df = self._downsample_if_needed(c)
            downsample_factors.append(df)

            cur_np = cur.detach().cpu().float().numpy()
            cur_np, _ = self._left_pad_to_len(cur_np, self.seq_len)

            x2d = cur_np[None, :]
            x_interp = np.copy(x2d)
            series = x2d[0]
            if np.any(np.isnan(series)):
                valid_mask = ~np.isnan(series)
                if np.sum(valid_mask) >= 2:
                    valid_idx = np.where(valid_mask)[0]
                    valid_val = series[valid_mask]
                    x_interp[0] = np.interp(
                        np.arange(len(series)), valid_idx, valid_val
                    )
                else:
                    x_interp = _numpy_fill(x2d)
            ff = _numpy_fill(x_interp)
            bf = np.flip(_numpy_fill(np.flip(x_interp, axis=1)), axis=1)
            x_imp = np.where(np.isnan(ff), bf, ff)
            x_imp = np.where(np.isnan(x_imp), 0.0, x_imp)
            xs.append(x_imp[0])

        x = torch.tensor(
            np.stack(xs), device=self.device, dtype=torch.float32
        ).unsqueeze(-1)
        return x, downsample_factors

    def _decode_autoregressive(
        self,
        init_ctx: torch.Tensor,
        use_bf16: bool,
        downsample_factors: List[int],
    ) -> torch.Tensor:
        B, _, C = init_ctx.shape
        roll_len = int(self.output_token_len)

        if self.adaptive_ar_rollout:
            try:
                from .periodogram import significant_periods
                periods, _s, _nv = significant_periods(
                    init_ctx[:, -self.seq_len:, 0].float(),
                    min_period=2, max_period=self.seq_len // 2, top_k=1,
                )
                pos = periods[periods > 0]
                if pos.numel() > 0:
                    p0 = int(pos.float().median().item())
                    if 2 <= p0 < roll_len:
                        roll_len = max(4, p0)
            except Exception:
                pass

        target_pred_lens = [
            int(self.prediction_length) // int(max(1, df))
            for df in downsample_factors
        ]
        max_target_pred_len = max(target_pred_lens)
        steps = math.ceil(max_target_pred_len / roll_len)
        preds: List[torch.Tensor] = []
        batch_ctx = init_ctx

        y_mark = torch.zeros(
            B, self.output_token_len, C,
            device=self.device, dtype=init_ctx.dtype,
        )

        for _ in range(steps):
            x_in = batch_ctx[:, -self.seq_len:, :]
            x_mark = torch.zeros_like(x_in)
            if _autocast_fp is not None and self.use_amp and use_bf16:
                try:
                    with _autocast_fp("cuda", dtype=torch.bfloat16):
                        out = self.model(x_in, x_mark, y_mark)
                except Exception:
                    out = self.model(x_in, x_mark, y_mark)
            else:
                out = self.model(x_in, x_mark, y_mark)
            chunk = out[:, -self.output_token_len:, :][:, :roll_len, :]  # (B, roll, Q)
            preds.append(chunk)
            # Feed only the MEDIAN quantile back into the context (the AR state
            # is a univariate series); keep all Q in preds for the forecast.
            q_mid = chunk.shape[-1] // 2
            batch_ctx = torch.cat([batch_ctx, chunk[:, :, q_mid:q_mid + 1]], dim=1)

        return torch.cat(preds, dim=1)                                   # (B, pl, Q)

    @torch.no_grad()
    def predict(self, test_data_input, use_bf16_if_available: bool = True):
        from gluonts.itertools import batcher
        from gluonts.model.forecast import SampleForecast, QuantileForecast

        forecasts: List = []
        use_bf16 = bool(
            use_bf16_if_available
            and self.device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )

        for batch in batcher(test_data_input, batch_size=self.batch_size):
            targets = [
                torch.tensor(entry["target"], dtype=torch.float32)
                for entry in batch
            ]
            batch_ctx, dfs = self._prepare_context_matrix(targets)
            pred_pos = self._decode_autoregressive(batch_ctx, use_bf16, dfs)  # (B,pl,Q)
            if self.force_flip_invariance:
                pred_neg = self._decode_autoregressive(-batch_ctx, use_bf16, dfs)
                # Flip-symmetrize. For quantiles the tau-quantile of -y is
                # -(the (1-tau)-quantile of y), so reverse the quantile axis on
                # the negated branch. Q=1 (median) reverse is a no-op.
                pred = 0.5 * (pred_pos - pred_neg.flip(dims=[-1]))
            else:
                pred = pred_pos

            Q = pred.shape[-1]
            pred_np = pred.float().detach().cpu().numpy()                 # (B, pl, Q)
            if not np.isfinite(pred_np).all():
                for qi in range(Q):
                    pred_np[:, :, qi] = _numpy_fill(pred_np[:, :, qi])

            for i, ts in enumerate(batch):
                df = int(max(1, dfs[i]))
                target_pl = int(self.prediction_length) // df
                arr = pred_np[i, :target_pl, :]                          # (target_pl, Q)
                if df > 1:
                    new_len = int(self.prediction_length)
                    src = np.linspace(0, 1, arr.shape[0])
                    dst = np.linspace(0, 1, new_len)
                    arr = np.stack([np.interp(dst, src, arr[:, qi])
                                    for qi in range(Q)], axis=1)         # (new_len, Q)
                start_date = ts["start"] + len(ts["target"])
                if Q > 1:
                    # Sort across quantiles to guarantee non-crossing, then a
                    # QuantileForecast so gluonts scores WQL/CRPS over the deciles.
                    arr = np.sort(arr, axis=1)
                    forecasts.append(QuantileForecast(
                        forecast_arrays=arr.T,                          # (Q, pl)
                        start_date=start_date,
                        forecast_keys=[str(q) for q in self.quantiles],
                    ))
                else:
                    samples = np.repeat(arr[:, 0][None, :], self.num_samples, axis=0)
                    forecasts.append(
                        SampleForecast(samples=samples, start_date=start_date)
                    )

        return forecasts


class _BackboneAdapter(nn.Module):
    """Adapts the model wrapper to the ``(x, x_mark, y_mark) -> (B, p, Q)`` contract."""

    def __init__(self, backbone: nn.Module, scale_factor: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.scale_factor = float(scale_factor)
        # single-shot counterfactual: emit this many steps in ONE forward.
        self.pred_len_override = None
        # Resolved once, and announced when on: a forecast must not change
        # because a variable was exported after the predictor was built.
        self.tilt_k, self.tilt_mode = _resolve_tilt()

    def forward(self, x, x_mark=None, y_mark=None, **kwargs):
        y_norm, x_min, x_range = self.backbone.encode(
            x, batch_first=True, scale_factor=self.scale_factor,
            horizon=self.pred_len_override,
        )
        # Match training: the chunk loss clamps y_norm to [-5,5] before the
        # pinball loss, so the model is never optimized outside that band.
        # Clamp at inference too: otherwise an un-penalized overshoot feeds
        # back into the AR rollout context and compounds over chunks.
        y_norm = y_norm.clamp(-5.0, 5.0)

        # num_quantiles > 1 => y_norm is (B, p, Q). Manual inverse since
        # WindowMinMax.inverse_transform squeezes x_min/x_range for the (B, p)
        # point case; x_min/x_range are (B, 1, 1) so they broadcast directly.
        if y_norm.dim() == 3:
            tk = self.tilt_k
            Q = y_norm.shape[-1]
            if tk != 0.0 and Q >= 3:
                # EVAL-ONLY de-hedge probe (no training); default off (tk=0).
                qm = Q // 2
                lo = y_norm[..., 0]; hi = y_norm[..., -1]; med = y_norm[..., qm]
                if self.tilt_mode == "fixed":
                    tau = torch.full_like(med, 0.5 + tk)
                else:                                            # skew-adaptive
                    asym = (hi + lo - 2 * med) / (hi - lo).abs().clamp(min=1e-6)
                    tau = 0.5 + tk * torch.tanh(asym)            # >0 right-skew
                tau = tau.clamp(qm / (Q + 1.0), (qm + 2) / (Q + 1.0))
                idxf = (tau * (Q + 1) - 1).clamp(0, Q - 1 - 1e-4)
                ilo = idxf.floor().long().clamp(0, Q - 2)
                frac = (idxf - ilo.to(idxf.dtype)).clamp(0, 1)
                qlo = torch.gather(y_norm, -1, ilo.unsqueeze(-1)).squeeze(-1)
                qhi = torch.gather(y_norm, -1, (ilo + 1).unsqueeze(-1)).squeeze(-1)
                y_norm = y_norm.clone()
                y_norm[..., qm] = qlo * (1 - frac) + qhi * frac
            return y_norm * x_range + x_min                    # (B, p, Q)
        y_pred = WindowMinMax.inverse_transform(y_norm, x_min, x_range)
        return y_pred.unsqueeze(-1)  # (B, p, 1)


class TinyCastPredictor(ARRolloutPredictor):
    """The deployed predictor: AR-rollout decoding around the TinyCast model."""

    def __init__(
        self,
        prediction_length: int,
        checkpoint_path: str,
        device: Optional[str] = None,
        num_samples: int = 100,
        batch_size: int = 256,
        use_amp: int = 1,
        downsample_factor: int = 1,
        force_flip_invariance: bool = False,
        freq: Optional[str] = None,
        domain: Optional[str] = None,
        no_daily: bool = False,
        single_shot: bool = False,
        adaptive_ar_rollout: bool = False,
        config_path: Optional[str] = None,
    ):
        self.single_shot = bool(single_shot)
        # 1. Load the weights (safetensors + config.json).
        model, cfg = load_checkpoint(checkpoint_path, config_path)

        # 2. Eval-time scale_factor from (freq, domain). bizitobs_l2c has no
        #    daily cycle -> /7.
        if freq is not None and domain is not None:
            sf_eval = seasonal_scale_factor(freq, domain)
            if no_daily:
                sf_eval /= 7
        else:
            sf_eval = 1.0
        self._eval_scale_factor = float(sf_eval)

        # 3. Base predictor bookkeeping (drives AR-rollout / batching / flip).
        super().__init__(
            prediction_length=prediction_length,
            device=device,
            seq_len=int(cfg.seq_len),
            input_token_len=int(cfg.seq_len),
            output_token_len=int(cfg.output_token_len),
            num_samples=num_samples,
            batch_size=batch_size,
            use_amp=use_amp,
            downsample_factor=downsample_factor,
            force_flip_invariance=force_flip_invariance,
            adaptive_ar_rollout=adaptive_ar_rollout,
        )
        self.quantiles = list(getattr(cfg, "quantiles", [0.5])) or [0.5]
        self._missing_aware = bool(getattr(cfg, "missing_channel", False))

        # 4. Install the model wrapped in the adapter interface.
        backbone = model.model  # TinyCastForPrediction -> TinyCastBackbone
        backbone.to(self.device).eval()
        # Optional INT8 post-training fake-quant. Off by default;
        # TINYCAST_INT8=w8 (weights) or w8a8 (+ dynamic activations).
        # quantize_int8_ prints the mode it applied.
        _int8 = _resolve_int8()
        if _int8 is not None:
            from .quant import quantize_int8_
            quantize_int8_(backbone, _int8)
        self.model = _BackboneAdapter(
            backbone, scale_factor=self._eval_scale_factor,
        ).to(self.device).eval()
        if self.single_shot:
            self.model.pred_len_override = int(prediction_length)

    def _decode_autoregressive(self, init_ctx, use_bf16, downsample_factors):
        """Single-shot override: one forward for the full horizon (no AR loop).
        Default deployed behavior (single_shot=False) falls back to the base
        AR rollout."""
        if not self.single_shot:
            return super()._decode_autoregressive(init_ctx, use_bf16, downsample_factors)
        x_in = init_ctx[:, -self.seq_len:, :]
        x_mark = torch.zeros_like(x_in)
        use_cuda = bool(self.use_amp and use_bf16 and str(self.device).startswith("cuda"))
        if use_cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(x_in, x_mark, None)
        else:
            out = self.model(x_in, x_mark, None)
        return out  # (B, prediction_length, 1)

    def _prepare_context_matrix(self, context):
        """Missing-aware override: keep genuine gaps as NaN so the observed-mask
        + observed-only normalization see true missingness. Falls back to the
        base (interpolating) behavior when the model is not missing-aware
        (the deployed model is not)."""
        if not getattr(self, "_missing_aware", False):
            return super()._prepare_context_matrix(context)
        xs, dfs = [], []
        for c in context:
            cur, df = self._downsample_if_needed(c)
            dfs.append(df)
            a = cur.detach().cpu().float().numpy()
            if a.shape[0] >= self.seq_len:
                a = a[-self.seq_len:]
            else:
                pad = np.full((self.seq_len - a.shape[0],), np.nan, dtype=a.dtype)
                a = np.concatenate([pad, a], axis=0)
            xs.append(a)
        x = torch.tensor(
            np.stack(xs), device=self.device, dtype=torch.float32
        ).unsqueeze(-1)
        return x, dfs

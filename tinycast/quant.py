"""Post-training INT8 quantization (fake-quant) for TinyCast.

Measures the GIFT-Eval accuracy of an INT8-deployed model by simulating INT8 arithmetic in
floating point. Per-output-channel symmetric INT8 weights on every Linear and Conv1d; optional
per-tensor dynamic INT8 activations. RMSNorm, the SiLU gate, the min-max (de)normalization,
and the rFFT period detector stay in full precision: they are off the convolutional mixing path
and run as fp/LUT ops on the target runtime.

Modes (env ``TINYCAST_INT8``):
  ``w8``    per-channel INT8 weights, fp activations. The ~145 KB weight footprint; isolates the
            weight-quantization error.
  ``w8a8``  + per-tensor dynamic INT8 activations (scale from each tensor's own range, so no
            calibration set is needed; an optimistic but faithful estimate of full INT8 compute).

The weight quant is applied in-place to the parameter tensors, so it is correct regardless of
whether a module is invoked via ``__call__`` or functionally (the separable pointwise conv is run
as ``F.linear(weight.squeeze(-1))``). Activation quant (``w8a8``) uses a forward-pre-hook on every
Linear/Conv1d to quantize inputs, plus a forward-hook on Conv1d to quantize the depthwise output
that feeds the functional pointwise, covering every activation site on the mixing path.
"""
from __future__ import annotations

import torch
import torch.nn as nn

_QMIN, _QMAX = -128, 127  # int8 symmetric (zero-point 0)


@torch.no_grad()
def _fq_weight_per_outchannel(w: torch.Tensor) -> torch.Tensor:
    """Symmetric per-output-channel (axis 0) int8 fake-quant of a weight tensor.

    Linear weight is (out, in); Conv1d weight is (out, in/groups, k). Axis 0 is the output
    channel in both, so a per-axis-0 scale is the standard per-channel weight scheme.
    """
    red = tuple(d for d in range(w.dim()) if d != 0)
    amax = w.abs().amax(dim=red, keepdim=True).clamp_(min=1e-12)
    scale = amax / _QMAX
    return (torch.round(w / scale).clamp_(_QMIN, _QMAX) * scale).to(w.dtype)


def _fq_act_dynamic(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor symmetric int8 fake-quant with a dynamic (this-tensor) scale."""
    if not torch.is_floating_point(x):
        return x
    amax = x.detach().abs().amax().clamp(min=1e-12)
    scale = amax / _QMAX
    return torch.round(x / scale).clamp(_QMIN, _QMAX) * scale


def _pre_hook(_mod, inp):
    if not inp:
        return None
    return (_fq_act_dynamic(inp[0]),) + tuple(inp[1:])


def _post_hook(_mod, _inp, out):
    return _fq_act_dynamic(out)


def quantize_int8_(model: nn.Module, mode: str = "w8") -> nn.Module:
    """In-place INT8 fake-quant of ``model``. ``mode`` in {"w8", "w8a8"}. Returns ``model``."""
    mode = mode.strip().lower()
    if mode not in ("w8", "w8a8"):
        raise ValueError(f"unknown INT8 mode {mode!r} (expected 'w8' or 'w8a8')")
    n_w = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                m.weight.data.copy_(_fq_weight_per_outchannel(m.weight.data))
                n_w += 1
                if mode == "w8a8":
                    m.register_forward_pre_hook(_pre_hook)
                    if isinstance(m, nn.Conv1d):
                        m.register_forward_hook(_post_hook)
    print(
        f"[quant] INT8 {mode}: fake-quantized {n_w} Linear/Conv1d weight tensors"
        + (" + per-tensor dynamic activation quant" if mode == "w8a8" else ""),
        flush=True,
    )
    return model

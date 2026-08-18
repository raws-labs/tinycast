"""Training entry point for TinyCast.

``train`` runs the shipped recipe: a four-block autoregressive rollout under
scheduled sampling, the nine-quantile pinball loss plus the gated committing
term, AdamW, and a warmup-stable-decay learning-rate schedule. It runs
single-process, and CPU is a first-class device.

What a short run does and does not reproduce. The released 146,505-parameter
checkpoint is 36,621 optimizer steps at an effective batch of 4096, in
bf16-mixed across eight GPUs, over GIFT-Eval-Pretrain plus Chronos KernelSynth
plus four synthetic shards, with the final weights the uniform mean of the last
eight periodic checkpoints. A short run on a small corpus reproduces the
mechanics: the rollout, the objective, the schedule shape, the optimizer. It
does not reproduce the checkpoint or its numbers.

``max_steps`` reshapes the run rather than truncating it. Warmup, stable and
decay are fractions of ``max_steps``, and the scheduled-sampling feedback
probability ramps over its first half, so a 200-step run warms up, holds and
decays inside 200 steps and reaches full feedback by step 100. A 200-step run
is therefore not a prefix of a 36,621-step run, and their step-100 states are
not comparable.

Two further departures from the released run, both deliberate. This entry point
trains in the model's own precision instead of bf16-mixed, and it writes
periodic checkpoints but does not average them.
"""
from __future__ import annotations

import collections.abc
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .config import TinyCastConfig
from .losses import (
    COMMIT_WEIGHT,
    committing_loss,
    pinball_loss,
    seasonal_copy_baseline,
)
from .model import TinyCastForPrediction

# --- the shipped recipe ----------------------------------------------------
# Every value below can be overridden by an attribute of the same name on the
# config object passed to ``train``; the defaults are what the released
# checkpoint was trained with.
AR_CHUNKS = 4                    # K: blocks rolled out per training window
SCHEDULED_SAMPLING_MAX = 0.5     # peak probability of feeding our own median
LEARNING_RATE = 3e-3
MIN_LEARNING_RATE = 1e-5
WARMUP_FRACTION = 0.05
DECAY_FRACTION = 0.35
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
ADAM_BETAS = (0.9, 0.95)
COMMIT_GATED = True

# Guards carried over from the released run: predictions are optimized inside a
# fixed normalized band, and a window whose context is flat is not scored.
_PRED_CLAMP = 5.0
_TARGET_CLAMP = 10.0
_MIN_RANGE = 1e-4


@dataclass
class TrainResult:
    """What a finished run leaves behind.

    Attributes:
        checkpoint_path: the final ``model.safetensors``, next to a
            ``config.json``, so ``tinycast.load_checkpoint`` reads it directly.
        steps: optimizer steps actually run.
        losses: one training loss per optimizer step, in order.
        learning_rates: the learning rate each of those steps used.
        checkpoints: the periodic checkpoints written along the way, oldest
            first. The released weights are the uniform mean of the last eight
            of these; ``train`` writes them but does not average them.
        window_width: the sample width the run required, ``L + K * p``.
    """

    checkpoint_path: str
    steps: int
    losses: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    window_width: int = 0


def training_window_width(
    config: TinyCastConfig, chunks: Optional[int] = None
) -> int:
    """Sample width the rollout consumes: ``L + K * p``.

    ``L`` is the encoder context, ``p`` the horizon unit and ``K`` the number of
    autoregressive blocks. Each block re-normalizes a context that has slid
    forward by ``p``, so the last block still needs a full ``L`` of history
    behind it. Call this instead of hard-coding the width.
    """
    k = int(_setting(config, "ar_chunks", AR_CHUNKS) if chunks is None else chunks)
    if k < 1:
        raise ValueError(f"ar_chunks must be at least 1, got {k}.")
    return int(config.seq_len) + k * int(config.output_token_len)


def _setting(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, None)
    return default if value is None else value


# --- reproducibility -------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _Determinism:
    """Turn deterministic kernels on, then put the process back as it was.

    The flags are process-global, so a library function has no business leaving
    them changed. Entered only when the caller asks for ``deterministic=True``.
    """

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self._saved: Dict[str, Any] = {}

    def __enter__(self) -> "_Determinism":
        if not self.enabled:
            return self
        self._saved["algorithms"] = torch.are_deterministic_algorithms_enabled()
        self._saved["cudnn_deterministic"] = torch.backends.cudnn.deterministic
        self._saved["cudnn_benchmark"] = torch.backends.cudnn.benchmark
        self._saved["cublas"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if self.device.type == "cuda" and self._saved["cublas"] is None:
            # cuBLAS needs this set before its handle is created for the
            # deterministic reduction path to be available.
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self.enabled:
            return
        torch.use_deterministic_algorithms(self._saved["algorithms"])
        torch.backends.cudnn.deterministic = self._saved["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = self._saved["cudnn_benchmark"]
        if self._saved["cublas"] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = self._saved["cublas"]


def _resolve_device(device: str) -> torch.device:
    """Return the requested device, or say why it is unavailable.

    A silent fall back to CPU would change the numbers with no signal, so a
    request for an absent accelerator is an error.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested but torch.cuda.is_available() is "
            "False. Pass device='cpu' to train on CPU."
        )
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "device='mps' was requested but MPS is unavailable. Pass "
            "device='cpu' to train on CPU."
        )
    return dev


# --- data ------------------------------------------------------------------

class _SampleBatcher:
    """Group a plain iterable of samples into collated batches.

    Re-iterated once per epoch, so a list, a sequence or any object with a
    fresh ``__iter__`` works. A one-shot generator is exhausted after its first
    pass and ``_epochs`` reports that rather than looping on nothing.
    """

    def __init__(self, samples: Iterable[Any], batch_size: int) -> None:
        self.samples = samples
        self.batch_size = int(batch_size)

    def __iter__(self):
        from torch.utils.data import default_collate

        buffer: List[Any] = []
        for sample in self.samples:
            buffer.append(sample)
            if len(buffer) == self.batch_size:
                yield default_collate(buffer)
                buffer = []
        # A short tail is dropped, matching drop_last on the DataLoader path:
        # a partial batch changes the effective batch size mid-run.


def _build_loader(
    data: Any,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> Iterable[Any]:
    from torch.utils.data import DataLoader, Dataset, IterableDataset

    if isinstance(data, DataLoader):
        # The caller has already made every batching decision; respect it.
        return data

    pin_memory = device.type == "cuda"

    def worker_init(worker_id: int) -> None:
        _seed_everything(seed + 1 + worker_id)

    if isinstance(data, IterableDataset):
        return DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            worker_init_fn=worker_init,
        )
    if isinstance(data, Dataset):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            generator=generator,
            worker_init_fn=worker_init,
        )
    if isinstance(data, collections.abc.Iterable):
        if num_workers:
            raise ValueError(
                "num_workers > 0 needs a Dataset or a DataLoader; a plain "
                "iterable is batched in this process."
            )
        return _SampleBatcher(data, batch_size)
    raise TypeError(
        "data must be a Dataset, an IterableDataset, a DataLoader or an "
        f"iterable of samples, got {type(data).__name__}."
    )


def _epochs(loader: Iterable[Any]):
    """Yield batches forever, restarting the loader between passes."""
    while True:
        produced = False
        for batch in loader:
            produced = True
            yield batch
        if not produced:
            raise ValueError(
                "the training data yielded no batches. A one-shot generator is "
                "exhausted after one pass: pass a Dataset, a DataLoader or a "
                "re-iterable sequence, and check that it holds at least one "
                "full batch."
            )


def _unpack_batch(
    batch: Any, width: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize a batch to ``(window, mask, scale_factor)`` on ``device``.

    Accepts a mapping with a ``window`` key (plus optional ``mask`` and
    ``scale_factor``), a bare tensor of windows, or a positional
    ``(window, mask, scale_factor)`` sequence.
    """
    mask: Any = None
    scale: Any = None
    if isinstance(batch, dict):
        if "window" not in batch:
            raise KeyError(
                f"batch dict needs a 'window' key, got {sorted(batch)}."
            )
        window = batch["window"]
        mask = batch.get("mask")
        scale = batch.get("scale_factor")
    elif torch.is_tensor(batch):
        window = batch
    elif isinstance(batch, (tuple, list)):
        window = batch[0]
        mask = batch[1] if len(batch) > 1 else None
        scale = batch[2] if len(batch) > 2 else None
    else:
        raise TypeError(
            "a batch must be a dict, a tensor or a sequence, got "
            f"{type(batch).__name__}."
        )

    window = torch.as_tensor(window).to(device=device, dtype=dtype)
    if window.dim() == 3 and window.shape[-1] == 1:
        window = window.squeeze(-1)
    if window.dim() != 2:
        raise ValueError(
            f"window must be (B, L + K*p), got {tuple(window.shape)}."
        )
    if window.shape[1] != width:
        raise ValueError(
            f"window is {window.shape[1]} long but the rollout needs "
            f"{width} = seq_len + ar_chunks * output_token_len. Use "
            "tinycast.train.training_window_width(config) to size the dataset."
        )

    if mask is None:
        observed = torch.isfinite(window).to(dtype)
    else:
        observed = torch.as_tensor(mask).to(device=device, dtype=dtype)
        if observed.dim() == 3 and observed.shape[-1] == 1:
            observed = observed.squeeze(-1)
        if observed.shape != window.shape:
            raise ValueError(
                f"mask {tuple(observed.shape)} does not match window "
                f"{tuple(window.shape)}."
            )
        observed = observed * torch.isfinite(window).to(dtype)
    window = torch.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

    if scale is None:
        # No metadata: one sample per canonical cycle. Datasets that carry a
        # frequency should supply tinycast.scale.seasonal_scale_factor instead,
        # since the local anchor channels and the seasonal copy both read it.
        scale_factor = torch.ones(
            window.shape[0], device=device, dtype=torch.float32
        )
    else:
        scale_factor = torch.as_tensor(scale).to(
            device=device, dtype=torch.float32
        ).reshape(-1)
        if scale_factor.numel() == 1:
            scale_factor = scale_factor.expand(window.shape[0])
        elif scale_factor.numel() != window.shape[0]:
            raise ValueError(
                f"scale_factor carries {scale_factor.numel()} entries for a "
                f"batch of {window.shape[0]}."
            )
    return window, observed, scale_factor


# --- schedule --------------------------------------------------------------

def _wsd_lambda(
    max_steps: int,
    warmup_fraction: float,
    decay_fraction: float,
    min_ratio: float,
) -> Callable[[int], float]:
    """Warmup, stable, decay multiplier.

    Linear to the peak over the warmup span, held there, then down to the floor
    as ``1 - sqrt(progress)``: a sharp initial drop and a shallow tail. All
    three spans are fractions of ``max_steps``, which is why ``max_steps``
    reshapes a run instead of truncating one.
    """
    warmup_steps = max(1, int(max_steps * warmup_fraction))
    decay_steps = max(1, int(max_steps * decay_fraction))
    decay_start = max(warmup_steps + 1, max_steps - decay_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if step < decay_start:
            return 1.0
        span = max(1, max_steps - decay_start)
        progress = min(1.0, (step - decay_start) / span)
        return min_ratio + (1.0 - min_ratio) * (1.0 - math.sqrt(progress))

    return lr_lambda


# --- the rollout -----------------------------------------------------------

def _chunk_loss(
    backbone: torch.nn.Module,
    context: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    scale_factor: torch.Tensor,
    quantiles: torch.Tensor,
    commit_weight: float,
    commit_gated: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One block: encode the context, score the block, hand back its median.

    Returns the scalar block loss and the raw-magnitude median forecast the
    next block feeds on.
    """
    horizon = target.shape[1]
    y_norm, x_min, x_range = backbone.encode(
        context, batch_first=True, scale_factor=scale_factor, horizon=horizon,
    )
    y_norm = y_norm.clamp(-_PRED_CLAMP, _PRED_CLAMP)
    x_min_b = x_min.squeeze(-1)                                  # (B, 1)
    x_range_b = x_range.squeeze(-1)                              # (B, 1)
    target_norm = ((target - x_min_b) / x_range_b).clamp(
        -_TARGET_CLAMP, _TARGET_CLAMP
    )

    # A sample is scored only if its context has range to normalize by, its
    # forecast is finite, and it has an observed target position.
    non_batch = tuple(range(1, y_norm.dim()))
    valid = (
        (x_range_b.squeeze(-1) > _MIN_RANGE)
        & torch.isfinite(y_norm).all(dim=non_batch)
        & (target_mask.sum(dim=1) > 0)
    ).to(y_norm.dtype)

    per_sample = pinball_loss(
        y_norm, target_norm, quantiles, target_mask, reduction="none",
    )
    per_sample = torch.nan_to_num(per_sample, 0.0, 0.0, 0.0) * valid

    q_mid = y_norm.shape[-1] // 2
    median_norm = y_norm[..., q_mid]                             # (B, H)

    if commit_weight > 0.0:
        copy_raw = seasonal_copy_baseline(context, horizon, scale_factor)
        copy_norm = (copy_raw - x_min_b) / x_range_b
        commit = committing_loss(
            median_norm, target_norm, copy_norm, target_mask,
            weight=commit_weight, gated=commit_gated, reduction="none",
        )
        per_sample = per_sample + torch.nan_to_num(commit, 0.0, 0.0, 0.0) * valid

    n_valid = valid.sum()
    if float(n_valid) < 1.0:
        loss = per_sample.sum() * 0.0
    else:
        loss = per_sample.sum() / n_valid
    if not torch.isfinite(loss):
        loss = per_sample.sum() * 0.0

    median_raw = median_norm * x_range_b + x_min_b
    return loss, median_raw


def _rollout_loss(
    backbone: torch.nn.Module,
    window: torch.Tensor,
    observed: torch.Tensor,
    scale_factor: torch.Tensor,
    *,
    seq_len: int,
    horizon_unit: int,
    chunks: int,
    epsilon: float,
    quantiles: torch.Tensor,
    commit_weight: float,
    commit_gated: bool,
) -> torch.Tensor:
    """Roll out ``chunks`` blocks under scheduled sampling and average them.

    Each block re-normalizes its own context window. Between blocks the model's
    own median replaces the true values with probability ``epsilon``, so the
    context the model sees late in the rollout is built the way it will be at
    inference. Fed values are detached: they are inputs, not a gradient path.

    An unobserved target position is always replaced by the median, whatever
    ``epsilon`` says, rather than by the zero its mask stands for. On gap-free
    data this makes no difference; on gappy data it keeps a filler value out of
    the next context.
    """
    length, unit = int(seq_len), int(horizon_unit)
    series = window
    total = window.new_zeros(())
    for k in range(chunks):
        context = series[:, k * unit: k * unit + length]
        start = length + k * unit
        stop = start + unit
        target = window[:, start:stop]
        target_mask = observed[:, start:stop]
        loss_k, median_raw = _chunk_loss(
            backbone, context, target, target_mask, scale_factor,
            quantiles, commit_weight, commit_gated,
        )
        total = total + loss_k
        if k < chunks - 1:
            use_pred = (
                torch.rand(window.shape[0], 1, device=window.device) < epsilon
            ) | (target_mask < 0.5)
            fed = torch.where(use_pred, median_raw, target)
            series = series.clone()
            series[:, start:stop] = fed.detach()
    return total / float(chunks)


# --- checkpoints -----------------------------------------------------------

def _save(model: torch.nn.Module, path: Path) -> None:
    """Write the model in the released format.

    ``safetensors`` stores each unique storage once, so the two weight-tied FFN
    stacks are written once and ``tinycast.load_checkpoint`` restores the
    sharing on load.
    """
    from safetensors.torch import save_model

    save_model(model, str(path))


# --- entry point -----------------------------------------------------------

def train(
    config: TinyCastConfig,
    *,
    data: Any,
    max_steps: int,
    output_dir: str,
    seed: int = 42,
    device: str = "cpu",
    batch_size: int,
    accumulate_grad_batches: int = 1,
    num_workers: int = 0,
    checkpoint_every: Optional[int] = None,
    callbacks: Sequence[Callable[[Dict[str, Any]], None]] = (),
    deterministic: bool = False,
) -> TrainResult:
    """Train a TinyCast model and return where it landed.

    Args:
        config: the architecture. Optional attributes named after the module
            constants (``ar_chunks``, ``learning_rate``, ``commit_weight`` and
            so on) override the shipped recipe; absent ones take it.
        data: a ``Dataset``, an ``IterableDataset``, a ``DataLoader``, or a
            plain iterable of samples. A sample is a mapping with a ``window``
            of ``L + K * p`` values and optionally a ``mask`` and a
            ``scale_factor``; a bare tensor or a positional triple works too.
            Use :func:`training_window_width` to size the window.
        max_steps: optimizer steps. This RESHAPES the run: the warmup, stable
            and decay spans of the schedule are fractions of it, and the
            scheduled-sampling probability ramps over its first half. A short
            run is not a prefix of a long one.
        output_dir: created if absent. Receives ``model.safetensors``,
            ``config.json`` and any periodic checkpoints.
        seed: seeds Python, NumPy, torch, the sampler and the workers, inside
            this call.
        device: ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``. An unavailable
            accelerator raises rather than falling back.
        batch_size: samples per micro-batch. The effective batch is this times
            ``accumulate_grad_batches``.
        accumulate_grad_batches: micro-batches per optimizer step.
        num_workers: DataLoader workers. Zero, single-process, by default.
        checkpoint_every: write a checkpoint every this many optimizer steps.
        callbacks: called after each optimizer step with a dict carrying
            ``step``, ``loss``, ``lr``, ``epsilon`` and ``grad_norm``.
        deterministic: request deterministic kernels for the duration of the
            call, then restore the process flags. An op with no deterministic
            implementation warns rather than raising, so this tightens
            reproducibility without ruling a device out.

    Returns:
        A :class:`TrainResult`.

    Nothing here reaches the network and nothing is uploaded.
    """
    if int(max_steps) < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}.")
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
    if int(accumulate_grad_batches) < 1:
        raise ValueError(
            "accumulate_grad_batches must be at least 1, got "
            f"{accumulate_grad_batches}."
        )
    max_steps = int(max_steps)
    accumulate_grad_batches = int(accumulate_grad_batches)

    chunks = int(_setting(config, "ar_chunks", AR_CHUNKS))
    width = training_window_width(config, chunks)
    seq_len = int(config.seq_len)
    horizon_unit = int(config.output_token_len)
    eps_max = float(_setting(config, "scheduled_sampling_max", SCHEDULED_SAMPLING_MAX))
    peak_lr = float(_setting(config, "learning_rate", LEARNING_RATE))
    min_lr = float(_setting(config, "min_learning_rate", MIN_LEARNING_RATE))
    warmup_fraction = float(_setting(config, "warmup_fraction", WARMUP_FRACTION))
    decay_fraction = float(_setting(config, "decay_fraction", DECAY_FRACTION))
    weight_decay = float(_setting(config, "weight_decay", WEIGHT_DECAY))
    grad_clip = float(_setting(config, "grad_clip", GRAD_CLIP))
    commit_weight = float(_setting(config, "commit_weight", COMMIT_WEIGHT))
    commit_gated = bool(_setting(config, "commit_gated", COMMIT_GATED))
    if warmup_fraction + decay_fraction > 1.0:
        raise ValueError(
            "warmup_fraction + decay_fraction must not exceed 1.0, got "
            f"{warmup_fraction} + {decay_fraction}."
        )

    torch_device = _resolve_device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with _Determinism(deterministic, torch_device):
        _seed_everything(int(seed))

        model = TinyCastForPrediction(config).to(torch_device)
        model.train()
        backbone = model.model
        param_dtype = next(model.parameters()).dtype
        quantiles = torch.tensor(
            list(config.quantiles), dtype=param_dtype, device=torch_device,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=peak_lr,
            betas=ADAM_BETAS,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            _wsd_lambda(max_steps, warmup_fraction, decay_fraction,
                        min_lr / peak_lr),
        )

        loader = _build_loader(
            data, int(batch_size), int(num_workers), torch_device, int(seed),
        )
        batches = _epochs(loader)

        losses: List[float] = []
        learning_rates: List[float] = []
        written: List[str] = []

        for step in range(max_steps):
            # The feedback probability ramps to its maximum over the first half
            # of the run, so it too is a function of max_steps.
            epsilon = eps_max * min(1.0, step / max(1.0, 0.5 * max_steps))
            lr = float(scheduler.get_last_lr()[0])

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for _ in range(accumulate_grad_batches):
                window, observed, scale_factor = _unpack_batch(
                    next(batches), width, torch_device, param_dtype,
                )
                loss = _rollout_loss(
                    backbone, window, observed, scale_factor,
                    seq_len=seq_len,
                    horizon_unit=horizon_unit,
                    chunks=chunks,
                    epsilon=epsilon,
                    quantiles=quantiles,
                    commit_weight=commit_weight,
                    commit_gated=commit_gated,
                )
                (loss / accumulate_grad_batches).backward()
                step_loss += float(loss.detach()) / accumulate_grad_batches

            if grad_clip > 0.0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                )
            else:
                grad_norm = float("nan")
            optimizer.step()
            scheduler.step()

            losses.append(step_loss)
            learning_rates.append(lr)
            record = {
                "step": step + 1,
                "loss": step_loss,
                "lr": lr,
                "epsilon": epsilon,
                "grad_norm": grad_norm,
            }
            for callback in callbacks:
                callback(record)

            if checkpoint_every and (step + 1) % int(checkpoint_every) == 0:
                path = out / f"step-{step + 1:06d}.safetensors"
                _save(model, path)
                written.append(str(path))

        final = out / "model.safetensors"
        _save(model, final)
        with open(out / "config.json", "w") as handle:
            json.dump(config.to_dict(), handle, indent=2)

    return TrainResult(
        checkpoint_path=str(final),
        steps=max_steps,
        losses=losses,
        learning_rates=learning_rates,
        checkpoints=written,
        window_width=width,
    )


__all__ = ["TrainResult", "train", "training_window_width"]

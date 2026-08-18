"""Training checkpoint to released artifact.

The released model is a ``model.safetensors`` + ``config.json`` pair; this module
is the bridge that produces it from whatever a training run left on disk.

Two things make that bridge non-trivial:

* **Weight tying.** Both FFN stacks are shared, so a naive ``state_dict()`` has 177
  entries summing to 321,225 values while the model has only 121 distinct
  parameter tensors holding 146,505 values. The export writes each storage once and
  records the aliases in the safetensors metadata, which is what
  :func:`tinycast.checkpoint.load_checkpoint` reads back. A parameter count is only
  meaningful when taken from a model instantiated from its config; never sum a
  checkpoint's tensors.
* **Checkpoint averaging.** The released weights are the uniform mean of the last
  eight periodic checkpoints of the training run, so exporting the final checkpoint
  alone does not reproduce them. :func:`average_checkpoints` is that mean.

Loading is deliberately restricted: torch-serialized files are read with
``weights_only=True`` and safetensors files with the safetensors reader, so no path
handed to this module can execute code.

Usage:
    # the last eight periodic checkpoints, in training order
    python -m tinycast.export --out-dir release/ --average CKPT [CKPT ...]

    >>> from tinycast.export import export_safetensors, check_export_roundtrip
    >>> export_safetensors(model, "release/")
    >>> check_export_roundtrip(model, expect_parameters=146_505)
"""
from __future__ import annotations

import json
import math
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .config import TinyCastConfig
from .model import TinyCastForPrediction

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"

#: Number of periodic checkpoints averaged to produce the released weights.
RELEASE_AVERAGE_N = 8

#: Keys a training checkpoint may nest its tensors under, most specific first.
_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "weights")

StateDict = Dict[str, torch.Tensor]


class ExportError(RuntimeError):
    """A checkpoint could not be read, averaged, or exported faithfully."""


# ---------------------------------------------------------------------------
# Safe loading
# ---------------------------------------------------------------------------
def load_state_dict_safely(path: Union[str, Path]) -> StateDict:
    """Read a state dict from ``path`` without ever executing code from it.

    ``.safetensors`` goes through the safetensors reader; everything else through
    ``torch.load(..., weights_only=True)``. A training checkpoint that nests its
    tensors under ``state_dict`` (or ``model_state_dict`` / ``model`` / ``weights``)
    is unwrapped; the optimizer state, schedulers and step counters around it are
    discarded, since none of them survive into the released artifact.
    """
    p = Path(path)
    if not p.is_file():
        raise ExportError(f"no such checkpoint: {p}")

    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(p), device="cpu"))

    obj = torch.load(str(p), map_location="cpu", weights_only=True)
    return _unwrap_state_dict(obj, p)


def _unwrap_state_dict(obj: object, origin: Path) -> StateDict:
    if not isinstance(obj, Mapping):
        raise ExportError(
            f"{origin}: expected a state dict or a checkpoint mapping, got "
            f"{type(obj).__name__}"
        )
    if _is_tensor_mapping(obj):
        return dict(obj)
    for key in _STATE_DICT_KEYS:
        inner = obj.get(key)
        if isinstance(inner, Mapping) and _is_tensor_mapping(inner):
            return dict(inner)
    raise ExportError(
        f"{origin}: no tensor mapping found at the top level or under any of "
        f"{_STATE_DICT_KEYS}"
    )


def _is_tensor_mapping(obj: Mapping) -> bool:
    return bool(obj) and all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in obj.items()
    )


# ---------------------------------------------------------------------------
# Checkpoint averaging
# ---------------------------------------------------------------------------
def average_checkpoints(paths: Sequence[Union[str, Path]]) -> StateDict:
    """Uniform mean of several checkpoints, returned as a state dict.

    This is how the released weights were made: the last
    :data:`RELEASE_AVERAGE_N` periodic checkpoints of the training run, averaged
    with equal weight. Floating-point tensors are accumulated in float32 in the
    order given and cast back to the source dtype; integer and boolean entries (step
    counters and the like) are taken from the last checkpoint in ``paths``, since a
    mean of those is meaningless. Pass the checkpoints in training order so "last"
    is the most recent one, and keep that order to reproduce the released weights
    exactly: an eight-term float32 sum is not associative, so reordering ``paths``
    moves the result by about one ulp.

    All checkpoints must carry the same keys with the same shapes; a mismatch is an
    error rather than a silent intersection, because a quietly dropped tensor would
    export as a randomly initialized one.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        raise ExportError("average_checkpoints needs at least one checkpoint")

    states = [load_state_dict_safely(p) for p in paths]
    reference = states[0]
    for path, state in zip(paths[1:], states[1:]):
        _assert_same_keys(reference, state, paths[0], path)

    n = len(states)
    averaged: StateDict = {}
    for key, ref in reference.items():
        tensors = [state[key] for state in states]
        if ref.is_floating_point():
            acc = tensors[0].to(torch.float32).clone()
            for t in tensors[1:]:
                acc += t.to(torch.float32)
            averaged[key] = (acc / n).to(ref.dtype)
        else:
            averaged[key] = tensors[-1].clone()
    return averaged


def _assert_same_keys(a: StateDict, b: StateDict, path_a: Path, path_b: Path) -> None:
    if a.keys() != b.keys():
        only_a = sorted(set(a) - set(b))[:5]
        only_b = sorted(set(b) - set(a))[:5]
        raise ExportError(
            f"checkpoint key mismatch between {path_a.name} and {path_b.name}: "
            f"only in the first {only_a}, only in the second {only_b}"
        )
    for key in a:
        if a[key].shape != b[key].shape:
            raise ExportError(
                f"shape mismatch for {key!r}: {tuple(a[key].shape)} in "
                f"{path_a.name} vs {tuple(b[key].shape)} in {path_b.name}"
            )


# ---------------------------------------------------------------------------
# Weight tying
# ---------------------------------------------------------------------------
def tied_parameter_groups(model: nn.Module) -> Dict[str, List[str]]:
    """Map each shared parameter's canonical name to the names aliasing it.

    Two names alias when they resolve to the same ``nn.Parameter`` object, which is
    what ``share_ffn`` produces. The canonical name is the one ``named_parameters``
    keeps, i.e. the first in registration order; only that one is written to the
    safetensors file.
    """
    canonical = {id(p): name for name, p in model.named_parameters()}
    groups: Dict[str, List[str]] = {name: [] for name in canonical.values()}
    for name, param in model.named_parameters(remove_duplicate=False):
        head = canonical[id(param)]
        if name != head:
            groups[head].append(name)
    return {head: alias for head, alias in groups.items() if alias}


def _assert_tied_values_agree(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Refuse a state dict whose tied entries disagree.

    Loading such a dict would keep whichever copy happened to be written last and
    discard the others without a word.
    """
    for head, aliases in tied_parameter_groups(model).items():
        if head not in state:
            continue
        for alias in aliases:
            if alias in state and not torch.equal(state[head], state[alias]):
                raise ExportError(
                    f"tied parameters disagree in the checkpoint: {alias!r} differs "
                    f"from {head!r}; the checkpoint does not come from this "
                    f"architecture"
                )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_safetensors(
    model_or_state_dict: Union[nn.Module, Mapping[str, torch.Tensor]],
    out_dir: Union[str, Path],
    config: Optional[TinyCastConfig] = None,
    expect_parameters: Optional[int] = None,
) -> Dict[str, object]:
    """Write the released ``model.safetensors`` + ``config.json`` pair to ``out_dir``.

    Accepts either a live :class:`~tinycast.model.TinyCastForPrediction` or a state
    dict from a training run (in which case ``config`` says which architecture to
    rebuild; it defaults to the released one). Tied FFN weights are written once
    each and their aliases recorded in the safetensors metadata, so the file holds
    the model's true 146,505-value footprint rather than the 321,225 values a
    ``state_dict`` reports.

    Pass ``expect_parameters`` to pin the count (146,505 for the released model);
    the check is against a model instantiated from ``config``, never against a sum
    of checkpoint tensors.

    Re-exporting the released weights reproduces the published ``config.json`` and
    the published ``model.safetensors`` tensor payload and tensor index byte for
    byte. The header's metadata block is the one exception: the safetensors writer
    emits those keys in an order of its own that varies between runs. The mapping
    they encode is stable and is what the loader reads, so a byte diff of the header
    is not a defect.

    Returns a report with the two paths, the tensor and parameter counts, and the
    number of aliases carried in the metadata.
    """
    from safetensors.torch import save_model

    model, config = _as_model(model_or_state_dict, config)
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters())
    if expect_parameters is not None and n_parameters != expect_parameters:
        raise ExportError(
            f"the model instantiated from this config has {n_parameters:,} "
            f"parameters, not the expected {expect_parameters:,}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights_path = out / WEIGHTS_NAME
    config_path = out / CONFIG_NAME

    # save_model can derive the alias map itself, but its choice of which name in a
    # sharing group to keep is its own. Seeding the map (save_model preserves
    # entries already present) makes the canonical name the one named_parameters
    # reports, and the map is verified against the written file below.
    alias_map = {
        alias: head
        for head, aliases in tied_parameter_groups(model).items()
        for alias in aliases
    }
    save_model(model, str(weights_path), metadata=dict(sorted(alias_map.items())))
    # No trailing newline: the released config.json has none, and a byte-for-byte
    # match lets a reader diff this export against the published artifact.
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    n_tensors, file_sum, written_aliases = _inspect_safetensors(weights_path)
    if file_sum != n_parameters:
        raise ExportError(
            f"{weights_path.name} holds {file_sum:,} values but the model has "
            f"{n_parameters:,} parameters; the tie deduplication is wrong"
        )
    if written_aliases != alias_map:
        raise ExportError(
            f"{weights_path.name} records {len(written_aliases)} tied aliases, "
            f"expected the {len(alias_map)} sharing groups of the model; loading it "
            f"would leave weights uninitialized"
        )

    return {
        "weights_path": weights_path,
        "config_path": config_path,
        "num_tensors": n_tensors,
        "num_parameters": n_parameters,
        "num_tied_aliases": len(written_aliases),
    }


def _as_model(
    model_or_state_dict: Union[nn.Module, Mapping[str, torch.Tensor]],
    config: Optional[TinyCastConfig],
) -> Tuple[TinyCastForPrediction, TinyCastConfig]:
    if isinstance(model_or_state_dict, nn.Module):
        model = model_or_state_dict
        config = config or getattr(model, "config", None)
        if config is None:
            raise ExportError("the model carries no config; pass config=...")
        return model, config

    if not isinstance(model_or_state_dict, Mapping):
        raise ExportError(
            "expected a TinyCast model or a state dict, got "
            f"{type(model_or_state_dict).__name__}"
        )

    config = config or TinyCastConfig()
    model = TinyCastForPrediction(config)
    state = dict(model_or_state_dict)
    _assert_tied_values_agree(model, state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ExportError(
            f"state dict does not match the architecture: {len(missing)} missing "
            f"{sorted(missing)[:5]}, {len(unexpected)} unexpected "
            f"{sorted(unexpected)[:5]}"
        )
    return model, config


def _inspect_safetensors(path: Path) -> Tuple[int, int, Dict[str, str]]:
    """Return (tensor count, summed values, alias map) read back from the header."""
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    metadata = header.pop("__metadata__", {}) or {}
    total = 0
    for spec in header.values():
        n = 1
        for dim in spec["shape"]:
            n *= dim
        total += n
    return len(header), total, metadata


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------
def fixed_batch(
    batch_size: int = 2, seq_len: int = 2048, seed: int = 0
) -> torch.Tensor:
    """A deterministic context batch: two seasonal components plus fixed noise.

    Seasonality is present on purpose, so the round trip exercises the period
    detector and the phase-folding path rather than the convolutions alone.
    """
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(seq_len, dtype=torch.float32).unsqueeze(0)
    offsets = torch.arange(batch_size, dtype=torch.float32).unsqueeze(1)
    daily = torch.sin(2 * math.pi * (t + 7 * offsets) / 24.0)
    weekly = 0.3 * torch.sin(2 * math.pi * t / 168.0)
    noise = 0.05 * torch.randn(batch_size, seq_len, generator=generator)
    return 10.0 + 3.0 * daily + weekly + noise


@torch.no_grad()
def check_export_roundtrip(
    model: nn.Module,
    out_dir: Optional[Union[str, Path]] = None,
    config: Optional[TinyCastConfig] = None,
    expect_parameters: Optional[int] = None,
    prediction_length: int = 48,
) -> Dict[str, object]:
    """Export ``model``, load it back, and verify nothing was lost.

    Three things are checked. The reloaded model must produce bitwise-identical
    quantiles on a fixed batch. The FFN tie must be restored as object identity and
    not merely as equal values, since equal-but-separate tensors would triple the
    deployed footprint. And the file's tensors must sum to the instantiated
    parameter count rather than to the inflated ``state_dict`` sum.

    ``out_dir`` defaults to a temporary directory. Pass ``expect_parameters``
    (146,505 for the released model) to pin the count. Returns a report; every
    failure raises :class:`ExportError`.
    """
    from .checkpoint import load_checkpoint

    scratch = tempfile.TemporaryDirectory() if out_dir is None else nullcontext(out_dir)
    with scratch as tmp:
        target = Path(tmp)
        report = export_safetensors(
            model, target, config=config, expect_parameters=expect_parameters
        )

        model.eval()
        x = fixed_batch(seq_len=int(model.config.seq_len))
        reference = model(
            past_values=x, scale_factor=1.0, prediction_length=prediction_length,
            batch_first=True,
        ).quantile_outputs

        reloaded, _ = load_checkpoint(str(report["weights_path"]))
        reloaded.eval()
        replayed = reloaded(
            past_values=x, scale_factor=1.0, prediction_length=prediction_length,
            batch_first=True,
        ).quantile_outputs

    if replayed.shape != reference.shape:
        raise ExportError(
            f"reloaded model returns {tuple(replayed.shape)}, expected "
            f"{tuple(reference.shape)}"
        )
    if not torch.equal(reference, replayed):
        gap = (reference - replayed).abs().max().item()
        raise ExportError(
            f"reloaded model is not bitwise identical: max absolute difference {gap}"
        )

    expected_groups = tied_parameter_groups(model)
    restored_groups = tied_parameter_groups(reloaded)
    if restored_groups != expected_groups:
        raise ExportError(
            "the FFN weight tie was not restored on load: "
            f"{sum(len(a) for a in restored_groups.values())} aliases share storage, "
            f"expected {sum(len(a) for a in expected_groups.values())}"
        )

    naive_sum = sum(t.numel() for t in reloaded.state_dict().values())
    if report["num_parameters"] >= naive_sum:
        raise ExportError(
            f"the export did not deduplicate: {report['num_parameters']:,} values "
            f"written against a state dict sum of {naive_sum:,}"
        )

    report = dict(report)
    report["state_dict_sum"] = naive_sum
    report["bitwise_identical"] = True
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tinycast.export",
        description="Convert training checkpoints into the released "
                    "model.safetensors + config.json pair.",
    )
    parser.add_argument(
        "--average", nargs="+", metavar="CKPT", required=True,
        help=f"checkpoints to average, in training order (the release used "
             f"the last {RELEASE_AVERAGE_N})",
    )
    parser.add_argument("--out-dir", required=True, help="directory to write into")
    parser.add_argument(
        "--config", default=None,
        help="config.json to build the architecture from; defaults to the "
             "released configuration",
    )
    parser.add_argument(
        "--expect-parameters", type=int, default=None,
        help="fail unless the instantiated model has this many parameters "
             "(146505 for the released model)",
    )
    parser.add_argument(
        "--no-check", action="store_true",
        help="skip the reload / bitwise-identity round trip",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = TinyCastConfig()
    if args.config:
        with open(args.config) as f:
            raw = json.load(f)
        config = TinyCastConfig(**{
            k: v for k, v in raw.items() if k in TinyCastConfig.__dataclass_fields__
        })

    if len(args.average) != RELEASE_AVERAGE_N:
        print(
            f"[export] averaging {len(args.average)} checkpoints; the released "
            f"weights used {RELEASE_AVERAGE_N}",
            flush=True,
        )
    state = average_checkpoints(args.average)
    model, config = _as_model(state, config)

    if args.no_check:
        report = export_safetensors(
            model, args.out_dir, config=config,
            expect_parameters=args.expect_parameters,
        )
    else:
        report = check_export_roundtrip(
            model, args.out_dir, config=config,
            expect_parameters=args.expect_parameters,
        )

    print(
        f"[export] wrote {report['weights_path']} "
        f"({report['num_tensors']} tensors, {report['num_parameters']:,} "
        f"parameters, {report['num_tied_aliases']} tied aliases)",
        flush=True,
    )
    print(f"[export] wrote {report['config_path']}", flush=True)
    if not args.no_check:
        print(
            f"[export] round trip: bitwise identical, tie restored, "
            f"{report['num_parameters']:,} values written against a state dict "
            f"sum of {report['state_dict_sum']:,}",
            flush=True,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

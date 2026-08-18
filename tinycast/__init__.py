"""TinyCast: an attention-free, 146,505-parameter dilated-convolution
time-series foundation model.

Forecast:
    >>> from tinycast import TinyCastPredictor, load_checkpoint
    >>> predictor = TinyCastPredictor(
    ...     prediction_length=48,
    ...     checkpoint_path="model.safetensors",
    ...     freq="H", domain="Energy", device="cpu",
    ...     force_flip_invariance=True,
    ... )
    >>> forecasts = predictor.predict(gluonts_test_input)

Evaluate:
    >>> from tinycast import summarize_by_freq_bin
    >>> summarize_by_freq_bin("all_results.csv")["overall"]["ncrps"]

Train, then release:
    >>> from tinycast import (TinyCastConfig, train, average_checkpoints,
    ...                       export_safetensors)
    >>> result = train(TinyCastConfig(), data=windows, max_steps=1000,
    ...                output_dir="run/", batch_size=32, checkpoint_every=100)
    >>> weights = average_checkpoints(result.checkpoints[-8:])
    >>> export_safetensors(weights, "release/", expect_parameters=146_505)

Rebuild the synthetic corpus (needs CUDA):
    >>> from tinycast import build_shard, verify_shard
    >>> build_shard()                       # published shard 0
    >>> verify_shard("synth4096_0", shard=0)

``tinycast.train`` is the training function, not the module: the two share a
name and the function wins. Module-level recipe constants are reachable as
``from tinycast.train import AR_CHUNKS``. A submodule the package does not
import itself, ``tinycast.backbone`` and ``tinycast.periodogram`` among them,
becomes an attribute only after ``import tinycast.backbone``.
"""

from .config import TinyCastConfig
from .model import TinyCastForPrediction, TinyCastBackbone, PredictionOutput
from .checkpoint import load_checkpoint, load_model
from .predictor import TinyCastPredictor, ARRolloutPredictor

# Training and the objectives it optimizes.
from .losses import committing_loss, pinball_loss, seasonal_copy_baseline
from .train import TrainResult, train, training_window_width

# The synthetic pretraining corpus generators.
from .synth import generate_gp, generate_spikes, generate_tsi

# eval, export and corpus each carry a ``python -m`` entry point, so the package
# must not import them eagerly: that puts them in sys.modules before runpy runs
# them as __main__, which warns and executes the module body twice. Resolving
# them on first attribute access (PEP 562) keeps ``from tinycast import
# evaluate`` working and leaves the command line quiet.
_LAZY_MODULES = ("eval", "export", "corpus")
_LAZY_EXPORTS = {
    "evaluate": "eval",
    "summarize_by_freq_bin": "eval",
    "export_safetensors": "export",
    "average_checkpoints": "export",
    "check_export_roundtrip": "export",
    "ExportError": "export",
    "build_shard": "corpus",
    "verify_shard": "corpus",
    "iter_shard_series": "corpus",
}


def __getattr__(name: str):
    from importlib import import_module

    if name in _LAZY_MODULES:
        value = import_module(f".{name}", __name__)
    else:
        module = _LAZY_EXPORTS.get(name)
        if module is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value          # resolve once, then it is a plain global
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(_LAZY_MODULES))


__all__ = [
    # model and inference
    "TinyCastConfig",
    "TinyCastForPrediction",
    "TinyCastBackbone",
    "PredictionOutput",
    "load_checkpoint",
    "load_model",
    "TinyCastPredictor",
    "ARRolloutPredictor",
    # training
    "train",
    "TrainResult",
    "training_window_width",
    "pinball_loss",
    "committing_loss",
    "seasonal_copy_baseline",
    # export
    "export_safetensors",
    "average_checkpoints",
    "check_export_roundtrip",
    "ExportError",
    # evaluation
    "evaluate",
    "summarize_by_freq_bin",
    # synthetic corpus
    "build_shard",
    "verify_shard",
    "iter_shard_series",
    "generate_gp",
    "generate_spikes",
    "generate_tsi",
]

__version__ = "1.0.0"

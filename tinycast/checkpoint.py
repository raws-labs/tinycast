"""Weight loading for TinyCast.

The released weights are a ``model.safetensors`` + a ``config.json`` (the
architecture config). ``load_checkpoint`` (aliased ``load_model``) handles the weight-tied
FFN sharing: the file stores each unique parameter storage once (the true
146,505-parameter footprint, ~0.6 MB) and restores the sharing on load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from .config import TinyCastConfig
from .model import TinyCastForPrediction


def load_checkpoint(
    weights_path: str,
    config_path: Optional[str] = None,
) -> Tuple[TinyCastForPrediction, TinyCastConfig]:
    """Load TinyCastForPrediction from ``model.safetensors`` + ``config.json``.

    ``config_path`` defaults to a sibling ``config.json`` next to the weights.
    """
    from safetensors.torch import load_model

    st = Path(weights_path)
    if config_path is None:
        config_path = st.parent / "config.json"
    with open(config_path) as f:
        cfg_dict = json.load(f)
    cfg = TinyCastConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in TinyCastConfig.__dataclass_fields__
    })
    model = TinyCastForPrediction(cfg)
    load_model(model, str(st))
    model.eval()
    return model, cfg


# Explicit alias used by the notebook / README examples.
load_model = load_checkpoint

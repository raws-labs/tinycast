"""TinyCast model configuration.

Only the fields that shape the deployed model's construction / forward are
kept. Field defaults are the released model's values, so ``TinyCastConfig()`` alone
reconstructs the deployed architecture; loading from ``config.json`` overrides
them.
"""

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class TinyCastConfig:
    """Configuration for the deployed TinyCast (dilated-conv) model."""

    # --- input / output geometry -------------------------------------------
    quantiles: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    seq_len: int = 2048                   # L: encoder context window
    output_token_len: int = 48            # p: single-shot / AR-chunk horizon

    # --- period detector (shared structural prior) -------------------------
    top_k_periods: int = 4
    significance_alpha: float = 0.05
    n_harmonics: int = 1

    # --- dilated-conv backbone ---------------------------------------------
    conv_dim: int = 64
    n_layers: int = 10
    kernel_size: int = 3
    ffn_mult: float = 1.0
    pool_kind: str = "mean_last"
    causal: bool = True
    phase_bins: int = 16
    decoder_depth: int = 1
    separable_conv: bool = True
    share_ffn: bool = True
    future_conv: bool = True
    future_conv_layers: int = 6
    future_conv_seed: int = 128

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)

    def to_dict(self) -> dict:
        return asdict(self)

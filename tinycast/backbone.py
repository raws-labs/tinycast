"""Dilated-convolution forecasting backbone.

An attention-free encoder: a stack of causal dilated convolutions. With
kernel K=3 and dilations d_i = 2^(i-1) for i=1..N the receptive field is

    RF = 1 + 2 * sum_i d_i = 1 + 2 * (2^N - 1)

so N=10 layers cover RF=2047, sufficient for the L=2048 context with zero
downsampling and no information loss at any time scale. Native multi-scale
via the dilation schedule; deployment-friendly (no L^2 attention matrix, no
softmax, pure matmul + element-wise; quantizes cleanly to INT8); streaming-
friendly (left-only causal padding).

Structural priors (zero-parameter):
  - a normalized-periodogram period detector driving a phase encoding
  - bounded recency basis (signed-linear/log, multi-scale exp decay)
  - position-parameterized decoder queries (single-shot arbitrary horizon)
  - no autoregressive rollout inside the backbone
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .periodogram import significant_periods
from .encoding import (
    N_RECENCY_CHANNELS,
    _norm_fp32,
    _phase_encoding,
    _positional_encoding,
)


class _SwiGLU(nn.Module):
    """Standard SwiGLU FFN."""

    def __init__(self, d: int, d_hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(d, 2 * d_hidden)
        self.down = nn.Linear(d_hidden, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, val = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * val)


class _DilatedConvBlock(nn.Module):
    """Dilated Conv1d → RMSNorm → SwiGLU → RMSNorm with residuals.

    ``causal=True``, which is what the released config sets, pads all (K-1)*d
    timesteps on the left, so each output position sees only its own past. With
    ``causal=False`` the padding is centered instead: length is still preserved
    and each position gets a symmetric view of (K-1)*d/2 timesteps on either
    side, at the cost of future-side context. Under both, the dilation scales
    the per-layer receptive field without adding parameters.
    """

    def __init__(
        self, d: int, kernel: int = 3, dilation: int = 1,
        ffn_mult: float = 1.5, causal: bool = False, gated: bool = False,
        separable: bool = False,
    ) -> None:
        super().__init__()
        self.k = int(kernel)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.gated = bool(gated)
        self.separable = bool(separable)
        # Conv with dilation; padding handled in forward. Centered padding
        # gives each position a symmetric view but feeds the right edge
        # FUTURE-side zeros: at the deepest layer the last (most recent)
        # position's representation is dominated by padding standing in for
        # the unknown forecast, and the model learns a train/inference
        # mismatch. Causal (all-left) padding removes both pathologies and
        # is a prerequisite for honest streaming inference.
        if self.separable:
            # Depthwise-separable factorization: depthwise (per-channel, dilated,
            # padding consumed in forward) + pointwise 1x1 (channel mix). Params
            # D*K + D*D against a full conv's D*D*K: the receptive field is
            # preserved at a fraction of the parameters per block. Padding
            # before self.conv feeds the depthwise stage.
            self.conv = nn.Sequential(
                nn.Conv1d(d, d, kernel_size=self.k, dilation=self.dilation, groups=d),
                nn.Conv1d(d, d, kernel_size=1),
            )
        else:
            self.conv = nn.Conv1d(d, d, kernel_size=self.k, dilation=self.dilation)
        # Lightweight gated conv (multiplicative gating): a cheap
        # DEPTHWISE gate conv produces a sigmoid mask over the main conv output,
        # conv_out * σ(gate). Adds data-dependent gating to the encoder at ~D·k
        # params/block (vs doubling the full conv). Off by default.
        self.gate = (
            nn.Conv1d(d, d, kernel_size=self.k, dilation=self.dilation, groups=d)
            if self.gated else None
        )
        self.norm1 = nn.RMSNorm(d)
        d_hidden = int(d * ffn_mult)
        self.ffn = _SwiGLU(d, d_hidden)
        self.norm2 = nn.RMSNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D). Conv1d takes (B, D, L).
        x_t = x.transpose(1, 2)
        # Effective kernel span = (K-1)*d + 1.
        pad_total = (self.k - 1) * self.dilation
        if self.causal:
            left, right = pad_total, 0
        else:
            left = pad_total // 2
            right = pad_total - left
        x_t = F.pad(x_t, (left, right))
        if self.separable and self.gate is None:
            # Depthwise (B,D,L); then the pointwise 1x1 conv == a per-position
            # Linear, run as F.linear (cublas GEMM) instead of an im2col/cudnn
            # 1x1-conv kernel. Bit-identical channel contraction; params stay as
            # conv.0/conv.1 so existing checkpoints load unchanged.
            dw = self.conv[0](x_t).transpose(1, 2)                  # (B, L, D)
            pw = self.conv[1]
            conv_out = F.linear(dw, pw.weight.squeeze(-1), pw.bias)  # (B, L, D)
        else:
            conv_out = self.conv(x_t)
            if self.gate is not None:
                conv_out = conv_out * torch.sigmoid(self.gate(x_t))
            conv_out = conv_out.transpose(1, 2)
        x = _norm_fp32(self.norm1, x + conv_out)
        x = _norm_fp32(self.norm2, x + self.ffn(x))
        return x


class DilatedConvBackbone(nn.Module):
    """Dilated-conv encoder with a phase-conditioned, position-parameterized decoder.

    Args:
        seq_len:        L, the context length.
        p_out:          H, the single-shot output length.
        n_quantiles:    Q, the number of output channels.
        d:              channel dimension.
        n_layers:       number of dilated-conv blocks (10 → RF 2047 ≥ L=2048).
        kernel:         conv kernel size (default 3).
        ffn_mult:       SwiGLU hidden multiplier (default 1.5).
        dilations:      explicit dilation schedule; if None, uses 2^(i-1).
        top_k_periods:  the periodogram detector's top-K (default 4).
        significance_alpha:   the periodogram detector's Bonferroni alpha (default 0.05).
        n_harmonics:    Fourier harmonics per detected period (default 1).
        pool_kind:      context-summary pooling: "mean_last" (default),
                        "mean", "last". Concat the chosen pool(s) into the
                        per-horizon query before query_proj.
        causal:         if True, all conv padding is left-only (no future
                        leakage, clean right edge, streaming-honest).
        phase_bins:     if > 0, augment the global pool with a period-folded
                        seasonal profile: for each period the periodogram
                        detector returns, fold the encoder output into this many
                        phase bins and let each decoder query gather the bin
                        matching its own phase. 0 disables phase folding and
                        leaves the plain global pool, which is the control the
                        paper's phase-folding ablation is measured against.
    """

    def __init__(
        self,
        seq_len: int,
        p_out: int,
        n_quantiles: int = 1,
        d: int = 80,
        n_layers: int = 10,
        kernel: int = 3,
        ffn_mult: float = 1.5,
        dilations: list[int] | None = None,
        top_k_periods: int = 4,
        significance_alpha: float = 0.05,
        n_harmonics: int = 1,
        pool_kind: str = "mean_last",
        causal: bool = False,
        phase_bins: int = 0,
        phase_stats: str = "mean",
        phase_recency_tau: float = 0.0,
        recency_bins: int = 0,
        sig_gate: bool = False,
        cross_cycle: bool = False,
        decoder_depth: int = 1,
        horizon_kernel: int = 0,
        horizon_recurrence: bool = False,
        min_cycles: int = 0,
        period_trust: str = "off",
        gated_conv: bool = False,
        residual_naive: bool = False,
        residual_multi: bool = False,
        residual_trend: bool = False,
        decompose_kernel: int = 0,
        periodogram_off: bool = False,
        res_adaptive: bool = False,
        res_period_target: int = 64,
        res_r_max: float = 32.0,
        with_missing: bool = False,
        missing_channel: bool = False,
        separable_conv: bool = False,
        share_ffn: bool = False,
        future_conv: bool = False,
        future_conv_layers: int = 6,
        future_conv_seed: int = 128,
        base_seasonality: float = 24.0,
        local_anchor: bool = False,
    ) -> None:
        super().__init__()
        self.L = int(seq_len)
        self.p_out = int(p_out)
        self.n_quantiles = int(n_quantiles)
        self.D = int(d)
        self.K = int(top_k_periods)
        self.significance_alpha = float(significance_alpha)
        self.n_harmonics = int(n_harmonics)
        self.pool_kind = str(pool_kind)
        self.causal = bool(causal)
        self.phase_bins = int(phase_bins)
        if phase_stats not in ("mean", "mean_var"):
            raise ValueError(f"phase_stats={phase_stats!r}; expected 'mean'|'mean_var'.")
        self.phase_stats = str(phase_stats)
        self.stat_mult = 2 if self.phase_stats == "mean_var" else 1
        self.phase_recency_tau = float(phase_recency_tau)
        self.recency_bins = int(recency_bins)
        self.sig_gate = bool(sig_gate)
        self.cross_cycle = bool(cross_cycle)
        self.decoder_depth = max(1, int(decoder_depth))
        self.horizon_kernel = int(horizon_kernel)
        self.horizon_recurrence = bool(horizon_recurrence)
        self.min_cycles = int(min_cycles)
        if period_trust not in ("off", "coverage", "full"):
            raise ValueError(f"period_trust={period_trust!r}; expected off|coverage|full")
        self.period_trust = str(period_trust)
        # Per-period reliability weight w_k = sigmoid(linear([margin, ln
        # coverage])): a continuous down-weighting of long or weakly-supported
        # periods, whose crossover is learned rather than set. It covers the
        # same ground as the integer min_cycles cutoff, which stays available
        # and independent; both are off in the released config. "coverage" uses
        # ln(L/period) alone (min_cycles is its hard-threshold limit); "full"
        # adds the significance margin ln(s_k/t_alpha) from the detector's
        # periodogram scores.
        if self.period_trust != "off":
            n_feat = 1 if self.period_trust == "coverage" else 2
            self.pt = nn.Linear(n_feat, 1)
            with torch.no_grad():
                self.pt.weight.zero_(); self.pt.weight[0, 0] = 1.0  # cov (or margin) coeff = 1
                self.pt.bias.zero_()
            # data-determined Bonferroni threshold t_alpha (no tuned knob)
            n_fft = 1 << int(math.ceil(math.log2(max(2, self.L))))
            self._n_bins = max(2, n_fft // 2)
            self._t_alpha = math.log(self._n_bins / max(self.significance_alpha, 1e-12)) / self._n_bins
        else:
            self.pt = None
        self.gated_conv = bool(gated_conv)
        self.residual_naive = bool(residual_naive)
        self.residual_multi = bool(residual_multi)
        self.residual_trend = bool(residual_trend)
        self.periodogram_off = bool(periodogram_off)
        self.res_adaptive = bool(res_adaptive)
        self.res_period_target = int(res_period_target)
        self.res_r_max = float(res_r_max)
        # Series decomposition: moving-avg trend / seasonal split fed
        # as two input channels. 0 disables. Even kernel → +1 for centered.
        self.decompose_kernel = int(decompose_kernel)
        self.with_missing = bool(with_missing)
        # Missing-value channel: feed the encoder a binary observed-mask so it
        # can distinguish a genuinely-unobserved position from a real value (the
        # faithful treatment, vs mean-fill which conflates the two).
        self.missing_channel = bool(missing_channel)
        if self.missing_channel and bool(res_adaptive):
            raise NotImplementedError(
                "missing_channel + res_adaptive: the observed-mask is not warped "
                "through _resolution_adapt; not supported together."
            )
        # Recent-anchor channels: gradient-connected causal local level/scale so the
        # encoder can re-anchor amplitude under non-stationarity (the denorm origin
        # x_min is a frozen detached GLOBAL min, with no learned path to the
        # recent regime).
        self.local_anchor = bool(local_anchor)

        if self.n_harmonics < 1:
            raise ValueError(f"n_harmonics must be >= 1; got {self.n_harmonics}")
        if pool_kind not in ("mean_last", "mean", "last"):
            raise ValueError(
                f"pool_kind={pool_kind!r}; expected 'mean_last' | 'mean' | 'last'."
            )

        # Positional encoding: phase channels + bounded recency channels.
        n_phase = 2 * self.K * self.n_harmonics
        n_pe = n_phase + N_RECENCY_CHANNELS
        # Input channels: raw value, or [trend, seasonal] if decomposing.
        n_value_ch = 2 if self.decompose_kernel > 0 else 1
        if self.missing_channel:
            n_value_ch += 1                                          # +observed-mask
        if self.local_anchor:
            n_value_ch += 2                                          # +[local-scale residual, log local-scale]
        in_channels = n_value_ch + n_pe
        self.in_proj = nn.Linear(in_channels, self.D)
        if self.local_anchor:
            # zero-init the 2 anchor columns (last of the value channels) so the
            # model is baseline-equivalent at init and learns the anchor from zero.
            with torch.no_grad():
                self.in_proj.weight[:, n_value_ch - 2:n_value_ch].zero_()

        # Dilation schedule.
        if dilations is None:
            dilations = [2**i for i in range(int(n_layers))]
        if len(dilations) != int(n_layers):
            raise ValueError(
                f"dilations length {len(dilations)} != n_layers {n_layers}"
            )
        self.dilations = list(dilations)

        # Receptive field sanity (informational only; fails soft if RF < L).
        rf = 1 + (kernel - 1) * sum(self.dilations)
        self.receptive_field = rf

        self.encoder = nn.ModuleList([
            _DilatedConvBlock(
                self.D, kernel=int(kernel), dilation=int(d_i),
                ffn_mult=float(ffn_mult), causal=self.causal,
                gated=self.gated_conv, separable=bool(separable_conv),
            )
            for d_i in self.dilations
        ])
        # Cross-layer FFN weight sharing (weight-tied): the SwiGLU FFN is the
        # largest param bucket and is dilation-independent, so one shared FFN
        # across all blocks recovers ~(n_layers-1)/n_layers of FFN params. The
        # per-block dilated convs (which carry the receptive field) stay distinct.
        self.share_ffn = bool(share_ffn)
        if self.share_ffn and len(self.encoder) > 1:
            shared_ffn = self.encoder[0].ffn
            for blk in self.encoder[1:]:
                blk.ffn = shared_ffn

        # Pooled context summary dim depends on pool_kind.
        pool_dim = {"mean_last": 2 * self.D, "mean": self.D, "last": self.D}[
            self.pool_kind
        ]

        # Phase-binned seasonal profile: K period-folded profiles, each
        # gathered by the decoder query's own phase, then mixed to D. A global
        # mean pool averages every phase of a cycle into one vector, so nothing
        # that varies with phase survives it; folding by phase keeps the
        # per-cycle waveform and hands each query the part of the cycle it is
        # forecasting.
        # Phase profile mixer: K periods × n_bins × (mean[,var]) → D.
        if self.phase_bins > 0:
            self.phase_mix = nn.Linear(self.K * self.stat_mult * self.D, self.D)
        else:
            self.phase_mix = None

        # Recency profile mixer: rb log-distance bins × (mean[,var])
        # → D. Always-valid aperiodic content path. Flattened (not gathered).
        if self.recency_bins > 0:
            self.recency_mix = nn.Linear(
                self.recency_bins * self.stat_mult * self.D, self.D,
            )
        else:
            self.recency_mix = None

        # Cross-cycle conv branch: a depthwise conv across cycles
        # at fixed phase, applied to the dominant period's [n_cycles × n_bins]
        # fold. Adds one D-dim feature to the query. See _cross_cycle_profile.
        if self.cross_cycle:
            self.cc_bins = self.phase_bins if self.phase_bins > 0 else 16
            self.cc_cycles = 8          # most-recent N cycles folded; older clamped
            # Depthwise conv ACROSS the cycle axis (length cc_cycles) at fixed
            # phase: models how each phase evolves cycle-to-cycle.
            self.cc_conv = nn.Conv1d(
                self.D, self.D, kernel_size=3, padding=1, groups=self.D,
            )
            self.cc_mix = nn.Linear(self.D, self.D)
        else:
            self.cc_conv = None

        # Decoder query input: PE + pool [+ phase D] [+ recency D] [+ cc D].
        # With sig_gate, phase & recency are blended into a single D (not
        # concatenated), so they contribute D once, not 2·D.
        query_in = n_pe + pool_dim
        if self.sig_gate and self.phase_mix is not None and self.recency_mix is not None:
            query_in += self.D
        else:
            query_in += self.D if self.phase_mix is not None else 0
            query_in += self.D if self.recency_mix is not None else 0
        query_in += self.D if self.cross_cycle else 0
        self.query_proj = nn.Linear(query_in, self.D)
        d_hidden = int(self.D * float(ffn_mult))
        # Decoder: `decoder_depth` residual SwiGLU blocks (depth 1 is a single
        # block). The decoder's inputs are rich (phase/recency profiles), and
        # depth lets it process them.
        self.decoder_ffns = nn.ModuleList(
            [_SwiGLU(self.D, d_hidden) for _ in range(self.decoder_depth)]
        )
        self.decoder_norms = nn.ModuleList(
            [nn.RMSNorm(self.D) for _ in range(self.decoder_depth)]
        )
        # Cross-horizon coherence: a causal depthwise conv across the
        # horizon axis couples adjacent forecast steps (the cross-step mixing
        # lost when attention was dropped). Causal + fixed kernel preserves the
        # single-shot arbitrary-horizon property. horizon_kernel=0 disables.
        if self.horizon_kernel > 0:
            self.horizon_conv = nn.Conv1d(
                self.D, self.D, kernel_size=self.horizon_kernel, groups=self.D,
            )
            self.horizon_norm = nn.RMSNorm(self.D)
        else:
            self.horizon_conv = None
        # Horizon-recurrent decode-state: a gated diagonal recurrence over the
        # SHORT horizon axis, scanning the precomputed query features. It never
        # re-feeds predicted values, so the decoder stays single-shot rather
        # than autoregressive. An unbounded carried state couples step h to ALL
        # earlier steps (vs the fixed-span horizon_conv), the property that
        # makes a recurrent decoder horizon-invariant. hr_o is zero-init so the
        # block is identity at start and cannot regress the baseline.
        if self.horizon_recurrence:
            self.hr_z = nn.Linear(self.D, self.D)   # update gate
            self.hr_c = nn.Linear(self.D, self.D)   # candidate
            self.hr_o = nn.Linear(self.D, self.D)   # output proj (zero-init)
            nn.init.zeros_(self.hr_o.weight); nn.init.zeros_(self.hr_o.bias)
            self.hr_norm = nn.RMSNorm(self.D)
        else:
            self.hr_z = None
        self.out_proj = nn.Linear(self.D, self.n_quantiles)


        # Future-conv decoder (horizon-axis state evolution; the conv-native
        # analog of a missing-token decoder). The rest of the decoder queries a
        # STATIC pooled summary at every horizon position, which is why error
        # grows with horizon. future_conv runs a CAUSAL dilated conv over
        # [context-tail seed ++ seasonal-naive future fill], producing
        # per-future-position hidden states that EVOLVE along the horizon (each
        # future position is a causal-conv function of recent context + earlier
        # future), and injects them additively into the decoder query. The fill
        # is the dominant-period seasonal-naive continuation (it carries
        # periodic structure, so the conv evolves a real waveform forward
        # rather than zeros), which leaves the decoder predicting the residual
        # over a copy. ``fc_out`` is zero-init => EXACT baseline at start
        # (zero-init additive idiom), and disabling it restores that baseline.
        # This differs from the two cheaper readouts in the same position: a
        # recurrence over queries derived from the static summary adds no new
        # dynamics, and a phase gather only COPIES context profiles, whereas
        # this path RUNS the conv forward.
        self.future_conv = bool(future_conv)
        if self.future_conv:
            if res_adaptive:
                raise ValueError("future_conv is incompatible with res_adaptive")
            self.fc_seed = int(future_conv_seed)
            fc_in = 1 + n_pe                       # fill value + the same PE layout
            self.fc_in_proj = nn.Linear(fc_in, self.D)
            fc_dils = [2 ** i for i in range(int(future_conv_layers))]
            self.fc_blocks = nn.ModuleList([
                _DilatedConvBlock(
                    self.D, kernel=int(kernel), dilation=int(d_i),
                    ffn_mult=float(ffn_mult), causal=True,
                    separable=True,   # auxiliary module: keep it light (~51K add)
                )
                for d_i in fc_dils
            ])
            # Weight-tie the FFN across fc blocks: the convs carry the horizon
            # dynamics, and one shared FFN keeps the param add modest.
            shared = self.fc_blocks[0].ffn
            for blk in self.fc_blocks[1:]:
                blk.ffn = shared
            self.fc_out = nn.Linear(self.D, self.D)   # zero-init => exact baseline
            nn.init.zeros_(self.fc_out.weight)
            nn.init.zeros_(self.fc_out.bias)

        self.base_seasonality = float(base_seasonality)

    # ---- helpers ----------------------------------------------------------

    def _future_conv_states(
        self, h: torch.Tensor, fut_pe: torch.Tensor, fill: torch.Tensor,
    ) -> torch.Tensor:
        """Causal-conv continuation states at the H future positions.

        h: (B, L, D) encoder output; fut_pe: (B, H, n_pe); fill: (B, H) the
        seasonal-naive future continuation. Returns (B, H, D).
        """
        # self.L is static (asserted == L in forward), so using it instead of
        # the traced h.shape[1] keeps dynamo from graph-splitting on a symint
        # bound.
        ft = self.fc_in_proj(torch.cat([fill.unsqueeze(-1), fut_pe], dim=-1))  # (B,H,D)
        seed = h[:, -min(self.fc_seed, self.L):, :]               # (B, seed, D)
        z = torch.cat([seed, ft], dim=1)                          # (B, seed+H, D)
        for blk in self.fc_blocks:                                # causal: no future leak
            z = blk(z)
        return z[:, -ft.shape[1]:, :]                             # (B, H, D)

    @torch.compiler.disable()
    def _detect_periods(
        self, x_fp32: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the significance-filtered periodogram in fp32 (no grad).

        @torch.compiler.disable: the periodogram uses a complex rfft that
        Inductor cannot codegen: left inside the compiled graph it forces an
        eager fallback + graph break every forward, early in `forward`, blocking
        fusion of the whole conv encoder/decoder downstream. Disabling compile on
        this (no_grad, fp32, produces integer periods the rest only reads) makes a
        clean eager boundary: the FFT runs eager, everything after fuses. Output
        bit-identical (only where it compiles changes).

        Returns ``(periods, n_valid, scores)``: the integer periods (0 =
        rejected), the count of significant periods per sample, and the
        per-period periodogram scores. ``n_valid`` is the periodicity-strength
        signal the significance gate reads.
        """
        B, L = x_fp32.shape
        if self.periodogram_off:
            # Control: no period detection → phase encoding zeros out, phase
            # machinery is inert. Tests whether the conv backbone matches with
            # a phase-free (recency-only) decoder.
            z = torch.zeros(B, self.K, dtype=torch.long, device=x_fp32.device)
            return (z, torch.zeros(B, dtype=torch.long, device=x_fp32.device),
                    torch.zeros(B, self.K, device=x_fp32.device))
        with torch.no_grad():
            periods, scores, n_valid = significant_periods(
                x_fp32,
                min_period=2,
                max_period=L // 2,
                top_k=self.K,
                significance_alpha=self.significance_alpha,
            )
        periods = periods.long()
        if self.min_cycles > 0:
            # "Do no harm": only TRUST a period with >= min_cycles full
            # cycles in the window (period <= L/min_cycles). Periods too long
            # to be reliably estimated (e.g. an 8640-sample daily cycle in a
            # 2048 window) are zeroed → the significance gate routes those
            # series to the recency/local path instead of mis-locking phase.
            max_p = L // self.min_cycles
            keep = (periods > 0) & (periods <= max_p)
            periods = torch.where(keep, periods, torch.zeros_like(periods))
            n_valid = keep.sum(dim=1).long()
        return periods, n_valid.long(), scores.float()

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """Pool encoder output to a fixed-size summary.

        h: (B, L, D)
        Returns: (B, pool_dim)
        """
        if self.pool_kind == "mean":
            return h.mean(dim=1)
        if self.pool_kind == "last":
            return h[:, -1, :]
        # mean_last:
        return torch.cat([h.mean(dim=1), h[:, -1, :]], dim=-1)

    def _scatter_profile(
        self, h: torch.Tensor, bins: torch.Tensor, nb: int,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Weighted scatter-mean (+ optional per-bin variance) of h into nb bins.

        h:      (B, L, D)
        bins:   (B, L) int in [0, nb)
        weight: (B, L) non-negative, or None for uniform.
        Returns (B, nb, stat_mult·D): per-bin mean, then per-bin variance if
        ``phase_stats == 'mean_var'``. Empty bins → global mean
        (and zero variance).
        """
        B, L, D = h.shape
        oh = F.one_hot(bins, nb).to(h.dtype)                        # (B,L,nb)
        if weight is not None:
            oh = oh * weight.unsqueeze(-1)
        cnt = oh.sum(dim=1).unsqueeze(-1)                           # (B,nb,1)
        denom = cnt.clamp(min=1e-6)
        mean = torch.bmm(oh.transpose(1, 2), h) / denom            # (B,nb,D)
        gmean = h.mean(dim=1, keepdim=True)                        # (B,1,D)
        empty = cnt <= 0
        mean = torch.where(empty, gmean.expand(B, nb, D), mean)
        if self.phase_stats == "mean_var":
            sq = torch.bmm(oh.transpose(1, 2), h * h) / denom      # E[h²]
            var = (sq - mean * mean).clamp(min=0.0)
            var = torch.where(empty, torch.zeros_like(var), var)
            return torch.cat([mean, var], dim=-1)                  # (B,nb,2D)
        return mean                                                # (B,nb,D)

    def _recency_weight(self, L: int, device) -> torch.Tensor | None:
        """exp recency weight over context positions: recent positions weigh
        more, so the folded profile tracks the CURRENT regime's waveform rather
        than the window average. None if tau<=0 (uniform)."""
        if self.phase_recency_tau <= 0.0:
            return None
        t = torch.arange(L, device=device).float()
        dist = (L - 1 - t) / L                                      # 0 at now
        return torch.exp(-dist / self.phase_recency_tau).view(1, L)

    def _phase_profile(
        self, h: torch.Tensor, periods: torch.Tensor,
    ) -> torch.Tensor:
        """Period-fold the encoder output into per-phase profiles.

        Returns (B, K, n_bins, stat_mult·D). Each detected period p_k folds
        the sequence into n_bins phase bins; per bin we keep the (recency-
        weighted) mean [and variance]. Empty bins → global mean.
        """
        B, L, D = h.shape
        K, nb = self.K, self.phase_bins
        t = torch.arange(L, device=h.device).view(1, L).float()    # (1,L)
        p_safe = periods.clamp(min=1).float()                      # (B,K)
        weight = self._recency_weight(L, h.device)
        if weight is not None:
            weight = weight.expand(B, L)
        profs = []
        for k in range(K):
            frac = (t % p_safe[:, k:k + 1]) / p_safe[:, k:k + 1]   # (B,L)
            bins = torch.clamp((frac * nb).long(), max=nb - 1)     # (B,L)
            profs.append(self._scatter_profile(h, bins, nb, weight))
        return torch.stack(profs, dim=1)                           # (B,K,nb,S·D)

    def _gather_phase(
        self, prof: torch.Tensor, fut_pos: torch.Tensor,
        periods: torch.Tensor, weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gather each future query's matching phase bin per period, mix → D.

        prof: (B, K, n_bins, S·D); fut_pos: (B, H); periods: (B, K).
        weight: optional (B,K) per-period reliability weight (period_trust).
        Returns (B, H, D).
        """
        B, K, nb, SD = prof.shape
        H = fut_pos.shape[1]
        p_safe = periods.clamp(min=1).float().view(B, 1, K)        # (B,1,K)
        frac = (fut_pos.unsqueeze(-1).float() % p_safe) / p_safe   # (B,H,K)
        fbins = torch.clamp((frac * nb).long(), max=nb - 1)        # (B,H,K)
        idx = fbins.permute(0, 2, 1).unsqueeze(-1).expand(B, K, H, SD)
        gathered = torch.gather(prof, 2, idx)                      # (B,K,H,S·D)
        if weight is not None:
            gathered = gathered * weight.view(B, K, 1, 1).to(gathered.dtype)
        gathered = gathered.permute(0, 2, 1, 3).reshape(B, H, K * SD)
        return self.phase_mix(gathered)                            # (B,H,D)

    def _period_trust_weights(
        self, periods: torch.Tensor, scores: torch.Tensor,
    ) -> torch.Tensor:
        """Hyperparameter-free per-period reliability weight w_k∈[0,1] (B,K).

        ln-coverage = ln(L/period) (data/structure-determined); for 'full' also
        the significance margin ln(s_k/t_alpha) (s_k = the periodogram score,
        t_alpha = data-determined Bonferroni threshold). The sigmoid crossover
        is LEARNED (the linear layer's weights and bias), not a hand-set
        threshold. 0 on rejected slots.
        """
        valid = periods > 0
        logcov = torch.log(float(self.L) / periods.clamp(min=1).float())   # (B,K)
        if self.period_trust == "coverage":
            feat = logcov.unsqueeze(-1)                                     # (B,K,1)
        else:
            margin = torch.log(scores.clamp(min=1e-12) / self._t_alpha)    # >=0 for survivors
            feat = torch.stack([margin, logcov], dim=-1)                   # (B,K,2)
        w = torch.sigmoid(self.pt(feat.to(self.pt.weight.dtype))).squeeze(-1)
        return torch.where(valid, w, torch.zeros_like(w))

    def _recency_feat(self, h: torch.Tensor) -> torch.Tensor:
        """Recency-binned profile: bin context positions by
        log-distance-from-now and pool. Always valid (no period needed);
        the aperiodic content path. Flattened to a single (B, D) descriptor
        (broadcast to all horizons; the query's own PE carries how-far-ahead).
        """
        B, L, D = h.shape
        rb = self.recency_bins
        t = torch.arange(L, device=h.device).view(1, L).float().expand(B, L)
        dist = (L - 1 - t).clamp(min=0.0)                          # 0=now
        frac = torch.log1p(dist) / math.log1p(float(L - 1) + 1e-9)
        bins = torch.clamp((frac * rb).long(), max=rb - 1)         # (B,L)
        prof = self._scatter_profile(h, bins, rb, None)           # (B,rb,S·D)
        return self.recency_mix(prof.reshape(B, rb * self.stat_mult * D))

    def _cross_cycle_profile(
        self, h: torch.Tensor, periods: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-cycle conv, true ragged form.

        For the dominant period p0, fold the sequence into a
        [cycles-back-from-now × phase] grid (B, nc, nb, D) by scatter-mean,
        then convolve ACROSS the cycle axis at fixed phase (depthwise conv1d
        over nc), modelling how each phase evolves cycle-to-cycle ("every
        Monday 9am, trending up"). Read out the most-recent cycle (post-conv,
        so it has seen the trend). Returns a (B, nb, D) phase profile the
        decoder gathers by its own phase.

        Per-sample period handled like phase-binning: phase resampled to nb
        fixed bins; cycles-back clamped to nc (older cycles fold into the
        oldest slot). Attention-free, fixed-shape, batchable.
        """
        B, L, D = h.shape
        nb, nc = self.cc_bins, self.cc_cycles
        t = torch.arange(L, device=h.device).view(1, L).float()    # (1,L)
        p0 = periods[:, :1].clamp(min=1).float()                   # (B,1)
        pbin = torch.clamp(((t % p0) / p0 * nb).long(), max=nb - 1)  # (B,L)
        cyc = torch.clamp(((L - 1 - t) // p0).long(), max=nc - 1)   # (B,L) 0=now
        comb = (cyc * nb + pbin).clamp(min=0, max=nc * nb - 1)      # (B,L)
        oh = F.one_hot(comb, nc * nb).to(h.dtype)                  # (B,L,nc·nb)
        cnt = oh.sum(dim=1).unsqueeze(-1).clamp(min=1e-6)
        grid = (torch.bmm(oh.transpose(1, 2), h) / cnt).view(B, nc, nb, D)
        # conv across cycles (nc) at fixed phase, per channel.
        x = grid.permute(0, 2, 3, 1).reshape(B * nb, D, nc)        # (B·nb, D, nc)
        x = self.cc_conv(x).reshape(B, nb, D, nc)
        return x[..., 0]                                           # most-recent cycle (B,nb,D)

    @staticmethod
    def _prefix_integral(
        f: torch.Tensor, Csum: torch.Tensor, xpad: torch.Tensor, L: int,
    ) -> torch.Tensor:
        """Integral of piecewise-constant x from 0 to fractional position f.
        f: (B,M) in native units. Csum: (B,L+1) prefix sums; xpad: (B,L+1)."""
        fc = f.clamp(0.0, float(L))
        k = torch.floor(fc).long()
        rem = (fc - k.float()).to(Csum.dtype)
        return torch.gather(Csum, 1, k) + rem * torch.gather(xpad, 1, k)

    def _resolution_adapt(
        self, x: torch.Tensor, periods: torch.Tensor, H: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolution adaptation (the TinyCast premise; zero params).

        Resample the context onto a canonical-period grid so the fixed dilation
        schedule spans consistent CYCLE-fractions across all sampling rates: the
        dominant detected period is warped to ``res_period_target`` samples/cycle.
        All detected periods scale by the same ratio; the future native horizon
        is queried at its canonical-mapped position (decoder is position-
        parameterized, so outputs are native values and no output resampling
        is needed).

        Returns (x_canon (B,L), periods_canon (B,K), fut_pos_canon (B,H) float).
        Aperiodic series (dominant period 0) pass through unchanged (r=1).
        """
        B, L = x.shape
        pt = float(self.res_period_target)
        p0 = periods[:, 0].float()                                  # (B,) dominant
        r = torch.where(p0 > 0, pt / p0.clamp(min=1.0), torch.ones_like(p0))
        # res_r_max=1.0 → downsample-only (high-freq squeezed to canonical;
        # low-freq left at native, no history truncation / no Δ blow-up).
        r = r.clamp(0.05, self.res_r_max).view(B, 1)                # canonical per native
        j = torch.arange(L, device=x.device).view(1, L).float()     # canonical idx 0..L-1
        # native time (center) for canonical index j; now = most recent canonical.
        t = (L - 1) - ((L - 1) - j) / r                             # (B,L), <0 = pre-context
        # Upsample/identity (r>=1): linear interpolation (true pass-through at
        # r=1). Downsample (r<1): area-average over the native window w=1/r,
        # ANTI-ALIASED (averages the whole window, not 2 endpoints). Zero params.
        t0 = torch.floor(t)
        frac = (t - t0).to(x.dtype)
        x_lin = (torch.gather(x, 1, t0.clamp(0, L - 1).long()) * (1.0 - frac)
                 + torch.gather(x, 1, (t0 + 1).clamp(0, L - 1).long()) * frac)
        w = 1.0 / r                                                 # (B,1) native window width
        lo, hi = t - w / 2.0, t + w / 2.0
        Csum = F.pad(x.cumsum(dim=1), (1, 0))                       # (B,L+1): Csum[k]=Σ x[:k]
        xpad = F.pad(x, (0, 1))                                     # (B,L+1): x[L]=0
        denom = (hi.clamp(0.0, L) - lo.clamp(0.0, L)).clamp(min=1e-6)
        x_area = (self._prefix_integral(hi, Csum, xpad, L)
                  - self._prefix_integral(lo, Csum, xpad, L)) / denom
        x_canon = torch.where(r < 1.0, x_area, x_lin)              # anti-alias only on downsample
        x_canon = x_canon * (t >= 0).to(x.dtype)                   # mask pre-context → 0
        periods_canon = torch.round(periods.float() * r).long()
        periods_canon = torch.where(
            periods > 0, periods_canon.clamp(min=2), torch.zeros_like(periods),
        )
        h_steps = torch.arange(1, H + 1, device=x.device).view(1, H).float()
        fut_pos_canon = (L - 1) + h_steps * r                       # (B,H) canonical
        return x_canon, periods_canon, fut_pos_canon

    def _seasonal_naive(
        self, x: torch.Tensor, fut_pos: torch.Tensor, periods: torch.Tensor,
    ) -> torch.Tensor:
        """Value-space seasonal-naive baseline (zero params).

        Fold the (normalized) input x by the dominant period into phase bins,
        take the per-phase mean VALUE, and gather the bin matching each future
        query's phase. The network then learns only the residual on top of this
        baseline: a target reframe, not added model complexity. Aperiodic /
        empty-bin → fall back to the context mean (persistence-of-level).

        x: (B, L)  fut_pos: (B, H)  periods: (B, K)  →  (B, H)
        """
        B, L = x.shape
        nb = self.phase_bins if self.phase_bins > 0 else 16
        t = torch.arange(L, device=x.device).view(1, L).float()
        p0 = periods[:, :1].clamp(min=1).float()                   # (B,1) dominant
        pbin = torch.clamp(((t % p0) / p0 * nb).long(), max=nb - 1)  # (B,L)
        oh = F.one_hot(pbin, nb).to(x.dtype)                       # (B,L,nb)
        cnt = oh.sum(dim=1)                                        # (B,nb)
        base = torch.bmm(oh.transpose(1, 2), x.unsqueeze(-1)).squeeze(-1)  # (B,nb)
        gmean = x.mean(dim=1, keepdim=True)                       # (B,1)
        base = torch.where(cnt > 0, base / cnt.clamp(min=1.0), gmean.expand(B, nb))
        fb = torch.clamp((fut_pos.float() % p0) / p0 * nb, max=nb - 1).long()  # (B,H)
        return torch.gather(base, 1, fb)                          # (B,H)

    def _super_naive(
        self, x: torch.Tensor, fut_pos: torch.Tensor, periods: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-seasonal "super-naive" baseline (zero params).

        Greedy additive decomposition over ALL significant periods: start from
        the context mean, then for each significant period (strongest first)
        fold the running residual into per-phase means, subtract it (deflate),
        and accumulate that component's value at the future phase. Result:
        baseline(L+h) = mean + Σ_k s_k[phase_k(L+h)], the genuine multi-period
        seasonal-naive forecast. Non-significant periods (p=0) contribute zero.

        x: (B, L)  fut_pos: (B, H)  periods: (B, K)  →  (B, H)
        """
        B, L = x.shape
        H = fut_pos.shape[1]
        nb = self.phase_bins if self.phase_bins > 0 else 16
        t = torch.arange(L, device=x.device).view(1, L).float()
        if self.residual_trend:
            # Level term = linear trend (closed-form LS), extrapolated forward.
            # baseline = trend + seasonal, the classical decomposition.
            tc = t - t.mean()                                     # centered (1,L)
            xc = x - x.mean(dim=1, keepdim=True)                  # (B,L)
            slope = (tc * xc).sum(1, keepdim=True) / (tc * tc).sum().clamp(min=1.0)
            intercept = x.mean(dim=1, keepdim=True)               # value at centered t=0
            tmean = t.mean()
            trend_ctx = intercept + slope * (t - tmean)           # (B,L)
            r = x - trend_ctx                                     # de-trended residual
            baseline = intercept + slope * (fut_pos.float() - tmean)  # (B,H) trend extrap
        else:
            mean = x.mean(dim=1, keepdim=True)                    # (B,1)
            r = x - mean                                          # residual
            baseline = mean.expand(B, H).clone()                  # (B,H)
        for k in range(self.K):
            pk = periods[:, k:k + 1].float()                      # (B,1), 0 if not sig
            sig = (pk > 0).to(x.dtype)                            # (B,1)
            pks = pk.clamp(min=1.0)
            pbin = torch.clamp((t % pks) / pks * nb, max=nb - 1).long()   # (B,L)
            oh = F.one_hot(pbin, nb).to(x.dtype)                 # (B,L,nb)
            cnt = oh.sum(dim=1).clamp(min=1.0)                   # (B,nb)
            s_k = torch.bmm(oh.transpose(1, 2), r.unsqueeze(-1)).squeeze(-1) / cnt
            s_k = s_k * sig                                      # (B,nb), zero if not sig
            r = r - torch.gather(s_k, 1, pbin)                   # deflate
            fb = torch.clamp((fut_pos.float() % pks) / pks * nb, max=nb - 1).long()
            baseline = baseline + torch.gather(s_k, 1, fb)       # (B,H)
        return baseline

    def _gather_cc(
        self, cc_prof: torch.Tensor, fut_pos: torch.Tensor,
        periods: torch.Tensor,
    ) -> torch.Tensor:
        """Gather each future query's matching phase bin from the cross-cycle
        profile (dominant period), mix → D. cc_prof: (B,nb,D)."""
        B, nb, D = cc_prof.shape
        H = fut_pos.shape[1]
        p0 = periods[:, :1].clamp(min=1).float()                   # (B,1)
        fb = torch.clamp((fut_pos.float() % p0) / p0 * nb, max=nb - 1).long()
        gathered = torch.gather(cc_prof, 1, fb.unsqueeze(-1).expand(B, H, D))
        return self.cc_mix(gathered)                               # (B,H,D)

    # ---- forward ----------------------------------------------------------

    def _local_anchor_channels(
        self, x: torch.Tensor, scale_factor: torch.Tensor | float | None,
    ) -> torch.Tensor:
        """Two causal local-statistics channels exposing the recent level/scale to
        the encoder (the gradient-connected re-anchoring signal WindowMinMax lacks):
          ch1 = (x_t - m_t) / (s_t + eps)   local-scale residual (a causal z-score)
          ch2 = log(s_t + eps)              log local scale (global normed range ~= 1)
        m_t, s_t = causal boxcar mean / std over a trailing window w ~ one canonical
        period round(base_seasonality / scale_factor), clamped [8, L//4], fallback 64.
        Vectorized via cumsum + per-sample-window gather (O(L), no python loop).
        """
        B, L = x.shape
        device = x.device
        if scale_factor is not None:
            sf = (scale_factor if torch.is_tensor(scale_factor)
                  else x.new_tensor(scale_factor)).reshape(-1).float()
            if sf.numel() == 1:
                sf = sf.expand(B)
            w = (self.base_seasonality / sf.clamp(min=1e-3)).round().long()
            w = w.clamp(min=8, max=max(8, L // 4))
        else:
            w = torch.full((B,), 64, device=device, dtype=torch.long)
        # Center by the per-series mean before a FP32 cumsum. The two-pass variance
        # (E[x^2]-E[x]^2) over a length-L cumsum otherwise suffers catastrophic
        # cancellation on long flat/sparse regions; variance is shift-invariant, so
        # centering changes nothing but keeps the cumsum magnitudes small enough that
        # fp32 stays accurate there, at no extra memory.
        xf = x.float()
        xc = xf - xf.mean(dim=1, keepdim=True)
        cs = F.pad(torch.cumsum(xc, dim=1), (1, 0))           # (B, L+1), cs[:,0]=0
        cs2 = F.pad(torch.cumsum(xc * xc, dim=1), (1, 0))
        t = torch.arange(L, device=device).view(1, L).expand(B, L)
        lo = (t - w.view(B, 1) + 1).clamp(min=0)              # trailing-window start
        cnt = (t - lo + 1).float()                            # window length (>= 1)
        sum_x = cs.gather(1, t + 1) - cs.gather(1, lo)
        sum_x2 = cs2.gather(1, t + 1) - cs2.gather(1, lo)
        m = sum_x / cnt                                       # centered local mean
        s = (sum_x2 / cnt - m * m).clamp(min=0.0).sqrt()      # local std (shift-invariant)
        eps = 1e-4   # floor vs the unit normed range -> flat regions give ch1 ~ 0, no blowup
        ch1 = (xc - m) / (s + eps)                            # = (x - local mean)/(s+eps)
        ch2 = torch.log(s + eps)
        return torch.stack([ch1, ch2], dim=-1).to(x.dtype)    # (B, L, 2)

    def forward(
        self,
        x_normed: torch.Tensor,
        nan_mask: torch.Tensor | None = None,
        scale_factor: torch.Tensor | float | None = None,
        horizon: int | None = None,
    ) -> torch.Tensor:
        # observed-mask (1=observed, 0=missing) for the missing-value channel.
        # res_adaptive is rejected with missing_channel (see __init__), so this
        # mask stays aligned with x throughout.
        obs_mask = None
        if self.missing_channel and nan_mask is not None:
            obs_mask = nan_mask[..., 0] if nan_mask.dim() == 3 else nan_mask  # (B,L)

        if x_normed.dim() == 3 and x_normed.shape[-1] > 1:
            x = x_normed[..., 0]
        elif x_normed.dim() == 3:
            x = x_normed.squeeze(-1)
        else:
            x = x_normed                                            # (B, L)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B, L = x.shape
        assert L == self.L, f"context length mismatch: got {L}, expected {self.L}"
        H = self.p_out if horizon is None else int(horizon)
        device = x.device

        # Period detection in fp32 (no grad).
        periods, n_valid, scores = self._detect_periods(x.float())  # (B,K),(B,),(B,K)

        # Resolution adaptation: warp context to a canonical cycle-resolution so
        # the fixed dilations span consistent cycle-fractions across rates.
        if self.res_adaptive:
            x, periods, fut_pos = self._resolution_adapt(x, periods, H)
            ctx_pos = torch.arange(L, device=device).view(1, L).expand(B, L)
        else:
            ctx_pos = torch.arange(L, device=device).view(1, L).expand(B, L)
            fut_pos = torch.arange(L, L + H, device=device).view(1, H).expand(B, H)
        with torch.amp.autocast(
            device_type=device.type if x.is_cuda else "cpu", enabled=False,
        ):
            ctx_pe = _positional_encoding(
                ctx_pos, periods, L, n_harmonics=self.n_harmonics,
            )
            fut_pe = _positional_encoding(
                fut_pos, periods, L, n_harmonics=self.n_harmonics,
            )
        ctx_pe = ctx_pe.to(x.dtype)
        fut_pe = fut_pe.to(x.dtype)

        # Per-period reliability weight (period_trust; off in the released config).
        # Down-weight unreliable/spurious periods continuously. Applied to the
        # phase-encoding channels here, and to the phase-binning gather + gate below.
        ptw = None
        if self.pt is not None:
            ptw = self._period_trust_weights(periods, scores).to(x.dtype)  # (B,K)
            # phase channels are the first n_phase cols, laid out per period as
            # n_harmonics*2 consecutive channels → repeat each w_k that many times.
            rep = self.n_harmonics * 2
            chan_w = ptw.repeat_interleave(rep, dim=1).view(B, 1, -1)        # (B,1,n_phase)
            np_ = chan_w.shape[-1]
            ctx_pe = torch.cat([ctx_pe[..., :np_] * chan_w, ctx_pe[..., np_:]], dim=-1)
            fut_pe = torch.cat([fut_pe[..., :np_] * chan_w, fut_pe[..., np_:]], dim=-1)

        # Embed context.
        if self.decompose_kernel > 0:
            # moving-average series decomposition: moving-average trend +
            # seasonal residual, fed as two channels.
            k = self.decompose_kernel
            pad = k // 2
            xp = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate")  # (B,1,L+2pad)
            trend = F.avg_pool1d(xp, kernel_size=k, stride=1)[..., :L].squeeze(1)
            seasonal = x - trend
            value_ch = torch.stack([trend, seasonal], dim=-1)       # (B,L,2)
        else:
            value_ch = x.unsqueeze(-1)                              # (B,L,1)
        if self.missing_channel and obs_mask is not None:
            value_ch = torch.cat(
                [value_ch, obs_mask.unsqueeze(-1).to(value_ch.dtype)], dim=-1
            )                                                       # (B,L,nv+1)
        if self.local_anchor:
            value_ch = torch.cat(
                [value_ch, self._local_anchor_channels(x, scale_factor)], dim=-1
            )                                                       # (B,L,nv+2)
        ctx_in = torch.cat([value_ch, ctx_pe], dim=-1)             # (B,L,nv+n_pe)
        h = self.in_proj(ctx_in)                                    # (B, L, D)

        # Dilated-conv encoder.
        for block in self.encoder:
            h = block(h)

        # Per-horizon context feature: static pooled summary, broadcast to all H.
        summary = self._pool(h)                                     # (B, pool_dim)
        ctx_feat = summary.unsqueeze(1).expand(B, H, -1)            # (B, H, pool_dim)

        # Decoder: per-horizon query from PE(L+h) + context feature
        # [+ phase profile] [+ recency profile] [+ cross-cycle feat].
        q_parts = [fut_pe, ctx_feat]

        phase_feat = None
        if self.phase_mix is not None:
            prof = self._phase_profile(h, periods)                  # (B,K,nb,S·D)
            phase_feat = self._gather_phase(prof, fut_pos, periods, weight=ptw)  # (B,H,D)

        rec_feat = None
        if self.recency_mix is not None:
            rec_feat = self._recency_feat(h).unsqueeze(1).expand(B, H, self.D)

        if self.sig_gate and phase_feat is not None and rec_feat is not None:
            # Blend by periodicity strength: many significant periods → trust
            # the phase profile; few/none → lean on the recency profile. With
            # period_trust, use the soft Σw instead of the integer n_valid.
            strength = ptw.sum(dim=1) if ptw is not None else n_valid.float()
            g = (strength / float(self.K)).clamp(0.0, 1.0).view(B, 1, 1)
            q_parts.append(g * phase_feat + (1.0 - g) * rec_feat)
        else:
            if phase_feat is not None:
                q_parts.append(phase_feat)
            if rec_feat is not None:
                q_parts.append(rec_feat)

        if self.cross_cycle:
            cc_prof = self._cross_cycle_profile(h, periods)         # (B,nb,D)
            q_parts.append(self._gather_cc(cc_prof, fut_pos, periods))  # (B,H,D)

        q_in = torch.cat(q_parts, dim=-1)
        q = self.query_proj(q_in)                                   # (B, H, D)

        if self.future_conv:
            # Horizon-axis evolving states from a causal conv over context-tail
            # + seasonal-naive fill, injected additively (fc_out zero-init =>
            # exact baseline at init). Gives the static-summary decoder the
            # per-horizon dynamics it lacks.
            fill = self._seasonal_naive(x, fut_pos, periods)        # (B, H)
            fut_states = self._future_conv_states(h, fut_pe, fill)  # (B, H, D)
            q = q + self.fc_out(fut_states)

        for ffn, norm in zip(self.decoder_ffns, self.decoder_norms):
            q = _norm_fp32(norm, q + ffn(q))

        if self.horizon_conv is not None:
            # Causal conv over the horizon axis: pad (k-1) on the left so step
            # h sees only h, h-1, …, h-(k-1): no future leakage, any H.
            qt = F.pad(q.transpose(1, 2), (self.horizon_kernel - 1, 0))
            hc = self.horizon_conv(qt).transpose(1, 2)             # (B,H,D)
            q = _norm_fp32(self.horizon_norm, q + hc)

        if self.hr_z is not None:
            # gated-recurrence decode-state: gated diagonal recurrence over the H axis.
            # Scans the query FEATURES only (no value feedback). Sequential
            # scan: the horizon is short, so this is cheap and stable.
            z = torch.sigmoid(self.hr_z(q))                        # (B,H,D) update gate
            c = torch.tanh(self.hr_c(q))                           # (B,H,D) candidate
            s = torch.zeros(B, self.D, dtype=q.dtype, device=q.device)
            states = []
            for t in range(q.shape[1]):
                s = (1.0 - z[:, t]) * s + z[:, t] * c[:, t]
                states.append(s)
            hstate = torch.stack(states, dim=1)                    # (B,H,D)
            q = _norm_fp32(self.hr_norm, q + self.hr_o(hstate))    # hr_o zero-init

        y = self.out_proj(q)                                        # (B, H, Q)


        if self.residual_naive:
            # Learn the residual over a (multi-)seasonal-naive baseline.
            if self.residual_multi:
                baseline = self._super_naive(x, fut_pos, periods)   # (B, H)
            else:
                baseline = self._seasonal_naive(x, fut_pos, periods)  # (B, H)
            y = y + baseline.unsqueeze(-1)

        return y


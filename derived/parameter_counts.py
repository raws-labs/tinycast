#!/usr/bin/env python3
"""Parameter counts for the deployed model and for every ablation arm.

Counts are taken by instantiating each configuration from the released
implementation and summing trainable scalars.  They are never summed from a
checkpoint: both SwiGLU stacks are weight-tied, so a loaded checkpoint
materializes the tied weights once per block and over-counts the encoder by
nine copies and the future-conv stack by five.  ``nn.Module.parameters()``
yields each shared tensor once, which is what makes instantiation the counting
method here.

The per-module rows reproduce the paper's parameter-budget table, and the
INT8 split reproduces the firmware's weight pack: every ``Linear`` and
``Conv1d`` weight is quantized per output channel, while biases and RMSNorm
scalars stay at higher precision and are packed separately.

Usage:
    python3 parameter_counts.py                    # print the table
    python3 parameter_counts.py --write FILE.json  # regenerate the record
    python3 parameter_counts.py --check FILE.json  # verify the record

The implementation is found via ``import tinycast``, the ``TINYCAST_ROOT``
environment variable, or ``--package-root``.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys

# Every arm the paper reports a parameter count for, as constructor keywords.
COMMON = dict(seq_len=2048, p_out=48, d=64, n_layers=10, kernel=3,
              pool_kind="mean_last", top_k_periods=4)

ARMS = [
    ("architecture family control",
     dict(COMMON, ffn_mult=1.5, n_quantiles=1)),
    ("architecture family, phase binning",
     dict(COMMON, ffn_mult=1.5, n_quantiles=1, phase_bins=16)),
    ("architecture family, phase binning and recency gate",
     dict(COMMON, ffn_mult=1.5, n_quantiles=1, phase_bins=16, recency_bins=8,
          sig_gate=True)),
    ("architecture family, substituted-recency arm",
     dict(COMMON, ffn_mult=1.5, n_quantiles=1, phase_bins=0, recency_bins=16,
          periodogram_off=True)),
    ("component family control",
     dict(COMMON, ffn_mult=1.5, n_quantiles=9, phase_bins=16, recency_bins=8,
          sig_gate=True)),
    ("component family, future-conv arm",
     dict(COMMON, ffn_mult=1.5, n_quantiles=9, phase_bins=16, recency_bins=8,
          sig_gate=True, future_conv=True)),
    ("deployed model",
     dict(COMMON, ffn_mult=1.0, n_quantiles=9, phase_bins=16, causal=True,
          separable_conv=True, share_ffn=True, future_conv=True)),
    ("deployed architecture, feed-forward untied and convolutions not separable",
     dict(COMMON, ffn_mult=1.0, n_quantiles=9, phase_bins=16, causal=True,
          separable_conv=False, share_ffn=False, future_conv=True)),
]

DEPLOYED = "deployed model"
UNTIED = "deployed architecture, feed-forward untied and convolutions not separable"

# The paper's parameter-budget table, row by row, as (stage, module, pattern).
# The pattern matches parameter names of the deployed model; the counts are the
# sum of the matched tensors, so the table is derived rather than transcribed.
BUDGET_ROWS = [
    ("input", "Linear_in (14->64)", r"^in_proj\."),
    ("encoder", "depthwise conv (10x(64,1,3)+bias)", r"^encoder\.\d+\.conv\.0\."),
    ("encoder", "pointwise 1x1 (10x(64,64,1)+bias)", r"^encoder\.\d+\.conv\.1\."),
    ("encoder", "shared SwiGLU (up 64->128, down 64->64)", r"^encoder\.\d+\.ffn\."),
    ("encoder", "RMSNorms (20x(64))", r"^encoder\.\d+\.norm[12]\."),
    ("phase", "W_phase (256->64)", r"^phase_mix\."),
    ("query", "W_q (205->64)", r"^query_proj\."),
    ("decoder", "SwiGLU + RMSNorm", r"^decoder_(ffns|norms)\."),
    ("future-conv", "W_fc-in (14->64)", r"^fc_in_proj\."),
    ("future-conv", "depthwise conv (6x(64,1,3)+bias)", r"^fc_blocks\.\d+\.conv\.0\."),
    ("future-conv", "pointwise 1x1 (6x(64,64,1)+bias)", r"^fc_blocks\.\d+\.conv\.1\."),
    ("future-conv", "shared SwiGLU", r"^fc_blocks\.\d+\.ffn\."),
    ("future-conv", "RMSNorms (12x(64))", r"^fc_blocks\.\d+\.norm[12]\."),
    ("future-conv", "W_fc-out (64->64)", r"^fc_out\."),
    ("output", "W_out (64->9)", r"^out_proj\."),
]

SUBTOTALS = [("encoder subtotal", "encoder"), ("future-conv subtotal", "future-conv")]


def load_backbone(package_root=None):
    """Import the released implementation, from the given root if one is named."""
    roots = [package_root, os.environ.get("TINYCAST_ROOT")]
    here = pathlib.Path(__file__).resolve().parent
    roots += [here, here.parent, here / "code", here.parent / "code"]
    for root in roots:
        if root and (pathlib.Path(root) / "tinycast" / "backbone.py").is_file():
            sys.path.insert(0, str(pathlib.Path(root).resolve()))
            break
    from tinycast.backbone import DilatedConvBackbone
    return DilatedConvBackbone


def build(backbone, kwargs):
    return backbone(**kwargs)


def total_params(model):
    """Trainable scalars, tied tensors counted once."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def int8_split(model):
    """(quantized weight scalars, everything else) under the deployed scheme."""
    import torch.nn as nn
    seen, quantized = set(), 0
    for _, mod in model.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d)) and id(mod.weight) not in seen:
            seen.add(id(mod.weight))
            quantized += mod.weight.numel()
    return quantized, total_params(model) - quantized


def budget_table(model):
    named = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    rows, claimed = [], set()
    for stage, module, pattern in BUDGET_ROWS:
        hit = [(n, k) for n, k in named if re.match(pattern, n)]
        if not hit:
            raise SystemExit(f"budget row matched nothing: {stage} / {module}")
        claimed.update(n for n, _ in hit)
        rows.append({"stage": stage, "module": module, "params": sum(k for _, k in hit),
                     "tensors": len(hit)})
    missed = [n for n, _ in named if n not in claimed]
    if missed:
        raise SystemExit(f"parameters outside the budget table: {missed}")
    return rows


def compute(backbone):
    models = {name: build(backbone, kw) for name, kw in ARMS}
    arms = [{"arm": name,
             "params": total_params(models[name]),
             "config": {k: v for k, v in kw.items()}}
            for name, kw in ARMS]

    deployed = models[DEPLOYED]
    quantized, carried = int8_split(deployed)
    rows = budget_table(deployed)
    subtotals = {label: sum(r["params"] for r in rows if r["stage"] == stage)
                 for label, stage in SUBTOTALS}
    tied_saving = total_params(models[UNTIED]) - total_params(deployed)

    return {
        "_note": (
            "Parameter counts by instantiation from the released implementation. "
            "Tied tensors are counted once, which a checkpoint sum does not do."
        ),
        "method": "sum(p.numel() for p in model.parameters() if p.requires_grad)",
        "deployed": {
            "total_params": total_params(deployed),
            "int8_quantized_weight_scalars": quantized,
            "bias_and_norm_scalars_at_higher_precision": carried,
            "budget_rows": rows,
            "subtotals": subtotals,
        },
        "feed_forward_tying_and_separable_convolutions": {
            "params_without_both": total_params(models[UNTIED]),
            "params_with_both": total_params(deployed),
            "params_removed": tied_saving,
        },
        "arms": arms,
    }


def main():
    ap = argparse.ArgumentParser(description="TinyCast parameter counts by instantiation")
    ap.add_argument("--write", metavar="PATH", help="write the record as JSON")
    ap.add_argument("--check", metavar="PATH", help="verify an existing record")
    ap.add_argument("--package-root", metavar="DIR",
                    help="directory containing the tinycast package")
    args = ap.parse_args()

    record = compute(load_backbone(args.package_root))

    if args.write:
        with open(args.write, "w") as fh:
            json.dump(record, fh, indent=1)
            fh.write("\n")
        print(f"wrote {args.write}")
        return

    if args.check:
        with open(args.check) as fh:
            frozen = json.load(fh)
        bad = []
        for key in ("deployed", "feed_forward_tying_and_separable_convolutions", "arms"):
            if frozen.get(key) != record[key]:
                bad.append(key)
        if bad:
            print("MISMATCH in: " + ", ".join(bad))
            raise SystemExit(1)
        print(f"{args.check}: every count reproduces")
        return

    dep = record["deployed"]
    for row in dep["budget_rows"]:
        print(f"{row['stage']:12s}{row['module']:42s}{row['params']:>8,d}")
    for label, value in dep["subtotals"].items():
        print(f"{'':12s}{label:42s}{value:>8,d}")
    print(f"{'total':12s}{'':42s}{dep['total_params']:>8,d}")
    print(f"{'':12s}{'INT8 weight scalars':42s}"
          f"{dep['int8_quantized_weight_scalars']:>8,d}")
    print(f"{'':12s}{'biases and norm scalars':42s}"
          f"{dep['bias_and_norm_scalars_at_higher_precision']:>8,d}")
    print()
    for arm in record["arms"]:
        print(f"{arm['params']:>8,d}  {arm['arm']}")


if __name__ == "__main__":
    main()

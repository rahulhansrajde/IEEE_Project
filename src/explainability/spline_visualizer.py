"""Extraction and plotting of learned univariate KAN edge functions phi_{j,i}(x)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.models.kan_layers import KANLinear

from .plots import IEEE_DOUBLE_COL, NEUTRAL, PALETTE, save_figure, set_ieee_style

__all__ = ["find_kan_linear_layers", "top_edges", "extract_edge_curves", "plot_learned_splines", "plot_edge_responses"]


def find_kan_linear_layers(model: nn.Module) -> List[Tuple[str, KANLinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, KANLinear)]


def top_edges(layer: KANLinear, k: int = 12, out_index: Optional[int] = None) -> List[Tuple[int, int]]:
    """``k`` edges ``(out, in)`` with the largest spline importance ``mean |w_s c|``."""
    with torch.no_grad():
        importance = layer.edge_l1().detach().cpu()
    if out_index is not None:
        order = torch.argsort(importance[out_index], descending=True)[:k]
        return [(out_index, int(i)) for i in order]
    flat = torch.argsort(importance.flatten(), descending=True)[:k]
    return [(int(i) // layer.in_features, int(i) % layer.in_features) for i in flat]


@torch.no_grad()
def extract_edge_curves(
    layer: KANLinear, edges: Sequence[Tuple[int, int]], n_points: int = 400
) -> List[Dict[str, np.ndarray]]:
    """Evaluate ``phi``, its SiLU base part and its B-spline part on each edge's grid domain."""
    curves = []
    k = layer.spline_order
    for out_i, in_i in edges:
        knots = layer.grid[in_i].detach().cpu()
        lo, hi = float(knots[k]), float(knots[-k - 1])
        span = hi - lo
        x = torch.linspace(lo - 0.1 * span, hi + 0.1 * span, n_points)
        phi, base, spline = layer.edge_function(x, out_i, in_i)
        curves.append(
            {
                "edge": (out_i, in_i),
                "x": x.numpy(),
                "phi": phi.cpu().numpy(),
                "base": base.cpu().numpy(),
                "spline": spline.cpu().numpy(),
                "knots": knots[k : len(knots) - k].numpy(),
                "importance": float(layer.edge_l1()[out_i, in_i]),
            }
        )
    return curves


def plot_learned_splines(
    layer: KANLinear,
    out_dir: str | Path,
    stem: str = "fig2_learned_kan_splines",
    grid_shape: Tuple[int, int] = (3, 4),
    layer_name: str = "KANLinear",
    out_names: Optional[Sequence[str]] = None,
    inputs: Optional[torch.Tensor] = None,
) -> List[Path]:
    """3x4 grid of the most important learned edge functions of ``layer``.

    Each panel shows the total activation ``phi`` (solid), its SiLU residual
    (dashed) and B-spline component (dotted), the interior knots (ticks) and,
    if ``inputs`` (``(N, in)``, already in the spline domain) are given, the
    empirical input density as a shaded histogram.
    """
    set_ieee_style()
    rows, cols = grid_shape
    edges = top_edges(layer, rows * cols)
    curves = extract_edge_curves(layer, edges)
    fig, axes = plt.subplots(rows, cols, figsize=(IEEE_DOUBLE_COL, 1.55 * rows + 0.3), squeeze=False)
    for ax, c in zip(axes.flat, curves):
        out_i, in_i = c["edge"]
        if inputs is not None:
            vals = inputs[:, in_i].detach().cpu().numpy()
            ax2 = ax.twinx()
            ax2.hist(vals, bins=30, color="#d9d9d6", alpha=0.6, density=True)
            ax2.set_yticks([])
            ax2.grid(False)
            ax2.spines["right"].set_visible(False)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
        ax.plot(c["x"], c["phi"], color=PALETTE[0], lw=1.6, label=r"$\phi(x)$")
        ax.plot(c["x"], c["base"], color=NEUTRAL, lw=0.8, ls="--", label=r"$w_b\,\mathrm{SiLU}(x)$")
        ax.plot(c["x"], c["spline"], color=PALETTE[1], lw=0.9, ls=":", label=r"$w_s\sum_m c_m B_m(x)$")
        ymin, ymax = ax.get_ylim()
        ax.vlines(c["knots"], ymin, ymin + 0.06 * (ymax - ymin), color="#111111", lw=0.6)
        ax.set_ylim(ymin, ymax)
        target = out_names[out_i] if out_names is not None and out_i < len(out_names) else str(out_i)
        ax.set_title(rf"$\phi_{{\mathrm{{{target}}},\,{in_i}}}$  ($\overline{{|c|}}$={c['importance']:.2e})", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(curves):]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Learned B-spline edge activations of {layer_name} (top-{len(curves)} edges by spline importance)",
                 fontsize=8.5, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return save_figure(fig, out_dir, stem)


@torch.no_grad()
def plot_edge_responses(
    layer: KANLinear,
    x_input: torch.Tensor,
    out_index: int,
    out_dir: str | Path,
    stem: str,
    k: int = 8,
    out_name: Optional[str] = None,
) -> List[Path]:
    """Per-image explanation: the ``k`` edges feeding ``out_index`` with largest
    contribution ``|phi(x_i)|`` for this image, marking the actual input value."""
    set_ieee_style()
    x_input = x_input.reshape(-1).to(layer.grid.device)
    contrib = []
    for i in range(layer.in_features):
        phi, _, _ = layer.edge_function(x_input[i : i + 1], out_index, i)
        contrib.append(float(phi.abs()))
    order = np.argsort(contrib)[::-1][:k]
    curves = extract_edge_curves(layer, [(out_index, int(i)) for i in order])
    cols = 4
    rows = int(np.ceil(len(curves) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(IEEE_DOUBLE_COL, 1.6 * rows + 0.3), squeeze=False)
    for ax, c, i in zip(axes.flat, curves, order):
        ax.plot(c["x"], c["phi"], color=PALETTE[0], lw=1.4)
        xv = float(x_input[i])
        yv, _, _ = layer.edge_function(torch.tensor([xv]), out_index, int(i))
        ax.scatter([xv], [float(yv)], color=PALETTE[1], s=22, zorder=4, edgecolors="white", linewidths=1.0)
        ax.set_title(rf"input {int(i)}: $\phi$={float(yv):+.3f}", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.flat[len(curves):]:
        ax.axis("off")
    fig.suptitle(f"Top-{len(curves)} spline contributions to '{out_name or out_index}'", fontsize=8.5)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)

"""IEEE-formatted publication figures (300 DPI PNG + vector PDF).

Styling: serif fonts, light grid lines, and a colour-vision-deficiency-validated
categorical palette with redundant line-style/marker encoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

__all__ = [
    "PALETTE",
    "LINESTYLES",
    "MARKERS",
    "set_ieee_style",
    "save_figure",
    "plot_roc_pr_curves",
    "plot_confusion_matrices",
    "plot_tsne",
    "plot_pareto_efficiency",
    "plot_training_dynamics",
]

# Categorical palette validated for CVD separation on a white surface.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
NEUTRAL = "#52514e"
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">"]

IEEE_SINGLE_COL = 3.5  # inches
IEEE_DOUBLE_COL = 7.16


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#3b3b3b",
            "axes.grid": True,
            "grid.color": "#d9d9d6",
            "grid.linewidth": 0.4,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.2,
            "legend.frameon": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: str | Path, stem: str) -> List[Path]:
    """Save ``fig`` as ``<stem>.png`` (300 DPI) and ``<stem>.pdf``; close it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    return paths


def plot_roc_pr_curves(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    out_dir: str | Path,
    stem: str = "fig3_roc_and_pr_curves",
    title_prefix: str = "",
) -> List[Path]:
    """Per-class + macro-average ROC and precision-recall curves."""
    set_ieee_style()
    n_cls = len(class_names)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 3.0))
    grid = np.linspace(0, 1, 501)
    tprs, precs = [], []
    for c in range(n_cls):
        y = (labels == c).astype(int)
        if y.min() == y.max():
            continue
        color, ls = PALETTE[c % len(PALETTE)], LINESTYLES[c % len(LINESTYLES)]
        fpr, tpr, _ = roc_curve(y, probs[:, c])
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, ls=ls, lw=1.0, label=f"{class_names[c]} ({roc_auc:.3f})")
        tprs.append(np.interp(grid, fpr, tpr))
        prec, rec, _ = precision_recall_curve(y, probs[:, c])
        ap = average_precision_score(y, probs[:, c])
        ax_pr.plot(rec, prec, color=color, ls=ls, lw=1.0, label=f"{class_names[c]} ({ap:.3f})")
        precs.append(np.interp(grid, rec[::-1], prec[::-1]))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    ax_roc.plot(grid, mean_tpr, color="#111111", lw=2.0, label=f"Macro avg. ({auc(grid, mean_tpr):.3f})")
    ax_roc.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.6, ls=":")
    mean_prec = np.mean(precs, axis=0)
    ax_pr.plot(grid, mean_prec, color="#111111", lw=2.0, label=f"Macro avg. ({auc(grid, mean_prec):.3f})")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax_pr.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01))
    ax_roc.set_title(f"{title_prefix}(a) ROC curves (AUC)")
    ax_pr.set_title(f"{title_prefix}(b) Precision-recall curves (AP)")
    ax_roc.legend(loc="lower right", handlelength=2.2)
    ax_pr.legend(loc="lower left", handlelength=2.2)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)


def plot_confusion_matrices(
    matrices: Dict[str, np.ndarray],
    class_names: Sequence[str],
    out_dir: str | Path,
    stem: str = "fig4_confusion_matrix",
) -> List[Path]:
    """Row-normalised confusion matrices side by side (one panel per model)."""
    set_ieee_style()
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(min(IEEE_DOUBLE_COL, 3.5 * n), 3.1), squeeze=False)
    im = None
    for ax, (name, cm) in zip(axes[0], matrices.items()):
        cm = np.asarray(cm, dtype=np.float64)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=5.8,
                        color="white" if norm[i, j] > 0.55 else "#111111")
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(name)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.02, label="Recall-normalised rate")
    return save_figure(fig, out_dir, stem)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    out_dir: str | Path,
    stem: str = "fig7_tsne_embeddings",
    seed: int = 0,
    max_points: int = 2000,
    title: str = "t-SNE of penultimate embeddings",
) -> List[Path]:
    from sklearn.manifold import TSNE

    set_ieee_style()
    rng = np.random.default_rng(seed)
    if len(labels) > max_points:
        idx = rng.choice(len(labels), max_points, replace=False)
        embeddings, labels = embeddings[idx], labels[idx]
    perplexity = float(min(30, max(5, (len(labels) - 1) // 3)))
    z = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, 3.0))
    for c, name in enumerate(class_names):
        m = labels == c
        if m.any():
            ax.scatter(z[m, 0], z[m, 1], s=8, color=PALETTE[c % len(PALETTE)], marker=MARKERS[c % len(MARKERS)],
                       edgecolors="white", linewidths=0.3, label=f"{name} (n={int(m.sum())})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=1.4)
    return save_figure(fig, out_dir, stem)


def plot_pareto_efficiency(
    rows: List[Dict[str, float]],
    out_dir: str | Path,
    stem: str = "fig5_pareto_efficiency",
    highlight: Optional[str] = None,
) -> List[Path]:
    """Macro-F1 vs. parameters (a) and vs. checkpoint size (b); bubble area ~ GFLOPs.

    Each row needs ``display_name``, ``macro_f1``, ``params_m``, ``model_size_mb``
    and ``gflops``. Points on the (min-size, max-F1) Pareto frontier are connected.
    """
    set_ieee_style()
    fig, axes = plt.subplots(1, 2, figsize=(IEEE_DOUBLE_COL, 3.0))
    gflops = np.array([r["gflops"] for r in rows])
    sizes = 30 + 600 * np.sqrt(gflops / max(gflops.max(), 1e-9))
    for ax, key, xlabel in ((axes[0], "params_m", "Trainable parameters (M, log scale)"),
                            (axes[1], "model_size_mb", "Checkpoint size (MB, log scale)")):
        x = np.array([r[key] for r in rows])
        y = np.array([r["macro_f1"] for r in rows])
        order = np.argsort(x)
        frontier, best = [], -np.inf
        for i in order:
            if y[i] > best:
                frontier.append(i)
                best = y[i]
        ax.plot(x[frontier], y[frontier], color=NEUTRAL, lw=0.8, ls="--", zorder=1, label="Pareto frontier")
        for i, r in enumerate(rows):
            is_ours = highlight is not None and r.get("model") == highlight
            color = PALETTE[1] if is_ours else PALETTE[0] if r.get("family") == "baseline" else PALETTE[2]
            ax.scatter(x[i], y[i], s=sizes[i], color=color, alpha=0.75, edgecolors="white", linewidths=1.5,
                       marker="*" if is_ours else "o", zorder=3)
            # Alternate label offsets so that neighbouring points stay legible.
            offset = (6, 6) if i % 2 == 0 else (6, -10)
            ax.annotate(r["display_name"], (x[i], y[i]), textcoords="offset points", xytext=offset,
                        fontsize=6, color="#111111", fontweight="bold" if is_ours else "normal")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(x.min() / 1.6, x.max() * 2.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Test macro-F1 (%)")
        pad = (y.max() - y.min()) * 0.15 + 0.5
        ax.set_ylim(y.min() - pad, y.max() + 2 * pad)
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=PALETTE[0], label="Baseline"),
        plt.Line2D([], [], marker="o", ls="", color=PALETTE[2], label="Ablation"),
        plt.Line2D([], [], marker="*", ls="", color=PALETTE[1], markersize=9, label="Att-ConvKAN (ours)"),
        plt.Line2D([], [], color=NEUTRAL, ls="--", label="Pareto frontier"),
    ]
    axes[0].legend(handles=handles, loc="lower right")
    axes[0].set_title("(a) Accuracy-parameter trade-off (bubble area $\\propto$ GFLOPs)")
    axes[1].set_title("(b) Accuracy-storage trade-off")
    fig.tight_layout()
    return save_figure(fig, out_dir, stem)


def plot_training_dynamics(
    histories: Dict[str, List[Dict[str, float]]],
    out_dir: str | Path,
    stem: str = "fig6_training_dynamics",
    highlight: Optional[str] = None,
) -> List[Path]:
    """Train/val loss, val macro-F1 and val AUROC per epoch for several models."""
    set_ieee_style()
    fig, axes = plt.subplots(1, 3, figsize=(IEEE_DOUBLE_COL, 2.6))
    for k, (name, hist) in enumerate(histories.items()):
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        is_ours = name == highlight
        color = PALETTE[1] if is_ours else PALETTE[[0, 2, 3, 4, 5, 6][k % 6]]
        lw = 1.8 if is_ours else 1.0
        mk = MARKERS[k % len(MARKERS)]
        axes[0].plot(ep, [h["train_loss"] for h in hist], color=color, lw=lw, ls="--", alpha=0.8)
        axes[0].plot(ep, [h["val_loss"] for h in hist], color=color, lw=lw, marker=mk, ms=3, label=name)
        axes[1].plot(ep, [100 * h["val_macro_f1"] for h in hist], color=color, lw=lw, marker=mk, ms=3, label=name)
        axes[2].plot(ep, [100 * h["val_auroc"] for h in hist], color=color, lw=lw, marker=mk, ms=3, label=name)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="(a) Loss (dashed: train, solid: val)")
    axes[1].set(xlabel="Epoch", ylabel="Val. macro-F1 (%)", title="(b) Validation macro-F1")
    axes[2].set(xlabel="Epoch", ylabel="Val. AUROC (%)", title="(c) Validation AUROC")
    handles, labels_ = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=min(6, len(labels_)), bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save_figure(fig, out_dir, stem)

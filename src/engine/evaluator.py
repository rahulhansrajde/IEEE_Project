"""Clinical evaluation suite.

Computes accuracy, balanced accuracy, macro/weighted F1 (with bootstrap 95 % CI),
macro precision / recall / specificity, one-vs-rest macro AUROC, Cohen's kappa,
expected calibration error (15 bins) and split-conformal prediction coverage /
set size, plus full per-class metrics.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

__all__ = [
    "predict",
    "expected_calibration_error",
    "multiclass_auroc",
    "specificity_per_class",
    "bootstrap_ci",
    "fit_temperature",
    "paired_bootstrap_delta",
    "conformal_calibrate",
    "conformal_evaluate",
    "apply_temperature",
    "compute_metrics",
    "evaluate_predictions",
]


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_embeddings: bool = False,
    amp: bool = False,
) -> Dict[str, np.ndarray]:
    """Run ``model`` over ``loader`` and return logits, probabilities and labels."""
    from src.models.baselines import extract_embeddings

    model.eval()
    logits_all, labels_all, emb_all = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)
            if return_embeddings:
                emb_all.append(extract_embeddings(model, images).float().cpu())
        logits_all.append(logits.float().cpu())
        labels_all.append(torch.as_tensor(labels).view(-1).cpu())
    logits = torch.cat(logits_all)
    out = {
        "logits": logits.numpy(),
        "probs": torch.softmax(logits, dim=1).numpy(),
        "labels": torch.cat(labels_all).numpy().astype(np.int64),
    }
    if return_embeddings:
        out["embeddings"] = torch.cat(emb_all).numpy()
    return out


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: ``sum_b |B_b|/N * |acc(B_b) - conf(B_b)|`` over equal-width bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def multiclass_auroc(labels: np.ndarray, probs: np.ndarray, num_classes: int) -> Dict[str, Any]:
    """Macro one-vs-rest AUROC and per-class AUROC (NaN for absent classes)."""
    per_class = []
    for c in range(num_classes):
        y = (labels == c).astype(int)
        if y.min() == y.max():
            per_class.append(float("nan"))
        else:
            per_class.append(float(roc_auc_score(y, probs[:, c])))
    valid = [a for a in per_class if not np.isnan(a)]
    return {"macro": float(np.mean(valid)) if valid else float("nan"), "per_class": per_class}


def specificity_per_class(cm: np.ndarray) -> np.ndarray:
    """``TN / (TN + FP)`` for each class of a confusion matrix."""
    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
    return spec


def bootstrap_ci(
    labels: np.ndarray,
    probs: np.ndarray,
    metric: str = "macro_f1",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap confidence interval of a test-set metric."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    num_classes = probs.shape[1]
    preds = probs.argmax(axis=1)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y, p = labels[idx], preds[idx]
        if metric == "macro_f1":
            values.append(f1_score(y, p, average="macro", labels=np.arange(num_classes), zero_division=0))
        elif metric == "accuracy":
            values.append(accuracy_score(y, p))
        elif metric == "balanced_accuracy":
            values.append(balanced_accuracy_score(y, p))
        elif metric == "auroc":
            values.append(multiclass_auroc(y, probs[idx], num_classes)["macro"])
        else:
            raise ValueError(f"Unsupported bootstrap metric '{metric}'")
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return {
        "lower": float(np.percentile(values, 100 * alpha / 2)),
        "upper": float(np.percentile(values, 100 * (1 - alpha / 2))),
        "std": float(values.std()),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> float:
    """Post-hoc temperature scaling (Guo et al., 2017): ``T = argmin NLL(softmax(z / T))``.

    ``logits`` may also be log-probabilities, which differ from logits only by a
    per-sample constant and therefore give the same temperature.
    """
    z = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = torch.as_tensor(logits, dtype=torch.float64) / temperature
    return torch.softmax(z, dim=1).numpy()


def paired_bootstrap_delta(
    labels: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired bootstrap of the macro-F1 / balanced-accuracy difference ``A - B``.

    Both models are evaluated on the *same* resampled test images, so the
    interval accounts for the correlation between their errors. ``p_value`` is
    the two-sided bootstrap p-value of the macro-F1 difference.
    """
    rng = np.random.default_rng(seed)
    n, k = len(labels), probs_a.shape[1]
    cls = np.arange(k)
    pa, pb = probs_a.argmax(1), probs_b.argmax(1)
    d_f1, d_bacc = [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        d_f1.append(f1_score(y, pa[idx], average="macro", labels=cls, zero_division=0)
                    - f1_score(y, pb[idx], average="macro", labels=cls, zero_division=0))
        d_bacc.append(balanced_accuracy_score(y, pa[idx]) - balanced_accuracy_score(y, pb[idx]))
    d_f1, d_bacc = np.asarray(d_f1), np.asarray(d_bacc)
    observed = f1_score(labels, pa, average="macro", labels=cls, zero_division=0) - f1_score(
        labels, pb, average="macro", labels=cls, zero_division=0)
    p_value = min(1.0, 2 * min((d_f1 <= 0).mean(), (d_f1 >= 0).mean()))
    return {
        "delta_macro_f1": 100 * float(observed),
        "delta_macro_f1_ci_lower": 100 * float(np.percentile(d_f1, 2.5)),
        "delta_macro_f1_ci_upper": 100 * float(np.percentile(d_f1, 97.5)),
        "delta_balanced_accuracy": 100 * float(balanced_accuracy_score(labels, pa) - balanced_accuracy_score(labels, pb)),
        "delta_balanced_accuracy_ci_lower": 100 * float(np.percentile(d_bacc, 2.5)),
        "delta_balanced_accuracy_ci_upper": 100 * float(np.percentile(d_bacc, 97.5)),
        "p_value": float(p_value),
    }


def conformal_calibrate(cal_probs: np.ndarray, cal_labels: np.ndarray, alpha: float = 0.1) -> float:
    """Split-conformal (LAC) threshold on non-conformity ``s = 1 - p_y``."""
    n = len(cal_labels)
    scores = 1.0 - cal_probs[np.arange(n), cal_labels]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def conformal_evaluate(probs: np.ndarray, labels: np.ndarray, qhat: float) -> Dict[str, float]:
    """Empirical coverage and mean prediction-set size of ``{c : 1 - p_c <= qhat}``."""
    sets = (1.0 - probs) <= qhat
    covered = sets[np.arange(len(labels)), labels]
    return {"coverage": float(covered.mean()), "avg_set_size": float(sets.sum(axis=1).mean())}


def compute_metrics(labels: np.ndarray, probs: np.ndarray, num_classes: int, ece_bins: int = 15) -> Dict[str, float]:
    """Fast scalar metrics (used for per-epoch validation)."""
    preds = probs.argmax(axis=1)
    cls = np.arange(num_classes)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=cls, zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", labels=cls, zero_division=0)),
        "auroc": multiclass_auroc(labels, probs, num_classes)["macro"],
        "ece": expected_calibration_error(probs, labels, ece_bins),
    }


def evaluate_predictions(
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str],
    n_bootstrap: int = 1000,
    ece_bins: int = 15,
    seed: int = 0,
    cal_labels: Optional[np.ndarray] = None,
    cal_probs: Optional[np.ndarray] = None,
    conformal_alpha: float = 0.1,
) -> Dict[str, Any]:
    """Full evaluation report. Percent-type metrics are returned in %, kappa in [-1, 1]."""
    num_classes = len(class_names)
    cls = np.arange(num_classes)
    preds = probs.argmax(axis=1)
    cm = confusion_matrix(labels, preds, labels=cls)
    prec, rec, f1, support = precision_recall_fscore_support(labels, preds, labels=cls, zero_division=0)
    spec = specificity_per_class(cm)
    auroc = multiclass_auroc(labels, probs, num_classes)

    report: Dict[str, Any] = {
        "accuracy": 100 * accuracy_score(labels, preds),
        "balanced_accuracy": 100 * balanced_accuracy_score(labels, preds),
        "macro_f1": 100 * f1_score(labels, preds, average="macro", labels=cls, zero_division=0),
        "weighted_f1": 100 * f1_score(labels, preds, average="weighted", labels=cls, zero_division=0),
        "macro_precision": 100 * float(np.mean(prec)),
        "macro_recall": 100 * float(np.mean(rec)),
        "macro_specificity": 100 * float(np.nanmean(spec)),
        "auroc": 100 * auroc["macro"],
        "cohen_kappa": float(cohen_kappa_score(labels, preds)),
        "ece": 100 * expected_calibration_error(probs, labels, ece_bins),
        "n_test": int(len(labels)),
    }
    for metric in ("macro_f1", "accuracy", "auroc"):
        ci = bootstrap_ci(labels, probs, metric, n_bootstrap=n_bootstrap, seed=seed)
        report[f"{metric}_ci_lower"] = 100 * ci["lower"]
        report[f"{metric}_ci_upper"] = 100 * ci["upper"]
        report[f"{metric}_ci_std"] = 100 * ci["std"]

    if cal_labels is not None and cal_probs is not None:
        qhat = conformal_calibrate(cal_probs, cal_labels, conformal_alpha)
        conf = conformal_evaluate(probs, labels, qhat)
        report["conformal_alpha"] = conformal_alpha
        report["conformal_qhat"] = qhat
        report["conformal_coverage"] = 100 * conf["coverage"]
        report["conformal_set_size"] = conf["avg_set_size"]

    per_class: List[Dict[str, Any]] = []
    for c in cls:
        per_class.append(
            {
                "class": class_names[c],
                "precision": 100 * float(prec[c]),
                "recall": 100 * float(rec[c]),
                "f1": 100 * float(f1[c]),
                "specificity": 100 * float(spec[c]),
                "auroc": 100 * auroc["per_class"][c],
                "support": int(support[c]),
            }
        )
    report["per_class"] = per_class
    report["confusion_matrix"] = cm.tolist()
    return report

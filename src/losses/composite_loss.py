"""Class-balanced focal cross-entropy with KAN spline regularisation.

    L_total = L_FocalCE(z, y; alpha, gamma, epsilon) + lambda_l1 * L_L1 + lambda_ent * L_H

with label-smoothed targets ``q = (1 - eps) * onehot(y) + eps / K`` and

    L_FocalCE = - alpha_y * sum_c q_c (1 - p_c)^gamma log p_c .
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["compute_class_weights", "FocalCrossEntropy", "CompositeKANLoss"]


def compute_class_weights(
    class_counts: Sequence[int], method: str = "effective_number", beta: float = 0.999
) -> torch.Tensor:
    """Per-class weights normalised to mean 1.

    Args:
        class_counts: number of training samples per class.
        method: ``"inverse"`` (``N / (K n_c)``), ``"inverse_sqrt"`` or
            ``"effective_number"`` (Cui et al., CVPR 2019:
            ``(1 - beta) / (1 - beta^{n_c})``), or ``"none"``.
        beta: hyper-parameter of the effective-number weighting.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    if np.any(counts <= 0):
        counts = np.maximum(counts, 1.0)
    if method == "inverse":
        weights = counts.sum() / (len(counts) * counts)
    elif method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
    elif method == "effective_number":
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    elif method == "none":
        weights = np.ones_like(counts)
    else:
        raise ValueError(f"Unknown class-weight method '{method}'")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalCrossEntropy(nn.Module):
    """Multi-class focal loss with class weights and label smoothing."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        num_classes = logits.size(1)
        log_p = F.log_softmax(logits, dim=1)
        p = log_p.exp()
        q = F.one_hot(targets, num_classes).float()
        if self.label_smoothing > 0:
            q = q * (1.0 - self.label_smoothing) + self.label_smoothing / num_classes
        focal = (1.0 - p).clamp_min(0.0).pow(self.gamma)
        per_sample = -(q * focal * log_p).sum(dim=1)
        if self.class_weights is not None:
            per_sample = per_sample * self.class_weights[targets]
        return per_sample.mean()


class CompositeKANLoss(nn.Module):
    """Focal-CE + ``lambda_l1`` * spline L1 + ``lambda_ent`` * spline entropy.

    The spline terms are obtained from ``model.get_kan_regularization_components()``;
    models without KAN layers only receive the focal term.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        lambda_l1: float = 1.0,
        lambda_ent: float = 0.1,
    ) -> None:
        super().__init__()
        self.focal = FocalCrossEntropy(class_weights, gamma, label_smoothing)
        self.lambda_l1 = lambda_l1
        self.lambda_ent = lambda_ent

    def regularization(self, model: Optional[nn.Module], device: torch.device) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device)
        if model is None or not hasattr(model, "get_kan_regularization_components"):
            return {"spline_l1": zero, "spline_entropy": zero}
        l1, ent = model.get_kan_regularization_components()
        return {"spline_l1": l1, "spline_entropy": ent}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[nn.Module] = None,
        targets_b: Optional[torch.Tensor] = None,
        lam: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Return a dict with ``loss`` (total) and its individual components.

        ``targets_b`` / ``lam`` support Mixup/CutMix: the classification term
        becomes ``lam * L(y_a) + (1 - lam) * L(y_b)``.
        """
        cls_loss = self.focal(logits, targets)
        if targets_b is not None and lam < 1.0:
            cls_loss = lam * cls_loss + (1.0 - lam) * self.focal(logits, targets_b)
        reg = self.regularization(model, logits.device)
        total = cls_loss
        if self.lambda_l1 > 0:
            total = total + self.lambda_l1 * reg["spline_l1"]
        if self.lambda_ent > 0:
            total = total + self.lambda_ent * reg["spline_entropy"]
        return {
            "loss": total,
            "cls_loss": cls_loss.detach(),
            "spline_l1": reg["spline_l1"].detach(),
            "spline_entropy": reg["spline_entropy"].detach(),
        }

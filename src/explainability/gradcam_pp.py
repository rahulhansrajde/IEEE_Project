"""Grad-CAM++ (Chattopadhay et al., WACV 2018) for CNN and ViT feature maps.

For a class score ``y^c`` and feature maps ``A^k`` of the target layer:

    alpha^{kc}_{ij} = (d y / d A)^2 / ( 2 (d y / d A)^2 + sum_ab A_ab (d y / d A)^3 )
    w^c_k          = sum_ij alpha^{kc}_{ij} ReLU(d y^c / d A^k_ij)
    L^c            = ReLU( sum_k w^c_k A^k )
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GradCAMPlusPlus", "overlay_heatmap"]


def _tokens_to_map(t: torch.Tensor) -> torch.Tensor:
    """Convert ViT tokens ``(B, N, D)`` (with or without CLS) to ``(B, D, h, w)``."""
    b, n, d = t.shape
    side = int(round(np.sqrt(n)))
    if side * side != n:
        side = int(round(np.sqrt(n - 1)))
        t = t[:, n - side * side :, :]
    return t.transpose(1, 2).reshape(b, d, side, side)


class GradCAMPlusPlus:
    """Grad-CAM++ attached to ``target_layer`` of ``model``.

    Gradients are captured with tensor hooks (robust to in-place activations).
    Use as a context manager or call :meth:`remove` when done.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, _module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAMPlusPlus":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx: Optional[int | torch.Tensor] = None) -> Tuple[np.ndarray, torch.Tensor]:
        """Return ``(cams, logits)``; ``cams`` is ``(B, H, W)`` in [0, 1] at input resolution."""
        was_training = self.model.training
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                target = logits.argmax(dim=1)
            elif isinstance(class_idx, int):
                target = torch.full((x.size(0),), class_idx, device=x.device, dtype=torch.long)
            else:
                target = class_idx.to(x.device).long().view(-1)
            score = logits.gather(1, target.view(-1, 1)).sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()
        acts, grads = self.activations.detach(), self.gradients.detach()
        if acts.dim() == 3:
            acts, grads = _tokens_to_map(acts), _tokens_to_map(grads)
        acts, grads = acts.float(), grads.float()
        g2, g3 = grads.pow(2), grads.pow(3)
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom = 2.0 * g2 + sum_acts * g3
        denom = torch.where(denom.abs() > 1e-12, denom, torch.ones_like(denom))
        alpha = g2 / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.view(cam.size(0), -1)
        mn, mx = flat.min(dim=1, keepdim=True)[0], flat.max(dim=1, keepdim=True)[0]
        span = mx - mn
        # A (numerically) constant map carries no localisation: return zeros
        # instead of amplifying interpolation round-off into speckle.
        constant = span <= 1e-5 * mx.abs().clamp_min(1e-8)
        cam = torch.where(constant, torch.zeros_like(flat), (flat - mn) / (span + 1e-8)).view_as(cam)
        self.model.train(was_training)
        return cam.cpu().numpy(), logits.detach()


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45, cmap: str = "jet") -> np.ndarray:
    """Blend an RGB image in [0, 1] (``H x W x 3``) with a heatmap in [0, 1]."""
    colormap = matplotlib.colormaps[cmap]
    heat = colormap(np.clip(cam, 0.0, 1.0))[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heat, 0.0, 1.0)

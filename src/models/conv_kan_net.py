"""Proposed Attention-Guided Convolutional KAN network (Att-ConvKAN).

Pipeline::

    image -> pretrained CNN backbone (feature map F)
          -> KANConvBottleneck (spline-based spatial refinement)
          -> CoordinateChannelAttention (lesion-focused re-weighting)
          -> KANClassificationHead (GAP -> LN -> Dropout -> KANLinear -> LN -> KANLinear)

The same class also instantiates the ablation variants by switching the
bottleneck, the attention block and the head type on or off.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import timm
import torch
import torch.nn as nn

from .attention import CoordinateChannelAttention
from .kan_layers import KANConv2d, KANConvBottleneck, KANLinear, iter_kan_layers

__all__ = [
    "create_timm_model",
    "KANClassificationHead",
    "MLPClassificationHead",
    "AttConvKANNet",
]

_LOG = logging.getLogger(__name__)


def create_timm_model(name: str, pretrained: bool = True, **kwargs) -> nn.Module:
    """Create a ``timm`` model, falling back to random init if weights are unreachable."""
    if pretrained:
        try:
            return timm.create_model(name, pretrained=True, **kwargs)
        except Exception as exc:  # network / hub failures must not stop the pipeline
            _LOG.warning("Could not load pretrained weights for %s (%s); using random init.", name, exc)
    return timm.create_model(name, pretrained=False, **kwargs)


class KANClassificationHead(nn.Module):
    """GAP -> LayerNorm -> Dropout -> KANLinear(in, hidden) -> LayerNorm -> KANLinear(hidden, K)."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm1 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.kan1 = KANLinear(in_dim, hidden_dim, grid_size, spline_order, input_norm="none")
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.kan2 = KANLinear(hidden_dim, num_classes, grid_size, spline_order, input_norm="none")

    def pre_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm1(self.pool(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.pre_logits(x))
        return self.kan2(self.norm2(self.kan1(z)))


class MLPClassificationHead(nn.Module):
    """GAP -> LayerNorm -> Dropout -> Linear -> ReLU -> Dropout -> Linear."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 1024, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm1 = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def pre_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm1(self.pool(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.pre_logits(x))
        return self.fc2(self.dropout2(self.act(self.fc1(z))))


class AttConvKANNet(nn.Module):
    """Att-ConvKAN and its ablation variants.

    Args:
        num_classes: number of diagnostic classes.
        backbone: any ``timm`` model name exposing ``forward_features``.
        pretrained: load ImageNet weights for the backbone.
        use_kan_bottleneck: insert the :class:`KANConvBottleneck` refinement block.
        use_attention: insert the :class:`CoordinateChannelAttention` block.
        head_type: ``"kan"`` or ``"mlp"``.
        kan_hidden: hidden width of the KAN head.
        mlp_hidden: hidden width of the MLP head (ablations).
        grid_size, spline_order: B-spline configuration of all KAN layers.
        dropout: dropout rate inside the head.
        attention_reduction: channel reduction ratio of the attention block.
        bottleneck_reduction: channel reduction ratio of the bottleneck.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        use_kan_bottleneck: bool = True,
        use_attention: bool = True,
        head_type: str = "kan",
        kan_hidden: int = 64,
        mlp_hidden: int = 1024,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.2,
        attention_reduction: int = 16,
        bottleneck_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone = create_timm_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        dummy = self._prune_unused_backbone_modules()
        if dummy.dim() != 4:
            raise ValueError(f"Backbone {backbone} must return a 4-D feature map, got {tuple(dummy.shape)}")
        feat_dim = dummy.size(1)
        self.feature_dim = feat_dim

        self.bottleneck = (
            KANConvBottleneck(feat_dim, feat_dim, reduction=bottleneck_reduction,
                              grid_size=grid_size, spline_order=spline_order)
            if use_kan_bottleneck
            else nn.Identity()
        )
        self.attention = (
            CoordinateChannelAttention(feat_dim, reduction=attention_reduction)
            if use_attention
            else None
        )
        if head_type == "kan":
            self.head = KANClassificationHead(feat_dim, num_classes, kan_hidden, grid_size, spline_order, dropout)
        elif head_type == "mlp":
            self.head = MLPClassificationHead(feat_dim, num_classes, mlp_hidden, dropout)
        else:
            raise ValueError(f"Unknown head_type '{head_type}'")
        self.head_type = head_type
        self.num_classes = num_classes
        self.config = dict(
            backbone=backbone,
            use_kan_bottleneck=use_kan_bottleneck,
            use_attention=use_attention,
            head_type=head_type,
            kan_hidden=kan_hidden,
            mlp_hidden=mlp_hidden,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def _prune_unused_backbone_modules(self) -> torch.Tensor:
        """Drop backbone children not used by ``forward_features`` (e.g. timm's
        MobileNetV3 ``conv_head``) so that parameter counts are exact."""
        was_training = self.backbone.training
        self.backbone.eval()
        self.backbone.zero_grad(set_to_none=True)
        out = self.backbone.forward_features(torch.zeros(1, 3, 64, 64))
        out.float().sum().backward()
        for name, child in list(self.backbone.named_children()):
            params = list(child.parameters())
            if params and all(p.grad is None for p in params):
                setattr(self.backbone, name, nn.Identity())
        self.backbone.zero_grad(set_to_none=True)
        self.backbone.train(was_training)
        return out.detach()

    # ------------------------------------------------------------------ forward
    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Refined spatial feature map before the classification head."""
        f = self.backbone.forward_features(x)
        f = self.bottleneck(f)
        if self.attention is not None:
            f = self.attention(f)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_feature_map(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled, normalised embedding fed to the classification head."""
        return self.head.pre_logits(self.forward_feature_map(x))

    # ------------------------------------------------------- KAN utilities
    def kan_layers(self) -> List[nn.Module]:
        return list(iter_kan_layers(self))

    def get_kan_regularization_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sum of ``(L1, entropy)`` over every KAN layer in the network."""
        device = next(self.parameters()).device
        l1 = torch.zeros((), device=device)
        ent = torch.zeros((), device=device)
        for layer in self.kan_layers():
            layer_l1, layer_ent = layer.regularization_components()
            l1 = l1 + layer_l1
            ent = ent + layer_ent
        return l1, ent

    def get_kan_regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        l1, ent = self.get_kan_regularization_components()
        return regularize_activation * l1 + regularize_entropy * ent

    # ------------------------------------------------ bookkeeping helpers
    @property
    def head_module_names(self) -> List[str]:
        return ["head"]

    @property
    def new_module_names(self) -> List[str]:
        """Randomly initialised (non-backbone) sub-modules."""
        names = ["head"]
        if not isinstance(self.bottleneck, nn.Identity):
            names.append("bottleneck")
        if self.attention is not None:
            names.append("attention")
        return names

    @property
    def gradcam_target_layer(self) -> nn.Module:
        if self.attention is not None:
            return self.attention
        if not isinstance(self.bottleneck, nn.Identity):
            return self.bottleneck
        return self.backbone

    def kan_statistics(self, sparsity_threshold: float = 0.01) -> Dict[str, float]:
        """Interpretability statistics of the learned splines.

        ``edge_sparsity`` is the fraction of KAN-head edges whose importance
        is below ``sparsity_threshold`` times the largest edge importance of
        their layer (i.e. edges that could be pruned).
        """
        stats: Dict[str, float] = {}
        layers = [m for m in self.head.modules() if isinstance(m, KANLinear)]
        if not layers:
            return stats
        n_edges, n_sparse, ent = 0, 0, 0.0
        with torch.no_grad():
            for layer in layers:
                e = layer.edge_l1()
                n_edges += e.numel()
                n_sparse += int((e < sparsity_threshold * e.max()).sum())
                ent += float(layer.regularization_components()[1])
            conv_layers = [m for m in self.modules() if isinstance(m, KANConv2d)]
            conv_l1 = sum(float(m.edge_l1().mean()) for m in conv_layers)
        stats["kan_head_edges"] = float(n_edges)
        stats["kan_edge_sparsity"] = n_sparse / max(n_edges, 1)
        stats["kan_spline_entropy"] = ent / len(layers)
        stats["kan_head_mean_l1"] = float(
            sum(float(layer.edge_l1().mean()) for layer in layers) / len(layers)
        )
        stats["kan_conv_mean_l1"] = conv_l1 / max(len(conv_layers), 1)
        return stats

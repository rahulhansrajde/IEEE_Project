"""Unified model registry: comparative baselines, ablation variants and the proposed model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .conv_kan_net import AttConvKANNet, create_timm_model

__all__ = [
    "ModelSpec",
    "MODEL_SPECS",
    "BASELINE_MODELS",
    "ABLATION_MODELS",
    "PROPOSED_MODEL",
    "PatchTransformer",
    "build_model",
    "get_model_spec",
    "get_head_module_names",
    "get_new_module_names",
    "get_gradcam_target_layer",
    "extract_embeddings",
]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """Static description of a registry entry."""

    name: str
    display_name: str
    family: str  # "baseline", "ablation" or "proposed"
    timm_name: Optional[str] = None
    kan_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_spline_regularization: bool = False
    components: Dict[str, bool] = field(default_factory=dict)


_FULL = dict(use_kan_bottleneck=True, use_attention=True, head_type="kan")

MODEL_SPECS: Dict[str, ModelSpec] = {
    "resnet50": ModelSpec("resnet50", "ResNet-50", "baseline", timm_name="resnet50"),
    "densenet121": ModelSpec("densenet121", "DenseNet-121", "baseline", timm_name="densenet121"),
    "mobilenetv3_small": ModelSpec(
        "mobilenetv3_small", "MobileNetV3-S", "baseline", timm_name="mobilenetv3_small_100"
    ),
    "vit_tiny": ModelSpec("vit_tiny", "ViT-Tiny", "baseline", timm_name="vit_tiny_patch16_224"),
    "convnext_tiny": ModelSpec("convnext_tiny", "ConvNeXt-T", "baseline", timm_name="convnext_tiny"),
    "ablation_1_backbone_mlp": ModelSpec(
        "ablation_1_backbone_mlp",
        "Backbone + MLP head",
        "ablation",
        kan_kwargs=dict(use_kan_bottleneck=False, use_attention=False, head_type="mlp"),
        components=dict(kan_conv=False, attention=False, kan_head=False, spline_reg=False),
    ),
    "ablation_2_backbone_kan_no_att": ModelSpec(
        "ablation_2_backbone_kan_no_att",
        "Backbone + KAN head",
        "ablation",
        kan_kwargs=dict(use_kan_bottleneck=False, use_attention=False, head_type="kan"),
        use_spline_regularization=True,
        components=dict(kan_conv=False, attention=False, kan_head=True, spline_reg=True),
    ),
    "ablation_3_backbone_att_mlp": ModelSpec(
        "ablation_3_backbone_att_mlp",
        "Backbone + Att. + MLP head",
        "ablation",
        kan_kwargs=dict(use_kan_bottleneck=False, use_attention=True, head_type="mlp"),
        components=dict(kan_conv=False, attention=True, kan_head=False, spline_reg=False),
    ),
    "ablation_4_full_no_spline_reg": ModelSpec(
        "ablation_4_full_no_spline_reg",
        "Full w/o spline reg.",
        "ablation",
        kan_kwargs=_FULL,
        use_spline_regularization=False,
        components=dict(kan_conv=True, attention=True, kan_head=True, spline_reg=False),
    ),
    "att_convkan_proposed": ModelSpec(
        "att_convkan_proposed",
        "Att-ConvKAN (Ours)",
        "proposed",
        kan_kwargs=_FULL,
        use_spline_regularization=True,
        components=dict(kan_conv=True, attention=True, kan_head=True, spline_reg=True),
    ),
}

BASELINE_MODELS: List[str] = ["resnet50", "densenet121", "mobilenetv3_small", "vit_tiny", "convnext_tiny"]
ABLATION_MODELS: List[str] = [
    "ablation_1_backbone_mlp",
    "ablation_2_backbone_kan_no_att",
    "ablation_3_backbone_att_mlp",
    "ablation_4_full_no_spline_reg",
]
PROPOSED_MODEL = "att_convkan_proposed"

# Classification-head sub-modules of the timm baselines.
_TIMM_HEADS: Dict[str, List[str]] = {
    "resnet50": ["fc"],
    "densenet121": ["classifier"],
    "mobilenetv3_small": ["conv_head", "norm_head", "classifier"],
    "vit_tiny": ["fc_norm", "head"],
    "convnext_tiny": ["head"],
}


class PatchTransformer(nn.Module):
    """Minimal ViT used as fallback when ``timm`` cannot build ``vit_tiny``."""

    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.norm(self.blocks(x))

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


def get_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {sorted(MODEL_SPECS)}")
    return MODEL_SPECS[model_name]


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    image_size: int = 224,
    model_cfg: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Instantiate any registry entry.

    Args:
        model_name: key of :data:`MODEL_SPECS`.
        num_classes: number of output classes.
        pretrained: initialise backbones with ImageNet weights.
        image_size: input resolution (needed by the ViT positional embedding).
        model_cfg: optional overrides for the Att-ConvKAN family (backbone,
            kan_hidden, mlp_hidden, grid_size, spline_order, dropout, ...).
    """
    spec = get_model_spec(model_name)
    model_cfg = dict(model_cfg or {})
    if spec.timm_name is not None:
        kwargs: Dict[str, Any] = {"num_classes": num_classes}
        if model_name == "vit_tiny":
            kwargs["img_size"] = image_size
            try:
                model = create_timm_model(spec.timm_name, pretrained=pretrained, **kwargs)
            except Exception as exc:
                _LOG.warning("timm ViT unavailable (%s); using PatchTransformer fallback.", exc)
                patch = 16 if image_size % 16 == 0 else 8
                model = PatchTransformer(num_classes, img_size=image_size, patch_size=patch)
        else:
            model = create_timm_model(spec.timm_name, pretrained=pretrained, **kwargs)
        model.registry_name = model_name
        return model

    allowed = {
        "backbone", "kan_hidden", "mlp_hidden", "grid_size", "spline_order",
        "dropout", "attention_reduction", "bottleneck_reduction",
    }
    kan_cfg = {k: v for k, v in model_cfg.items() if k in allowed}
    kan_cfg.update(spec.kan_kwargs)
    model = AttConvKANNet(num_classes=num_classes, pretrained=pretrained, **kan_cfg)
    model.registry_name = model_name
    return model


def _registry_name(model: nn.Module, model_name: Optional[str]) -> Optional[str]:
    return model_name or getattr(model, "registry_name", None)


def get_head_module_names(model: nn.Module, model_name: Optional[str] = None) -> List[str]:
    """Names of the top-level sub-modules forming the classification head."""
    if hasattr(model, "head_module_names"):
        return list(model.head_module_names)
    name = _registry_name(model, model_name)
    if name in _TIMM_HEADS:
        return [n for n in _TIMM_HEADS[name] if hasattr(model, n)]
    if isinstance(model, PatchTransformer):
        return ["head"]
    for candidate in ("fc", "classifier", "head"):
        if hasattr(model, candidate):
            return [candidate]
    return []


def get_new_module_names(model: nn.Module, model_name: Optional[str] = None) -> List[str]:
    """Randomly initialised sub-modules (trained with the head learning rate)."""
    if hasattr(model, "new_module_names"):
        return list(model.new_module_names)
    name = _registry_name(model, model_name)
    if name == "mobilenetv3_small":
        return ["classifier"]  # conv_head is pretrained in timm
    return get_head_module_names(model, model_name)


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Last spatial feature layer used for Grad-CAM++."""
    if hasattr(model, "gradcam_target_layer"):
        return model.gradcam_target_layer
    for attr in ("layer4", "features", "stages", "blocks"):
        if hasattr(model, attr):
            mod = getattr(model, attr)
            if attr in ("stages", "blocks") and isinstance(mod, nn.Sequential):
                return mod[-1]
            return mod
    raise ValueError(f"Cannot determine a Grad-CAM target layer for {type(model).__name__}")


@torch.no_grad()
def extract_embeddings(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Penultimate (pre-logit) embedding of any registry model."""
    if hasattr(model, "extract_features"):
        return model.extract_features(x)
    if hasattr(model, "forward_features") and hasattr(model, "forward_head"):
        return model.forward_head(model.forward_features(x), pre_logits=True).flatten(1)
    raise ValueError(f"Model {type(model).__name__} does not expose embeddings")

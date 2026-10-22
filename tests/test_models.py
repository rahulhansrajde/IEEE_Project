"""Tests for the proposed Att-ConvKAN model, the 5 baselines and the 4 ablations."""

import pytest
import torch

from src.losses import CompositeKANLoss, FocalCrossEntropy, compute_class_weights
from src.models import (
    ABLATION_MODELS,
    BASELINE_MODELS,
    MODEL_SPECS,
    PROPOSED_MODEL,
    AttConvKANNet,
    CoordinateChannelAttention,
    KANConv2d,
    KANLinear,
    build_model,
    extract_embeddings,
    get_gradcam_target_layer,
    get_head_module_names,
)

IMG = 64
NUM_CLASSES = 7


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_registry_contains_all_models():
    assert len(BASELINE_MODELS) == 5
    assert len(ABLATION_MODELS) == 4
    for name in BASELINE_MODELS + ABLATION_MODELS + [PROPOSED_MODEL]:
        assert name in MODEL_SPECS


@pytest.mark.parametrize("name", BASELINE_MODELS)
def test_baselines_forward_backward(name):
    model = build_model(name, NUM_CLASSES, pretrained=False, image_size=IMG)
    x = torch.randn(2, 3, IMG, IMG)
    y = model(x)
    assert y.shape == (2, NUM_CLASSES)
    y.sum().backward()
    assert get_head_module_names(model, name)
    emb = extract_embeddings(model.eval(), x)
    assert emb.shape[0] == 2 and emb.dim() == 2
    assert get_gradcam_target_layer(model) is not None


@pytest.mark.parametrize("name", ABLATION_MODELS + [PROPOSED_MODEL])
def test_attconvkan_family(name):
    model = build_model(name, NUM_CLASSES, pretrained=False, image_size=IMG)
    assert isinstance(model, AttConvKANNet)
    spec = MODEL_SPECS[name]
    comps = spec.components
    assert (model.attention is not None) == comps["attention"]
    assert any(isinstance(m, KANConv2d) for m in model.modules()) == comps["kan_conv"]
    assert any(isinstance(m, KANLinear) for m in model.head.modules()) == comps["kan_head"]
    x = torch.randn(2, 3, IMG, IMG)
    out = model(x)
    assert out.shape == (2, NUM_CLASSES)
    loss = out.sum() + model.get_kan_regularization_loss()
    loss.backward()
    assert extract_embeddings(model.eval(), x).shape == (2, model.feature_dim)


def test_proposed_model_structure_and_regularization():
    model = build_model(PROPOSED_MODEL, NUM_CLASSES, pretrained=False, image_size=IMG)
    kan_layers = model.kan_layers()
    assert sum(isinstance(m, KANLinear) for m in kan_layers) == 2
    assert sum(isinstance(m, KANConv2d) for m in kan_layers) == 1
    assert model.head.kan1.in_features == model.feature_dim
    assert model.head.kan1.out_features == 64
    assert model.head.kan2.out_features == NUM_CLASSES
    l1, ent = model.get_kan_regularization_components()
    expected = sum(m.regularization_components()[0] for m in kan_layers)
    assert torch.allclose(l1, expected)
    assert float(ent) > 0
    stats = model.kan_statistics()
    assert 0.0 <= stats["kan_edge_sparsity"] <= 1.0


def test_kan_head_has_fewer_parameters_than_mlp_head():
    kan = build_model("ablation_2_backbone_kan_no_att", NUM_CLASSES, pretrained=False, image_size=IMG)
    mlp = build_model("ablation_1_backbone_mlp", NUM_CLASSES, pretrained=False, image_size=IMG)
    n_kan = sum(p.numel() for p in kan.head.parameters())
    n_mlp = sum(p.numel() for p in mlp.head.parameters())
    assert n_kan < n_mlp


def test_unused_backbone_modules_are_pruned():
    model = build_model(PROPOSED_MODEL, NUM_CLASSES, pretrained=False, image_size=IMG)
    assert isinstance(model.backbone.conv_head, torch.nn.Identity)


def test_coordinate_channel_attention_caches_map():
    att = CoordinateChannelAttention(32, reduction=4)
    x = torch.randn(2, 32, 6, 5)
    y = att(x)
    assert y.shape == x.shape
    assert att.last_attention_map.shape == (2, 6, 5)
    assert torch.all((att.last_attention_map >= 0) & (att.last_attention_map <= 1))


def test_class_weights_and_focal_loss():
    counts = [228, 359, 769, 80, 779, 4693, 99]
    for method in ("inverse", "inverse_sqrt", "effective_number", "none"):
        w = compute_class_weights(counts, method)
        assert w.shape == (7,)
        assert torch.isclose(w.mean(), torch.tensor(1.0))
    w = compute_class_weights(counts, "inverse")
    assert w[3] > w[5]  # rare dermatofibroma weighted above common nevi
    logits = torch.randn(16, 7, requires_grad=True)
    targets = torch.randint(0, 7, (16,))
    focal = FocalCrossEntropy(w, gamma=2.0, label_smoothing=0.05)(logits, targets)
    focal.backward()
    assert focal > 0 and logits.grad is not None
    ce_like = FocalCrossEntropy(None, gamma=0.0, label_smoothing=0.0)(logits.detach(), targets)
    assert torch.allclose(ce_like, torch.nn.functional.cross_entropy(logits.detach(), targets), atol=1e-5)


def test_composite_loss_adds_spline_terms():
    model = build_model(PROPOSED_MODEL, NUM_CLASSES, pretrained=False, image_size=IMG)
    x = torch.randn(2, 3, IMG, IMG)
    t = torch.tensor([0, 3])
    logits = model(x)
    with_reg = CompositeKANLoss(lambda_l1=1.0, lambda_ent=0.1)(logits, t, model)
    no_reg = CompositeKANLoss(lambda_l1=0.0, lambda_ent=0.0)(logits, t, model)
    assert with_reg["loss"] > no_reg["loss"]
    expected = no_reg["loss"] + with_reg["spline_l1"] + 0.1 * with_reg["spline_entropy"]
    assert torch.allclose(with_reg["loss"], expected, atol=1e-5)
    mixed = CompositeKANLoss()(logits, t, model, targets_b=torch.tensor([1, 2]), lam=0.3)
    assert torch.isfinite(mixed["loss"])

"""Unit tests for the vectorised B-spline KAN layers."""

import math

import pytest
import torch

from src.models.kan_layers import (
    KANConv2d,
    KANConvBottleneck,
    KANLinear,
    b_spline_basis,
    make_uniform_grid,
)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_uniform_grid_has_extended_knots():
    grid = make_uniform_grid(5, 3, (-1.0, 1.0))
    assert grid.shape == (5 + 2 * 3 + 1,)
    h = 2.0 / 5
    assert torch.allclose(grid[3], torch.tensor(-1.0))
    assert torch.allclose(grid[-4], torch.tensor(1.0))
    assert torch.allclose(grid[1:] - grid[:-1], torch.full((11,), h))


def test_b_spline_basis_shape_and_partition_of_unity():
    layer = KANLinear(4, 3, grid_size=5, spline_order=3)
    x = torch.rand(64, 4) * 1.98 - 0.99
    bases = layer.b_splines(x)
    assert bases.shape == (64, 4, 5 + 3)
    assert torch.all(bases >= 0)
    # Cubic B-splines on an extended uniform knot vector sum to one in the interior.
    assert torch.allclose(bases.sum(-1), torch.ones(64, 4), atol=1e-5)


def test_b_spline_basis_matches_scalar_cox_de_boor():
    grid = make_uniform_grid(5, 3)
    k = 3

    def cox_de_boor(x, i, p):
        if p == 0:
            return 1.0 if grid[i] <= x < grid[i + 1] else 0.0
        left = (x - grid[i]) / (grid[i + p] - grid[i]) * cox_de_boor(x, i, p - 1)
        right = (grid[i + p + 1] - x) / (grid[i + p + 1] - grid[i + 1]) * cox_de_boor(x, i + 1, p - 1)
        return float(left + right)

    xs = torch.tensor([-0.95, -0.3, 0.0, 0.41, 0.87])
    vec = b_spline_basis(xs, grid, k)
    for a, x in enumerate(xs):
        ref = torch.tensor([cox_de_boor(float(x), i, k) for i in range(len(grid) - 1 - k)])
        assert torch.allclose(vec[a], ref.float(), atol=1e-5)


def test_kanlinear_forward_backward():
    layer = KANLinear(10, 6)
    x = torch.randn(8, 10, requires_grad=True)
    y = layer(x)
    assert y.shape == (8, 6)
    y.pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in (layer.base_weight, layer.spline_weight, layer.spline_scaler):
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_kanlinear_parameter_shapes_and_count():
    layer = KANLinear(7, 5, grid_size=5, spline_order=3, input_norm="none")
    assert layer.base_weight.shape == (5, 7)
    assert layer.spline_weight.shape == (5, 7, 8)
    assert layer.spline_scaler.shape == (5, 7)
    assert sum(p.numel() for p in layer.parameters()) == 5 * 7 * (1 + 8 + 1)


def test_kanlinear_supports_extra_leading_dims_and_norms():
    for norm in ("layernorm", "batchnorm", "none"):
        layer = KANLinear(12, 4, input_norm=norm)
        out = layer(torch.randn(2, 3, 12)) if norm != "batchnorm" else layer(torch.randn(6, 12))
        assert out.shape[-1] == 4


def test_regularization_loss_components():
    layer = KANLinear(6, 4)
    l1, ent = layer.regularization_components()
    assert l1 > 0
    assert 0.0 <= float(ent) <= 1.0 + 1e-6
    total = layer.regularization_loss(regularize_activation=2.0, regularize_entropy=0.5)
    assert torch.allclose(total, 2.0 * l1 + 0.5 * ent)
    total.backward()
    assert layer.spline_weight.grad is not None


def test_entropy_is_minimal_for_single_active_edge():
    layer = KANLinear(4, 3, input_norm="none")
    with torch.no_grad():
        layer.spline_weight.zero_()
        layer.spline_weight[0, 0] = 1.0
    _, ent = layer.regularization_components()
    assert float(ent) < 1e-3


def test_edge_function_matches_layer_output():
    layer = KANLinear(3, 2, input_norm="none")
    x = torch.tensor([[0.2, -0.4, 0.7]])
    out = layer(x)
    manual = torch.zeros(2)
    for j in range(2):
        for i in range(3):
            phi, _, _ = layer.edge_function(x[0, i : i + 1], j, i)
            manual[j] += phi[0]
    assert torch.allclose(out[0], manual, atol=1e-5)


def test_update_grid_preserves_function():
    layer = KANLinear(5, 3, input_norm="none")
    x = torch.randn(512, 5) * 0.6
    before = layer(x)
    layer.update_grid(x)
    after = layer(x)
    assert torch.isfinite(after).all()
    rel = (before - after).norm() / before.norm()
    assert rel < 0.1
    # Grid must stay strictly increasing.
    assert torch.all(layer.grid[:, 1:] > layer.grid[:, :-1])


@pytest.mark.parametrize("groups,stride,padding", [(1, 1, 1), (4, 1, 1), (8, 2, 1), (1, 1, 0)])
def test_kanconv2d_shapes(groups, stride, padding):
    conv = KANConv2d(8, 16 if groups != 8 else 8, kernel_size=3, stride=stride, padding=padding, groups=groups)
    x = torch.randn(2, 8, 9, 9)
    y = conv(x)
    h = (9 + 2 * padding - 3) // stride + 1
    assert y.shape == (2, conv.out_channels, h, h)


def test_kanconv2d_backward_and_regularization():
    conv = KANConv2d(4, 4, kernel_size=3, padding=1, groups=4)
    x = torch.randn(3, 4, 6, 6, requires_grad=True)
    loss = conv(x).pow(2).mean() + conv.regularization_loss()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert conv.spline_weight.grad.abs().sum() > 0


def test_kanconv2d_rejects_bad_groups():
    with pytest.raises(ValueError):
        KANConv2d(6, 4, groups=4)


def test_kan_conv_bottleneck_residual_shapes():
    block = KANConvBottleneck(32, 32, reduction=4)
    assert block.mid_channels == 8
    y = block(torch.randn(2, 32, 5, 5))
    assert y.shape == (2, 32, 5, 5)
    block2 = KANConvBottleneck(32, 48, reduction=4)
    assert block2(torch.randn(1, 32, 4, 4)).shape == (1, 48, 4, 4)

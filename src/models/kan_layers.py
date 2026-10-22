"""Pure-PyTorch, fully vectorised Kolmogorov-Arnold Network (KAN) layers.

Every edge (i -> j) of a KAN layer carries a learnable univariate function

    phi_{j,i}(x) = w_b[j,i] * SiLU(x) + w_s[j,i] * sum_m c_{j,i,m} B_{m,k}(x),

where ``B_{m,k}`` are cubic (k = 3) B-spline basis functions defined on a
uniform knot vector of ``G`` intervals over ``grid_range`` that is extended by
``k`` knots on both sides, giving ``G + k`` basis functions per edge.

The basis functions are evaluated with an exact, vectorised Cox-de Boor
recursion. No custom CUDA/C++ kernels are used, so the layers run unchanged on
CPU, Apple-Silicon (MPS) and CUDA devices with full autograd support.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "b_spline_basis",
    "make_uniform_grid",
    "KANLinear",
    "KANConv2d",
    "KANConvBottleneck",
    "iter_kan_layers",
]


def make_uniform_grid(
    grid_size: int, spline_order: int, grid_range: Tuple[float, float] = (-1.0, 1.0)
) -> torch.Tensor:
    """Uniform knot vector of ``grid_size + 2 * spline_order + 1`` knots.

    The interior interval ``grid_range`` is split into ``grid_size`` pieces and
    extended by ``spline_order`` knots on both sides so that every point of the
    interior is covered by exactly ``spline_order + 1`` non-zero bases.
    """
    h = (grid_range[1] - grid_range[0]) / grid_size
    steps = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
    return steps * h + grid_range[0]


def b_spline_basis(x: torch.Tensor, grid: torch.Tensor, spline_order: int) -> torch.Tensor:
    """Evaluate all B-spline bases of order ``spline_order`` at ``x``.

    Implements the Cox-de Boor recursion

        B_{m,0}(x) = 1[t_m <= x < t_{m+1}]
        B_{m,p}(x) = (x - t_m) / (t_{m+p} - t_m) * B_{m,p-1}(x)
                   + (t_{m+p+1} - x) / (t_{m+p+1} - t_{m+1}) * B_{m+1,p-1}(x)

    Args:
        x: tensor of arbitrary shape ``(...)``.
        grid: knot tensor whose last dimension holds the knots and whose leading
            dimensions broadcast against ``x`` (e.g. ``(in_features, n_knots)``
            for ``x`` of shape ``(batch, in_features)``, or ``(n_knots,)``).
        spline_order: polynomial order ``k`` of the spline.

    Returns:
        Tensor of shape ``(..., n_knots - 1 - spline_order)``.
    """
    x = x.unsqueeze(-1)
    grid = grid.to(dtype=x.dtype)
    bases = ((x >= grid[..., :-1]) & (x < grid[..., 1:])).to(x.dtype)
    for p in range(1, spline_order + 1):
        left_num = x - grid[..., : -(p + 1)]
        left_den = grid[..., p:-1] - grid[..., : -(p + 1)]
        right_num = grid[..., p + 1 :] - x
        right_den = grid[..., p + 1 :] - grid[..., 1:(-p)]
        bases = (left_num / left_den) * bases[..., :-1] + (right_num / right_den) * bases[..., 1:]
    return bases.contiguous()


def _normalised_entropy(edge_l1: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy of the edge-importance distribution, normalised to [0, 1]."""
    flat = edge_l1.flatten()
    p = flat / (flat.sum() + eps)
    entropy = -(p * torch.log(p + eps)).sum()
    return entropy / math.log(max(flat.numel(), 2))


class KANLinear(nn.Module):
    """Fully connected KAN layer with learnable B-spline edge activations.

    Args:
        in_features: number of input features.
        out_features: number of output features.
        grid_size: number of grid intervals ``G``.
        spline_order: spline order ``k`` (3 = cubic).
        scale_noise: std of the noise used to initialise spline coefficients.
        scale_base: gain of the Kaiming-uniform initialisation of ``base_weight``.
        scale_spline: gain of the Kaiming-uniform initialisation of ``spline_scaler``.
        enable_standalone_scale_spline: learn a per-edge scale ``w_s``.
        base_activation: residual activation ``b(x)`` (SiLU by default).
        grid_eps: mixing factor between adaptive and uniform grids in
            :meth:`update_grid` (1 = uniform, 0 = fully adaptive).
        grid_range: interior grid domain.
        input_norm: ``"layernorm"``, ``"batchnorm"`` or ``"none"`` normalisation
            applied to ``x`` before the spline so that inputs stay inside the grid.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        input_norm: str = "layernorm",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.grid_eps = grid_eps
        self.grid_range = tuple(grid_range)

        grid = make_uniform_grid(grid_size, spline_order, grid_range)
        self.register_buffer("grid", grid.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.register_parameter("spline_scaler", None)

        if input_norm == "layernorm":
            self.input_norm = nn.LayerNorm(in_features)
        elif input_norm == "batchnorm":
            self.input_norm = nn.BatchNorm1d(in_features)
        elif input_norm == "none":
            self.input_norm = nn.Identity()
        else:
            raise ValueError(f"Unknown input_norm '{input_norm}'")

        self.base_activation = base_activation()
        self.reset_parameters()

    @property
    def num_coefficients(self) -> int:
        return self.grid_size + self.spline_order

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # Initialise the spline as a least-squares fit of small random curves
            # sampled at the interior knots (as in the original KAN).
            n_pts = self.grid_size + 1
            noise = (
                (torch.rand(n_pts, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            x_pts = self.grid.T[self.spline_order : -self.spline_order]
            coeff = self.curve2coeff(x_pts, noise)
            scale = 1.0 if self.enable_standalone_scale_spline else self.scale_spline
            self.spline_weight.copy_(scale * coeff)
        if self.spline_scaler is not None:
            nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """B-spline bases of ``x`` (``(N, in)``) -> ``(N, in, G + k)``."""
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected input of shape (N, {self.in_features}), got {tuple(x.shape)}"
            )
        return b_spline_basis(x, self.grid, self.spline_order)

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Least-squares spline coefficients interpolating ``y`` at points ``x``.

        Args:
            x: ``(N, in_features)`` sample locations.
            y: ``(N, in_features, out_features)`` target values.

        Returns:
            ``(out_features, in_features, G + k)`` coefficients.
        """
        device = x.device
        # torch.linalg.lstsq is not implemented on MPS; solve on CPU there.
        solve_device = torch.device("cpu") if device.type == "mps" else device
        A = self.b_splines(x).transpose(0, 1).to(solve_device, torch.float32)
        B = y.transpose(0, 1).to(solve_device, torch.float32)
        driver = "gelsd" if solve_device.type == "cpu" else None
        solution = torch.linalg.lstsq(A, B, driver=driver).solution
        return solution.permute(2, 0, 1).contiguous().to(device)

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        """Effective spline coefficients ``w_s * c`` of shape ``(out, in, G + k)``."""
        if self.spline_scaler is None:
            return self.spline_weight
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        x = self.input_norm(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        bases = self.b_splines(x)
        spline_output = F.linear(
            bases.reshape(x.size(0), -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Adapt the knot vector to the empirical input distribution.

        The current edge functions are re-sampled at the new inputs and refitted
        on the new grid by least squares, so the function computed by the layer
        is (approximately) preserved while its resolution follows the data.
        """
        x = x.reshape(-1, self.in_features)
        x = self.input_norm(x)
        batch = x.size(0)
        if batch < self.grid_size + 1:
            raise ValueError("update_grid needs at least grid_size + 1 samples")

        splines = self.b_splines(x).permute(1, 0, 2)  # (in, N, coeff)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)  # (N, in, out)

        x_sorted = torch.sort(x, dim=0)[0]
        idx = torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        grid_adaptive = x_sorted[idx]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=x.dtype, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        k = self.spline_order
        steps_left = torch.arange(k, 0, -1, device=x.device, dtype=x.dtype).unsqueeze(1)
        steps_right = torch.arange(1, k + 1, device=x.device, dtype=x.dtype).unsqueeze(1)
        grid = torch.cat(
            [grid[:1] - uniform_step * steps_left, grid, grid[-1:] + uniform_step * steps_right],
            dim=0,
        )
        # Guard against degenerate (zero-width) intervals for constant features.
        grid = torch.cummax(grid + 1e-6 * torch.arange(grid.size(0), device=x.device).unsqueeze(1), 0)[0]
        self.grid.copy_(grid.T.contiguous())

        new_coeff = self.curve2coeff(x, unreduced_output)
        if self.spline_scaler is not None:
            scaler = self.spline_scaler.unsqueeze(-1)
            safe = torch.where(scaler.abs() < 1e-8, torch.full_like(scaler, 1e-8), scaler)
            new_coeff = new_coeff / safe
        self.spline_weight.copy_(new_coeff)

    def edge_l1(self) -> torch.Tensor:
        """Per-edge importance ``mean_m |w_s c_m|`` of shape ``(out, in)``."""
        return self.scaled_spline_weight.abs().mean(-1)

    def regularization_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(L1, H)``: mean absolute spline weight and normalised entropy."""
        edge_l1 = self.edge_l1()
        return edge_l1.mean(), _normalised_entropy(edge_l1)

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        l1, entropy = self.regularization_components()
        return regularize_activation * l1 + regularize_entropy * entropy

    @torch.no_grad()
    def edge_function(
        self, x: torch.Tensor, out_index: int, in_index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate a single learned edge ``phi_{out,in}`` on 1-D points ``x``.

        ``x`` is interpreted in the (normalised) domain seen by the spline.

        Returns:
            ``(phi, base_part, spline_part)`` each with the shape of ``x``.
        """
        x = x.to(self.grid.device, self.base_weight.dtype)
        grid = self.grid[in_index]
        bases = b_spline_basis(x, grid, self.spline_order)
        coeff = self.scaled_spline_weight[out_index, in_index]
        spline_part = bases @ coeff
        base_part = self.base_weight[out_index, in_index] * self.base_activation(x)
        return base_part + spline_part, base_part, spline_part

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}"
        )


class KANConv2d(nn.Module):
    """2-D convolution whose kernel taps are learnable B-spline edge functions.

    The input is unfolded into patches ``(B, C_in * K_h * K_w, L)``; every patch
    element ``x_p`` belonging to group ``g`` contributes
    ``phi_{o,p}(x_p) = w_b SiLU(x_p) + sum_m c_{o,p,m} B_m(x_p)`` to output
    channel ``o`` of that group. Outputs are folded back to
    ``(B, C_out, H_out, W_out)`` and batch-normalised.

    A ``BatchNorm2d`` precedes the unfolding so that activations lie inside the
    spline grid domain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int | Tuple[int, int] = 1,
        padding: int | Tuple[int, int] = 0,
        dilation: int | Tuple[int, int] = 1,
        groups: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        base_activation: type = nn.SiLU,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = nn.modules.utils._pair(padding)
        self.dilation = nn.modules.utils._pair(dilation)
        self.groups = groups
        self.grid_size = grid_size
        self.spline_order = spline_order

        kh, kw = self.kernel_size
        self.in_per_group = in_channels // groups
        self.out_per_group = out_channels // groups
        self.patch_dim = self.in_per_group * kh * kw
        n_coeff = grid_size + spline_order

        self.register_buffer("grid", make_uniform_grid(grid_size, spline_order, grid_range))
        self.input_norm = nn.BatchNorm2d(in_channels)
        self.base_activation = base_activation()
        self.base_weight = nn.Parameter(torch.empty(groups, self.out_per_group, self.patch_dim))
        self.spline_weight = nn.Parameter(
            torch.empty(groups, self.out_per_group, self.patch_dim, n_coeff)
        )
        self.spline_scaler = nn.Parameter(torch.empty(groups, self.out_per_group, self.patch_dim))
        self.output_norm = nn.BatchNorm2d(out_channels) if output_norm else nn.Identity()
        self.scale_noise = scale_noise
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.patch_dim
        bound = 1.0 / math.sqrt(fan_in)
        nn.init.uniform_(self.base_weight, -bound, bound)
        nn.init.uniform_(self.spline_scaler, -bound, bound)
        with torch.no_grad():
            self.spline_weight.normal_(0.0, self.scale_noise / self.grid_size)

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def output_size(self, h: int, w: int) -> Tuple[int, int]:
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        dh, dw = self.dilation
        h_out = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        w_out = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        return h_out, w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {c}")
        h_out, w_out = self.output_size(h, w)
        x = self.input_norm(x)
        patches = F.unfold(
            x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride
        )  # (B, C_in*kh*kw, L)
        n_loc = patches.size(-1)
        patches = patches.view(b, self.groups, self.patch_dim, n_loc)
        base_out = torch.einsum("bgpl,gop->bgol", self.base_activation(patches), self.base_weight)
        bases = b_spline_basis(patches, self.grid, self.spline_order)  # (B, g, P, L, C)
        spline_out = torch.einsum("bgplc,gopc->bgol", bases, self.scaled_spline_weight)
        out = (base_out + spline_out).reshape(b, self.out_channels, h_out, w_out)
        return self.output_norm(out)

    def edge_l1(self) -> torch.Tensor:
        return self.scaled_spline_weight.abs().mean(-1)

    def regularization_components(self) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_l1 = self.edge_l1()
        return edge_l1.mean(), _normalised_entropy(edge_l1)

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        l1, entropy = self.regularization_components()
        return regularize_activation * l1 + regularize_entropy * entropy

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, groups={self.groups}, "
            f"grid_size={self.grid_size}, spline_order={self.spline_order}"
        )


class KANConvBottleneck(nn.Module):
    """Residual bottleneck: 1x1 reduce -> depthwise 3x3 KANConv2d -> 1x1 expand."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 4,
        kernel_size: int = 3,
        grid_size: int = 5,
        spline_order: int = 3,
    ) -> None:
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)
        self.mid_channels = mid_channels
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )
        self.kan_conv = KANConv2d(
            mid_channels,
            mid_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=mid_channels,
            grid_size=grid_size,
            spline_order=spline_order,
        )
        self.expand = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(self.kan_conv(self.reduce(x)))
        return self.act(out + self.shortcut(x))


def iter_kan_layers(module: nn.Module) -> Iterable[nn.Module]:
    """Yield every :class:`KANLinear` / :class:`KANConv2d` inside ``module``."""
    for m in module.modules():
        if isinstance(m, (KANLinear, KANConv2d)):
            yield m

"""Dual-domain attention: CBAM-style channel attention + coordinate spatial attention.

Channel branch (CBAM, Woo et al. 2018):
    M_c = sigma(MLP(GAP(X)) + MLP(GMP(X)))

Coordinate branch (Coordinate Attention, Hou et al. 2021): features are pooled
independently along the horizontal and vertical directions, which keeps the
exact row/column position of a lesion,

    z^h_c(h) = 1/W sum_w X_c(h, w),     z^w_c(w) = 1/H sum_h X_c(h, w)
    f = delta(BN(W_1 [z^h ; z^w]))
    M_h = sigma(W_h f^h),               M_w = sigma(W_w f^w)

and the output is ``Y = X * M_c * M_h * M_w``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["ChannelAttention", "CoordinateSpatialAttention", "CoordinateChannelAttention"]


class ChannelAttention(nn.Module):
    """Channel attention using both global average and global max pooling."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        return self.gate(self.shared_mlp(avg) + self.shared_mlp(mx))


class CoordinateSpatialAttention(nn.Module):
    """Direction-aware spatial attention producing ``M_h`` (B,C,H,1) and ``M_w`` (B,C,1,W)."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish(inplace=True)
        self.conv_h = nn.Conv2d(mip, channels, 1)
        self.conv_w = nn.Conv2d(mip, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        _, _, h, w = x.shape
        x_h = torch.mean(x, dim=3, keepdim=True)  # (B, C, H, 1)
        x_w = torch.mean(x, dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        m_h = self.gate(self.conv_h(y_h))  # (B, C, H, 1)
        m_w = self.gate(self.conv_w(y_w.permute(0, 1, 3, 2)))  # (B, C, 1, W)
        return m_h, m_w


class CoordinateChannelAttention(nn.Module):
    """Sequential channel -> coordinate-spatial attention block.

    After every forward pass, the channel-averaged spatial mask
    ``mean_c(M_h * M_w)`` of shape ``(B, H, W)`` is cached in
    :attr:`last_attention_map` for visual inspection.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = CoordinateSpatialAttention(channels, reduction)
        self.last_attention_map: Optional[torch.Tensor] = None
        self.last_channel_weights: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m_c = self.channel_attention(x)
        x = x * m_c
        m_h, m_w = self.spatial_attention(x)
        spatial = m_h * m_w  # (B, C, H, W)
        self.last_attention_map = spatial.mean(dim=1).detach()
        self.last_channel_weights = m_c.flatten(1).detach()
        return x * spatial

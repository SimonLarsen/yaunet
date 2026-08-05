import torch
from torch import Tensor, nn

from ..conditioning import ConditionScaleShiftGate
from ..norms import LayerNorm2d
from ..types import NormConstructor


class SimpleGate(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SCA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        a = self.proj(self.pool(x))
        return a * x


class NAFNetBlock(nn.Module):
    """
    NAFNet block.

    See: https://arxiv.org/abs/2204.04676
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        dw_expand_factor: float = 2.0,
        ffn_expand_factor: float = 2.0,
        norm_layer: NormConstructor = LayerNorm2d,
    ):
        """
        Constructor.

        Parameters
        ----------
        channels
            Base channel width.
        condition_dim
            Optional conditioning width.
        dw_expand_factor
            Depth-wise convolution expansion.
        ffn_expand_factor
            FFN expansion.
        norm_layer
            Normalization layer constructor.
        """
        super().__init__()

        dw_channels = round(channels * dw_expand_factor)
        ffn_channels = round(channels * ffn_expand_factor)

        self.norm1 = norm_layer(channels)
        self.expand1 = nn.Conv2d(channels, dw_channels, 1)
        self.dwconv = nn.Conv2d(dw_channels, dw_channels, 3, 1, 1, groups=dw_channels)
        self.sg = SimpleGate()
        self.sca = SCA(dw_channels // 2)
        self.contract1 = nn.Conv2d(dw_channels // 2, channels, 1)

        self.norm2 = norm_layer(channels)
        self.expand2 = nn.Conv2d(channels, ffn_channels, 1)
        self.contract2 = nn.Conv2d(ffn_channels // 2, channels, 1)

        if condition_dim is not None:
            self.cond_proj1 = ConditionScaleShiftGate(condition_dim, channels)
            self.cond_proj2 = ConditionScaleShiftGate(condition_dim, channels)
        else:
            self.gate1 = nn.Parameter(torch.zeros(channels, 1, 1))
            self.gate2 = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        if c is not None:
            scale1, shift1, gate1 = self.cond_proj1(c)
            scale2, shift2, gate2 = self.cond_proj2(c)
        else:
            scale1, shift1, gate1 = 0.0, 0.0, self.gate1
            scale2, shift2, gate2 = 0.0, 0.0, self.gate2

        h = self.norm1(x) * (1 + scale1) + shift1
        h = self.dwconv(self.expand1(h))
        h = self.sg(h)
        h = self.sca(h)
        h = self.contract1(h)
        x = x + h * gate1

        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.expand2(h)
        h = self.sg(h)
        h = self.contract2(h)
        x = x + h * gate2

        return x

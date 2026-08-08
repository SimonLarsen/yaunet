from torch import Tensor, nn

from ..conditioning import ConditionScaleShiftGate
from ..norms import LayerNorm2d
from ..types import ActConstructor, NormConstructor


class CCA(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        a = self.proj(self.pool(x)).sigmoid()
        return x * a


class DiCoBlock(nn.Module):
    """
    DiCo block.

    See https://arxiv.org/abs/2505.11196
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        ffn_expand_factor: float = 4.0,
        norm_layer: NormConstructor = LayerNorm2d,
        act_layer: ActConstructor = nn.SiLU,
    ):
        """
        Constructor.

        Parameters
        ----------
        channels
            Base channel width.
        condition_dim
            Optional conditioning width.
        ffn_expand_factor
            FFN expansion.
        norm_layer
            Normalization layer constructor.
        act_layer
            Activation layer constructor.
        """
        super().__init__()

        ffn_channels = round(ffn_expand_factor * channels)

        self.act = act_layer()

        self.norm1 = norm_layer(channels)
        self.conv1 = nn.Conv2d(channels, channels, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.cca = CCA(channels)
        self.conv3 = nn.Conv2d(channels, channels, 1)

        self.norm2 = norm_layer(channels)
        self.expand = nn.Conv2d(channels, ffn_channels, 1)
        self.contract = nn.Conv2d(ffn_channels, channels, 1)

        if condition_dim is not None:
            self.cond_proj1 = ConditionScaleShiftGate(condition_dim, channels)
            self.cond_proj2 = ConditionScaleShiftGate(condition_dim, channels)

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        if c is not None:
            scale1, shift1, gate1 = self.cond_proj1(c)
            scale2, shift2, gate2 = self.cond_proj2(c)
        else:
            scale1, shift1, gate1 = 0.0, 0.0, 1.0
            scale2, shift2, gate2 = 0.0, 0.0, 1.0

        h = self.norm1(x) * (1 + scale1) + shift1
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.act(h)
        h = self.cca(h)
        h = self.conv3(h)
        x = x + h * gate1

        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.expand(h)
        h = self.act(h)
        h = self.contract(h)
        x = x + h * gate2

        return x

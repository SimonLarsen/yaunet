from torch import Tensor, nn

from ..conditioning import ConditionScaleShiftGate
from ..norms import LayerNorm2d
from ..types import NormConstructor
from .nafnet import SCA, SimpleGate


class NAFNextBlock(nn.Module):
    """
    ConvNeXt block where activation and GRN/layer scale is replaced with
    SimpleGate and Simplified Channel Attention from NAFNet.

    * ConvNeXt: [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545).
    * ConvNeXtV2: [https://arxiv.org/abs/2301.00808](https://arxiv.org/abs/2301.00808).
    * NAFNet: [https://arxiv.org/abs/2204.04676](https://arxiv.org/abs/2204.04676).
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        kernel_size: int = 7,
        expand_factor: float = 4.0,
        norm_layer: NormConstructor = LayerNorm2d,
    ):
        """Constructor."""
        super().__init__()

        hidden_dim = round(expand_factor * channels)

        self.dwconv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            groups=channels,
            bias=False,
        )
        self.norm = norm_layer(channels)
        self.expand = nn.Conv2d(channels, hidden_dim, 1)
        self.sg = SimpleGate()
        self.sca = SCA(hidden_dim // 2)
        self.contract = nn.Conv2d(hidden_dim // 2, channels, 1)

        if condition_dim is not None:
            self.cond_proj = ConditionScaleShiftGate(condition_dim, channels)

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        if c is not None:
            scale, shift, gate = self.cond_proj(c)
        else:
            scale, shift, gate = 0.0, 0.0, 1.0

        h = self.dwconv(x)
        h = self.norm(h) * (1 + scale) + shift
        h = self.expand(h)
        h = self.sg(h)
        h = self.sca(h)
        h = self.contract(h)
        return x + h * gate

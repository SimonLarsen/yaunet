from torch import Tensor, nn

from ..conditioning import ConditionScaleShiftGate
from ..norms import LayerNorm2d
from ..types import ActConstructor, NormConstructor


class ResNetBlock(nn.Module):
    """
    ResNet basic block with full pre-activation.

    See: https://arxiv.org/abs/1512.03385
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        kernel_size: int = 3,
        norm_layer: NormConstructor = LayerNorm2d,
        act_layer: ActConstructor = nn.SiLU,
    ):
        """
        Constructor

        Parameters
        ----------
        channels
            Base channel width.
        condition_dim
            Optional conditioning width.
        kernel_size
            Convolution kernel size.
        norm_layer
            Normalization layer constructor.
        act_layer
            Activation layer constructor.
        """
        super().__init__()

        self.act = act_layer()

        self.norm1 = norm_layer(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, 1, "same", bias=False)

        self.norm2 = norm_layer(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, 1, "same")

        nn.init.normal_(self.conv2.weight, 0.0, 1e-3)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        if condition_dim is not None:
            self.cond_proj = ConditionScaleShiftGate(condition_dim, channels)

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        if c is not None:
            scale, shift, gate = self.cond_proj(c)
        else:
            scale, shift, gate = 0.0, 0.0, 1.0

        h = self.conv1(self.act(self.norm1(x)))
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.act(h))
        return x + h * gate

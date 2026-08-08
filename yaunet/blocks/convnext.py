import torch
from torch import Tensor, nn

from ..conditioning import ConditionScaleShiftGate
from ..norms import LayerNorm2d
from ..types import ActConstructor, NormConstructor


class ConvNextBlock(nn.Module):
    """
    ConvNeXt (v1) block.

    See: https://arxiv.org/abs/2201.03545
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        kernel_size: int = 7,
        expand_factor: float = 4.0,
        layer_scale_init: float = 1e-6,
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
        kernel_size
            Depth-wise convolution kernel size.
        expand_factor
            Inverted bottleneck expansion.
        layer_scale_init
            Layer scale initial value.
        norm_layer
            Normalization layer constructor.
        act_layer
            Activation layer constructor.
        """
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
        self.act = act_layer()
        self.contract = nn.Conv2d(hidden_dim, channels, 1)

        if condition_dim is not None:
            self.cond_proj = ConditionScaleShiftGate(condition_dim, channels)
        else:
            self.scale = nn.Parameter(torch.full((channels, 1, 1), layer_scale_init))

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        if c is not None:
            scale, shift, gate = self.cond_proj(c)
        else:
            scale, shift, gate = 0.0, 0.0, self.scale

        h = self.dwconv(x)
        h = self.norm(h) * (1 + scale) + shift
        h = self.expand(h)
        h = self.act(h)
        h = self.contract(h)
        return x + h * gate


class GRN(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta * x


class ConvNextV2Block(nn.Module):
    """
    ConvNeXt V2 block.

    See: https://arxiv.org/abs/2301.00808
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        kernel_size: int = 7,
        expand_factor: float = 4.0,
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
        kernel_size
            Depth-wise convolution kernel size.
        expand_factor
            Inverted bottleneck expansion.
        norm_layer
            Normalization layer constructor.
        act_layer
            Activation layer constructor.
        """
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
        self.act = act_layer()
        self.grn = GRN(hidden_dim)
        self.contract = nn.Conv2d(hidden_dim, channels, 1)

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
        h = self.act(h)
        h = self.grn(h)
        h = self.contract(h)
        return x + h * gate

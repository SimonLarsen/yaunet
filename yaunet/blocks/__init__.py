from torch import Tensor, nn

from .convnext import ConvNextBlock, ConvNextV2Block
from .nafnet import NAFNetBlock
from .resnet import ResNetBlock
from .restormer import RestormerBlock
from .dico import DiCoBlock


class IdentityBlock(nn.Module):
    """Identity block. Does nothing."""

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
    ):
        super().__init__()

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        return x


__all__ = [
    "ConvNextBlock",
    "ConvNextV2Block",
    "IdentityBlock",
    "NAFNetBlock",
    "ResNetBlock",
    "RestormerBlock",
    "DiCoBlock",
]

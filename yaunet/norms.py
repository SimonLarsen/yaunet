from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """Layer norm for channel dimension of (N, C, H, W) tensor."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class RMSNorm2d(nn.Module):
    """RMS norm for channel dimension of (N, C, H, W) tensor."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.RMSNorm(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

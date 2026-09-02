from torch import Tensor, nn

from .types import ActConstructor


class MLP(nn.Module):
    """Basic MLP."""

    def __init__(
        self,
        channels: int,
        expand_factor: float = 4.0,
        act_layer: ActConstructor = nn.SiLU,
    ):
        super().__init__()

        hidden_dim = round(channels * expand_factor)
        self.expand = nn.Conv2d(channels, hidden_dim, 1)
        self.act = act_layer()
        self.contract = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.contract(self.act(self.expand(x)))


class GLU(nn.Module):
    """Gated linear unit MLP."""

    def __init__(
        self,
        channels: int,
        expand_factor: float = 4.0,
        act_layer: ActConstructor = nn.SiLU,
    ):
        super().__init__()

        hidden_dim = round(channels * expand_factor)
        self.expand = nn.Conv2d(channels, 2 * hidden_dim, 1)
        self.act = act_layer()
        self.contract = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h1, h2 = self.expand(x).chunk(2, dim=1)
        return self.contract(self.act(h1) * h2)


__all__ = [
    "GLU",
    "MLP",
]

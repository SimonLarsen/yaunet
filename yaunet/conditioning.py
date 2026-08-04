from einops import rearrange
from torch import Tensor, nn


class ConditionScaleShift(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        channels: int,
    ):
        super().__init__()

        self.proj = nn.Linear(condition_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        scale, shift = rearrange(
            self.proj(x),
            "b (ch c) -> ch b c 1 1",
            ch=2,
        )
        return scale, shift


class ConditionScale2Shift(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        channels: int,
    ):
        super().__init__()

        self.proj = nn.Linear(condition_dim, 3 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        scale1, shift, scale2 = rearrange(
            self.proj(x),
            "b (ch c) -> ch b c 1 1",
            ch=3,
        )
        return scale1, shift, scale2

import torch
from einops import rearrange
from torch import Tensor, nn

from .conditioning import ConditionScale2Shift, ConditionScaleShift
from .norms import LayerNorm2d
from .types import ActConstructor, NormConstructor


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


class ResBlock(nn.Module):
    """
    ResNet basic block with full pre-activation.

    See: https://arxiv.org/abs/1512.03385.
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        kernel_size: int = 3,
        norm_layer: NormConstructor = LayerNorm2d,
        act_layer: ActConstructor = nn.SiLU,
    ):
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
            self.cond_proj = ConditionScaleShift(condition_dim, channels)

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        h = self.conv1(self.act(self.norm1(x)))

        h = self.norm2(h)

        if c is not None:
            scale, shift = self.cond_proj(c)
            h = h * (1 + scale) + shift

        h = self.conv2(self.act(h))
        return x + h


class ConvNextBlock(nn.Module):
    """
    ConvNeXt (v1) block.

    See: https://arxiv.org/abs/2201.03545.
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
        self.scale = nn.Parameter(torch.full((channels, 1, 1), layer_scale_init))

        if condition_dim is not None:
            self.cond_proj = ConditionScaleShift(condition_dim, channels)

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        h = self.dwconv(x)
        h = self.norm(h)
        if c is not None:
            scale, shift = self.cond_proj(c)
            h = h * (1 + scale) + shift
        h = self.expand(h)
        h = self.act(h)
        h = self.contract(h)
        return x + h * self.scale


class RestormerAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.temperature = nn.Parameter(torch.ones(dim // head_dim, 1, 1))
        self.to_qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.dwconv = nn.Conv2d(3 * dim, 3 * dim, 3, 1, 1, groups=3 * dim)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = rearrange(
            self.dwconv(self.to_qkv(x)),
            "b (qkv nh hd) h w -> qkv b nh hd (h w)",
            qkv=3,
            hd=self.head_dim,
        )
        q = nn.functional.normalize(q, dim=-1)
        k = nn.functional.normalize(k, dim=-1)

        attn = (q @ k.mT) * self.temperature
        attn = attn.softmax(dim=-1)

        out = rearrange(
            attn @ v,
            "b nh hd (h w) -> b (nh hd) h w",
            w=x.size(-1),
        )
        return self.proj_out(out)


class RestormerFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        expand_factor: float = 4.0,
        act_layer: ActConstructor = nn.SiLU,
    ):
        super().__init__()

        hidden_dim = round(dim * expand_factor)
        self.expand = nn.Conv2d(dim, 2 * hidden_dim, 1)
        self.dwconv = nn.Conv2d(
            2 * hidden_dim, 2 * hidden_dim, 3, 1, 1, groups=2 * hidden_dim
        )
        self.act = act_layer()
        self.contract = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        h, gate = self.dwconv(self.expand(x)).chunk(2, dim=1)
        return self.contract(h * self.act(gate))


class RestormerBlock(nn.Module):
    """
    Restormer transformer block.

    See: https://arxiv.org/abs/2111.09881
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        head_dim: int = 32,
        ffn_expand_factor: float = 4.0,
        norm_layer: NormConstructor = LayerNorm2d,
        act_layer: ActConstructor = nn.SiLU,
    ):
        super().__init__()

        self.norm1 = norm_layer(channels)
        self.norm2 = norm_layer(channels)

        self.attn = RestormerAttention(channels, head_dim)
        self.ffn = RestormerFFN(channels, ffn_expand_factor, act_layer)

        if condition_dim is not None:
            self.cond_proj1 = ConditionScale2Shift(condition_dim, channels)
            self.cond_proj2 = ConditionScale2Shift(condition_dim, channels)

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        if c is not None:
            scale1_1, shift1, scale1_2 = self.cond_proj1(c)
            scale2_1, shift2, scale2_2 = self.cond_proj2(c)
        else:
            scale1_1, shift1, scale1_2 = 0.0, 0.0, 0.0
            scale2_1, shift2, scale2_2 = 0.0, 0.0, 0.0

        h = self.norm1(x) * (1 + scale1_1) + shift1
        h = self.attn(h)
        x = x + h * scale1_2

        h = self.norm2(x) * (1 + scale2_1) + shift2
        h = self.ffn(h)
        x = x + h * scale2_2

        return x

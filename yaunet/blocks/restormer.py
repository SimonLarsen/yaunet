import torch
from einops import rearrange
from torch import Tensor, nn

from ..conditioning import ConditionScaleShiftGate
from ..norms import LayerNorm2d
from ..types import ActConstructor, NormConstructor


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

    See: [https://arxiv.org/abs/2111.09881](https://arxiv.org/abs/2111.09881).
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
        """
        Constructor.

        Parameters
        ----------
        channels
            Base channel width.
        condition_dim
            Optional conditioning width.
        head_dim
            Attention head dimension.
        ffn_expand_factor
            FFN expansion.
        norm_layer
            Normalization layer constructor.
        act_layer
            Activation layer constructor.
        """
        super().__init__()

        self.norm1 = norm_layer(channels)
        self.norm2 = norm_layer(channels)

        self.attn = RestormerAttention(channels, head_dim)
        self.ffn = RestormerFFN(channels, ffn_expand_factor, act_layer)

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
        h = self.attn(h)
        x = x + h * gate1

        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.ffn(h)
        x = x + h * gate2

        return x

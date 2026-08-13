from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from ..conditioning import ConditionScaleShiftGate
from ..mlp import MLP
from ..norms import LayerNorm2d
from ..types import ActConstructor, MLPConstructor, NormConstructor


class Attention(nn.Module):
    def __init__(
        self,
        channels: int,
        head_dim: int,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.to_qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        q, k, v = rearrange(
            self.to_qkv(x),
            "b (qkv nh hd) h w -> qkv b nh (h w) hd",
            qkv=3,
            hd=self.head_dim,
        )
        o = scaled_dot_product_attention(q, k, v)
        o = rearrange(o, "b nh (h w) hd -> b (nh hd) h w", w=x.size(-1))
        o = self.proj_out(o)
        return o


class ViTBlock(nn.Module):
    """
    ViT-like transformer block.

    See: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929).
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        head_dim: int = 32,
        norm_layer: NormConstructor = LayerNorm2d,
        act_layer: ActConstructor = nn.SiLU,
        ffn_layer: MLPConstructor = MLP,
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
        norm_layer
            Normalization layer constructor.
        act_layer
            Activation layer constructor.
        ffn_layer
            FFN layer constructor.
        """
        super().__init__()

        self.norm1 = norm_layer(channels)
        self.norm2 = norm_layer(channels)

        self.attn = Attention(channels, head_dim)
        self.mlp = ffn_layer(channels)

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
        h = self.mlp(h)
        x = x + h * gate2

        return x

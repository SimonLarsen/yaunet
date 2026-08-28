from einops import rearrange
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from ..conditioning import ConditionScaleShiftGate
from ..norms import LayerNorm2d
from ..types import ActConstructor, NormConstructor


class SegformerAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        head_dim: int,
        downsample_factor: int,
        norm_layer: NormConstructor = LayerNorm2d,
    ):
        super().__init__()

        self.head_dim = head_dim

        self.downsample = nn.Sequential(
            nn.Conv2d(channels, channels, downsample_factor, downsample_factor),
            norm_layer(channels),
        )

        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_kv = nn.Conv2d(channels, 2 * channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        q = rearrange(
            self.to_q(x),
            "b (nh hd) h w -> b nh (h w) hd",
            hd=self.head_dim,
        )

        k, v = rearrange(
            self.to_kv(self.downsample(x)),
            "b (kv nh hd) h w -> kv b nh (h w) hd",
            kv=2,
            hd=self.head_dim,
        )

        o = scaled_dot_product_attention(q, k, v)
        o = rearrange(o, "b nh (h w) hd -> b (nh hd) h w", w=x.size(-1))
        o = self.proj_out(o)
        return o


class MixFFN(nn.Module):
    def __init__(
        self,
        channels: int,
        expand_factor: float = 4.0,
        act_layer: ActConstructor = nn.SiLU,
    ):
        super().__init__()

        hidden_dim = round(channels * expand_factor)

        self.expand = nn.Conv2d(channels, hidden_dim, 1)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim)
        self.act = act_layer()
        self.contract = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h = self.expand(x)
        h = self.dwconv(h)
        h = self.act(h)
        h = self.contract(h)
        return h


class SegformerBlock(nn.Module):
    """
    Segformer block.

    See [https://arxiv.org/abs/2105.15203](https://arxiv.org/abs/2105.15203).
    """

    def __init__(
        self,
        channels: int,
        condition_dim: int | None = None,
        head_dim: int = 32,
        downsample_factor: int = 16,
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
        downsample_factor
            Downsampling to apply to keys/values before attention.
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

        self.attn = SegformerAttention(
            channels=channels,
            head_dim=head_dim,
            downsample_factor=downsample_factor,
            norm_layer=norm_layer,
        )
        self.ffn = MixFFN(channels, ffn_expand_factor, act_layer)

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

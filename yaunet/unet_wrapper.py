from collections.abc import Sequence

from torch import Tensor, nn

from .blocks import IdentityBlock, ResBlock
from .types import BlockConstructor, FuseMethod, InterpolationMethod, UpsampleMethod
from .unet import UNet


class UNetWrapper(nn.Module):
    """U-Net shell for wrapping other models."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_channels: int,
        down_widths: Sequence[int] = (32, 64, 128, 256),
        down_depths: Sequence[int] = (2, 2, 2, 8),
        up_widths: Sequence[int] = (256, 128, 64, 32),
        up_depths: Sequence[int] = (2, 2, 2, 2),
        fuse_method: FuseMethod = "concat",
        upsample_method: UpsampleMethod = "pixel-shuffle",
        interpolation: InterpolationMethod = "nearest-exact",
        down_block_layer: BlockConstructor = ResBlock,
        up_block_layer: BlockConstructor = ResBlock,
    ):
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            down_widths=down_widths,
            down_depths=down_depths,
            up_widths=up_widths,
            up_depths=up_depths,
            mid_width=feature_channels,
            mid_depth=0,
            fuse_method=fuse_method,
            upsample_method=upsample_method,
            interpolation=interpolation,
            down_block_layer=down_block_layer,
            mid_block_layer=IdentityBlock,
            up_block_layer=up_block_layer,
        )

        self.unet.mid = nn.Identity()

    def forward(self, x: Tensor, features: Tensor) -> Tensor:
        skips = self.unet.encode(x)
        skips[-1] = features
        output = self.unet.decode(skips)
        return self.unet.proj_out(output)

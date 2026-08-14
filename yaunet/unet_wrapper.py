from collections.abc import Sequence

from torch import Tensor, nn

from .blocks import IdentityBlock, ResNetBlock
from .types import BlockConstructor, FuseMethod, InterpolationMethod, UpsampleMethod
from .unet import UNet


class UNetWrapper(nn.Module):
    """
    U-Net shell for wrapping other models.

    This model is meant to be used for upsampling features from a separate (typically frozen) model.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_channels: int,
        condition_dim: int | None = None,
        down_widths: Sequence[int] = (32, 64, 128, 256),
        down_depths: Sequence[int] = (2, 2, 2, 8),
        up_widths: Sequence[int] = (256, 128, 64, 32),
        up_depths: Sequence[int] = (2, 2, 2, 2),
        fuse_method: FuseMethod = "concat",
        upsample_method: UpsampleMethod = "pixel-shuffle",
        interpolation: InterpolationMethod = "nearest-exact",
        down_block_layer: BlockConstructor = ResNetBlock,
        up_block_layer: BlockConstructor = ResNetBlock,
    ):
        """
        Constructor.

        Parameters
        ----------
        feature_channels
            Number of channels to be injected at the bottleneck.
        """
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            condition_dim=condition_dim,
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

        self.unet.mid = IdentityBlock(0)

    def forward(
        self,
        x: Tensor,
        features: Tensor,
        c: Tensor | None = None,
    ) -> Tensor:
        """
        Define the computation performed at every call.

        Parameters
        ----------
        x
            Full resolution reference image with shape `(B, C, H, W)`
            where `C` is `in_channels`.
        features
            Features to be injected in bottleneck.
            Should have shape `(B, E, h, w)` where `E` is `feature_channels`.
        c
            Optional condition with shape `(B, D)` where `D` is `condition_dim`.
        """
        skips = self.unet.encode(x, c)
        skips[-1] = features
        output = self.unet.decode(skips, c)
        return self.unet.proj_out(output)

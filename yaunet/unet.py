from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .blocks import ResBlock
from .types import BlockConstructor, FuseMethod, InterpolationMethod, UpsampleMethod


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int | None = None,
        depth: int = 1,
        downsample: bool = False,
        block_layer: BlockConstructor = ResBlock,
    ):
        super().__init__()

        if downsample:
            self.downsample = nn.Conv2d(in_channels, out_channels, 2, 2)
        elif in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(block_layer(out_channels, condition_dim))

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        h = self.downsample(x)
        for block in self.blocks:
            h = block(h, c)
        return h


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        condition_dim: int | None = None,
        depth: int = 1,
        fuse_method: FuseMethod = "concat",
        upsample_method: UpsampleMethod = "pixel-shuffle",
        interpolation: InterpolationMethod = "nearest-exact",
        block_layer: BlockConstructor = ResBlock,
    ):
        super().__init__()

        self.fuse_method = fuse_method

        fuse_in_channels = in_channels
        if upsample_method == "interpolate":
            self.upsample = nn.Upsample(
                scale_factor=2,
                mode=interpolation,
            )
        elif upsample_method == "pixel-shuffle":
            self.upsample = nn.Sequential(
                nn.Conv2d(in_channels, 4 * skip_channels, 1),
                nn.PixelShuffle(2),
            )
            fuse_in_channels = skip_channels
        else:
            raise ValueError(f"Unknown upsample method {upsample_method}")

        self.fuse_proj = nn.Identity()
        if fuse_method == "add":
            if fuse_in_channels != skip_channels:
                self.fuse_proj = nn.Conv2d(fuse_in_channels, skip_channels, 1)
        elif fuse_method == "concat":
            self.fuse = nn.Conv2d(fuse_in_channels + skip_channels, out_channels, 1)
        else:
            raise ValueError(f"Unknown fuse method {fuse_method}")

        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(block_layer(out_channels, condition_dim))

    def forward(self, x: Tensor, x_skip: Tensor, c: Tensor | None = None) -> Tensor:
        h = self.fuse_proj(self.upsample(x))

        if self.fuse_method == "add":
            h = (h + x_skip) / 2**0.5
        elif self.fuse_method == "concat":
            h = torch.cat((h, x_skip), dim=1)
            h = self.fuse(h)

        for block in self.blocks:
            h = block(h, c)
        return h


class UNet(nn.Module):
    """U-Net model."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int | None = None,
        down_widths: Sequence[int] = (32, 64, 128, 256),
        down_depths: Sequence[int] = (2, 2, 2, 8),
        mid_width: int = 512,
        mid_depth: int = 2,
        up_widths: Sequence[int] = (256, 128, 64, 32),
        up_depths: Sequence[int] = (2, 2, 2, 2),
        fuse_method: FuseMethod = "concat",
        upsample_method: UpsampleMethod = "pixel-shuffle",
        interpolation: InterpolationMethod = "nearest-exact",
        down_block_layer: BlockConstructor | Sequence[BlockConstructor] = ResBlock,
        mid_block_layer: BlockConstructor = ResBlock,
        up_block_layer: BlockConstructor | Sequence[BlockConstructor] = ResBlock,
    ):
        super().__init__()

        if not isinstance(down_block_layer, Sequence):
            down_block_layer = (down_block_layer,) * len(down_widths)
        if not isinstance(up_block_layer, Sequence):
            up_block_layer = (up_block_layer,) * len(up_widths)

        assert len(down_widths) == len(down_depths)
        assert len(down_block_layer) == len(down_widths)

        assert len(up_widths) == len(up_depths)
        assert len(up_block_layer) == len(up_widths)

        assert len(down_widths) == len(up_widths)

        self.down = nn.ModuleList()
        prev_channels = in_channels
        for i in range(len(down_widths)):
            self.down.append(
                DownBlock(
                    in_channels=prev_channels,
                    out_channels=down_widths[i],
                    condition_dim=condition_dim,
                    depth=down_depths[i],
                    downsample=i > 0,
                    block_layer=down_block_layer[i],
                )
            )
            prev_channels = down_widths[i]

        self.mid: nn.Module = DownBlock(
            in_channels=prev_channels,
            out_channels=mid_width,
            condition_dim=condition_dim,
            depth=mid_depth,
            downsample=True,
            block_layer=mid_block_layer,
        )
        prev_channels = mid_width

        self.up = nn.ModuleList()
        for i in range(len(up_widths)):
            self.up.append(
                UpBlock(
                    in_channels=prev_channels,
                    skip_channels=down_widths[-1 - i],
                    out_channels=up_widths[i],
                    condition_dim=condition_dim,
                    depth=up_depths[i],
                    fuse_method=fuse_method,
                    upsample_method=upsample_method,
                    interpolation=interpolation,
                    block_layer=up_block_layer[i],
                )
            )
            prev_channels = up_widths[i]

        self.proj_out = nn.Conv2d(up_widths[-1], out_channels, 1)

    def encode(self, x: Tensor, c: Tensor | None = None) -> list[Tensor]:
        features = []
        h = x
        for block in self.down:
            h = block(h, c)
            features.append(h)
        features.append(self.mid(h, c))
        return features

    def decode(self, features: Sequence[Tensor], c: Tensor | None = None) -> Tensor:
        h = features[-1]
        for i in range(len(self.up)):
            h = self.up[i](h, features[-2 - i], c)
        return h

    def forward(self, x: Tensor, c: Tensor | None = None) -> Tensor:
        features = self.encode(x, c)
        output = self.decode(features, c)
        return self.proj_out(output)

from collections.abc import Callable
from typing import Literal, TypeAlias

from torch import nn

FuseMethod: TypeAlias = Literal["add", "concat"]
"""
Methods for fusing skip connections.

Attributes
----------
add
    Fuse by element-wise addition.
concat
    Fuse by concatenation followed by linear projection.
"""

DownsampleMethod: TypeAlias = Literal["conv", "avg-pool"]
"""
Methods for downsampling features.

Attributes
----------
conv
    Stride convolution.
avg-pool
    Average pooling followed py linear projection.
"""

UpsampleMethod: TypeAlias = Literal["interpolate", "pixel-shuffle"]
"""
Methods for upsampling features.

Attributes
----------
interpolate
    Interpolation.
pixel-shuffle
    Linear projection followed by pixel shuffle.
"""

InterpolationMethod: TypeAlias = Literal["bilinear", "nearest-exact"]
"""
Interpolation methods.

Attributes
----------
bilinear
    Bilinear interpolation.
nearest-exact
    Nearest neighbor interpolation.
"""

BlockConstructor: TypeAlias = Callable[[int, int | None], nn.Module]
NormConstructor: TypeAlias = Callable[[int], nn.Module]
ActConstructor: TypeAlias = Callable[[], nn.Module]
MLPConstructor: TypeAlias = Callable[[int], nn.Module]

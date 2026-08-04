from collections.abc import Callable
from typing import Literal, TypeAlias

from torch import nn

BlockConstructor: TypeAlias = Callable[[int, int | None], nn.Module]
NormConstructor: TypeAlias = Callable[[int], nn.Module]
ActConstructor: TypeAlias = Callable[[], nn.Module]
FuseMethod: TypeAlias = Literal["add", "concat"]
UpsampleMethod: TypeAlias = Literal["interpolate", "pixel-shuffle"]
InterpolationMethod: TypeAlias = Literal["bilinear", "nearest-exact"]

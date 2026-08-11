from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch._subclasses import FakeTensorMode


class GlobalAvgPool2dTLC(nn.Module):
    """
    Global average pooling module with TLC.
    """

    def __init__(
        self,
        input_size: tuple[int, int] | None = None,
    ):
        """
        Constructor

        Parameters
        ----------
        input_size
            The dimensions (H, W) passed to this layer during training.
        """
        super().__init__()

        self.input_size = input_size

    def forward(self, x: Tensor) -> Tensor:
        if self.input_size is None:
            # If input size was not specified in the constructor,
            # the first forward pass should receive an input size tensor
            self.input_size = x.shape[-2:]

        h, w = x.shape[-2:]
        input_height, input_width = self.input_size

        kh = min(h, input_height)
        kw = min(w, input_width)

        s = x.cumsum(dim=-1).cumsum_(dim=-2)
        s = nn.functional.pad(s, (1, 0, 1, 0))
        s1, s2, s3, s4 = (
            s[:, :, :-kh, :-kw],
            s[:, :, :-kh, kw:],
            s[:, :, kh:, :-kw],
            s[:, :, kh:, kw:],
        )
        out = (s4 + s1 - s2 - s3) / (kh * kw)

        oh, ow = out.shape[-2:]
        pad2d = ((w - ow) // 2, (w - ow + 1) // 2, (h - oh) // 2, (h - oh + 1) // 2)
        out = nn.functional.pad(out, pad2d, mode="replicate")
        return out


def _apply(model: nn.Module):
    for n, m in model.named_children():
        if isinstance(m, nn.AdaptiveAvgPool2d):
            if isinstance(m.output_size, Sequence):
                assert m.output_size[0] == 1 and m.output_size[1] == 1
            else:
                assert m.output_size == 1
            m = GlobalAvgPool2dTLC()
            setattr(model, n, GlobalAvgPool2dTLC())

        else:
            _apply(m)


def apply_tlc(
    model: nn.Module,
    train_size: Sequence[int],
) -> nn.Module:
    """
    Apply Test-time Local Converter (TLC) to all supported modules in the model.

    Currently only global average pooling (`AdaptiveAvgPool2d` with output size 1)
    is supported.

    See: [https://arxiv.org/abs/2112.04491](https://arxiv.org/abs/2112.04491).

    Parameters
    ----------
    model
        The model to apply TLC to.
    train_size
        A sequence (..., C, H, W) specifying the size of the tensors passed to
        the model during training, excluding the batch dimension.

    Returns
    -------
    model
        The module passed as `model`.
    """
    _apply(model)

    param = next(model.parameters())
    device, dtype = param.device, param.dtype

    with FakeTensorMode(allow_non_fake_inputs=True):
        x = torch.empty(1, *train_size, device=device, dtype=dtype)
        _ = model(x)

    return model

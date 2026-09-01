import torch

import yaunet.blocks


def test_blocks():
    x = torch.randn(1, 64, 256, 256)

    for block_name in yaunet.blocks.__all__:
        cls = getattr(yaunet.blocks, block_name)
        block = cls(64)

        with torch.no_grad():
            out = block(x)

        assert x.shape == out.shape


def test_blocks_condition():
    x = torch.randn(1, 64, 256, 256)
    c = torch.randn(1, 128)

    for block_name in yaunet.blocks.__all__:
        cls = getattr(yaunet.blocks, block_name)
        block = cls(64, 128)

        with torch.no_grad():
            out = block(x, c)

        assert x.shape == out.shape

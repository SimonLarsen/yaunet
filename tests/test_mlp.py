import torch

import yaunet.mlp


def test_mlp():
    x = torch.randn(1, 64, 256, 256)

    for mlp_name in yaunet.mlp.__all__:
        cls = getattr(yaunet.mlp, mlp_name)
        mlp = cls(64)

        with torch.no_grad():
            out = mlp(x)

        assert x.shape == out.shape

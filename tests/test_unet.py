import torch

from yaunet.unet import UNet


def test_unet():
    model = UNet(
        in_channels=3,
        out_channels=2,
        down_widths=(16, 32, 64),
        mid_width=128,
        up_widths=(64, 32, 16),
        down_depths=(1, 2, 3),
        mid_depth=3,
        up_depths=(3, 2, 1),
    )

    x = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 2, 256, 256)


def test_unet_condition():
    model = UNet(
        in_channels=3,
        out_channels=2,
        condition_dim=64,
        down_widths=(16, 32, 64),
        mid_width=128,
        up_widths=(64, 32, 16),
        down_depths=(1, 2, 3),
        mid_depth=3,
        up_depths=(3, 2, 1),
    )

    x = torch.randn(1, 3, 256, 256)
    c = torch.randn(1, 64)

    with torch.no_grad():
        out = model(x, c)

    assert out.shape == (1, 2, 256, 256)


def test_unet_wrapper():
    model = UNet(
        wrapper=True,
        in_channels=3,
        out_channels=2,
        condition_dim=64,
        down_widths=(16, 32, 64),
        mid_width=128,
        up_widths=(64, 32, 16),
        down_depths=(1, 2, 3),
        mid_depth=3,
        up_depths=(3, 2, 1),
    )

    x = torch.randn(1, 3, 256, 256)
    features = torch.randn(1, 128, 32, 32)

    with torch.no_grad():
        out = model(x, wrap=features)

    assert out.shape == (1, 2, 256, 256)

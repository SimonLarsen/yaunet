---
hide:
- navigation
---

--8<-- "README.md:1:15"

## Usage

### Creating a basic U-Net model

Just import the [`UNet`][yaunet.unet.UNet] class from `yaunet.unet` and instantiate:

```python
import torch
from yaunet.unet import UNet

model = UNet(
    in_channels=3,
    out_channels=1,
    down_widths=(32, 64, 128, 256),
    mid_width=512,
    up_widths=(256, 128, 64, 32),
    down_depths=(2, 2, 3, 4),
    mid_depth=2,
    up_depths=(4, 3, 2, 2),
)

x = torch.randn(2, 3, 256, 256)

pred = model(x)
print(x.shape)  # torch.Size([2, 1, 256, 256])
```

### Changing block types

A number of blocks are implemented in [`yaunet.blocks`][yaunet.blocks].
Pass the blocks' constructors to [`UNet`][yaunet.unet.UNet] to use:

```python
from yaunet.unet import UNet
from yaunet.blocks import ConvNextBlock, ViTBlock, ResNetBlock

model = UNet(
    # ...
    down_block_layer=ConvNextBlock,
    mid_block_layer=ViTBlock,
    up_block_layer=ResNetBlock,
)
```

`down_block_layer` and `up_block_layer` and be a sequence if the blocks should be different at each level:

```python
model = UNet(
    # ...
    down_widths=(32, 64, 128, 256),
    down_depths=(2, 2, 9, 2),
    down_block_layer=(ResNetBlock, ResNetBlock, ConvNextBlock, ConvNextBlock),
)
```

### Setting block parameters

Use [`functools.partial`][functools.partial] to set block parameters:

```python
from yaunet.blocks import ConvNextBlock
from functools import partial

cheap_block = partial(
    ConvNextBlock,
    kernel_size=3,
    expand_factor=2.0,
)

expensive_block = partial(
    ConvNextBlock,
    kernel_size=7,
    expand_factor=4.0,
)

model = UNet(
    # ...
    down_block_layer=cheap_block
    mid_block_layer=expensive_block,
    up_block_layer=cheap_block,
)
```

### Conditioning the model

All blocks support AdaLN-Zero-like conditioning <a href="#ref1">[1]</a>.
Set `condition_dim` in `UNet` and pass a second tensor with shape `(B, condition_dim)` to condition the model:

```python
unet = UNet(
    # ...
    condition_dim=128,
)

x = torch.randn(2, 3, 256, 256)
c = torch.randn(2, 128)

pred = model(x, c)
```

### Using it as a wrapper for a second model

You can use the model as a feature upscaler for another (potentially frozen) model by setting `wrapper=True`
and passing the features as `wrap` to `UNet.forward`.
The passed features should have shape `(B, mid_width, H/p, W/p)` where `p = 2 ** len(down_widths)`.

```python
from transformers import DINOv3ViTBackbone
from yaunet import UNet

backbone = DINOv3ViTBackbone.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
backbone.eval().requires_grad_(False)

model = UNet(
    wrapper=True,
    in_channels=3,
    out_channels=3,
    down_widths=(32, 64, 128, 256),
    mid_width=backbone.config.hidden_size,
    up_widths=(256, 128, 64, 32),
    down_depths=(2, 2, 3, 4),
    mid_depth=0,
    up_depths=(5, 4, 2, 2),
)

x = torch.randn(2, 3, 256, 256)

features = backbone(x).feature_maps[-1]
print(features.shape)  # torch.Size([2, 384, 16, 16])

pred = model(x, wrap=features)
print(pred.shape)  # torch.Size([2, 3, 256, 256])
```

## References

<ul class="ul-bib">
<li id="ref1">[1] Peebles, William, and Saining Xie. "Scalable diffusion models with transformers." <i>2023 IEEE/CVF International Conference on Computer Vision (ICCV)</i>. IEEE, 2023.</li>
</ul>

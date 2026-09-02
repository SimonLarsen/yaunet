Yet Another U-Net
=================

yaunet (*ya' you net*) is a simple, modular U-Net implementation in PyTorch.

Features:
* Simple implementation only dependent on [PyTorch](https://pytorch.org) and [einops](https://github.com/arogozhnikov/einops).
* Aims to be small, modular and highly configurable.
* Provides a decent number of commonly used building blocks.
* All blocks support AdaLN-Zero-like conditioning for training conditional models e.g. class- and/or time-conditioned diffusion models.

## Installation

```sh
pip install yaunet  # or `uv add yaunet`
```

## Documentation

Read the API documentation at [yaunet.readthedocs.io](https://yaunet.readthedocs.io).

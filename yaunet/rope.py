import torch
from torch import Tensor


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    num_patches = sin.size(-2)

    q_prefix_tokens, q_patches = q.split((q.size(2) - num_patches, num_patches), dim=-2)
    k_prefix_tokens, k_patches = k.split((k.size(2) - num_patches, num_patches), dim=-2)

    q_patches = (q_patches * cos) + (_rotate_half(q_patches) * sin)
    k_patches = (k_patches * cos) + (_rotate_half(k_patches) * sin)

    q = torch.cat((q_prefix_tokens, q_patches), dim=-2)
    k = torch.cat((k_prefix_tokens, k_patches), dim=-2)
    return q, k

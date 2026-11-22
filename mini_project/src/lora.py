"""LoRA for Conv2d (W12) and encoder freezing (W9) for parameter-efficient fine-tuning.

LoRAConv2d adds a low-rank residual B(A(x)) to a frozen base conv: A is a
k x k conv in->r (Kaiming init) and B a 1x1 conv r->out (zero init), so the
wrapped model equals the base model at initialisation. Because B is 1x1, B∘A
is itself a k x k conv and can be merged back into the base weights.
"""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float | None = None):
        super().__init__()
        if base.groups != 1:
            raise ValueError('LoRAConv2d supports groups=1 only')
        self.base, self.rank = base, int(rank)
        self.scale = (float(alpha) if alpha is not None else float(rank)) / self.rank
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Conv2d(base.in_channels, self.rank, base.kernel_size, stride=base.stride,
                           padding=base.padding, dilation=base.dilation, bias=False,
                           padding_mode=base.padding_mode)
        self.B = nn.Conv2d(self.rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))

    def merged_conv(self):
        conv = copy.deepcopy(self.base)
        delta = torch.einsum('or,rikl->oikl', self.B.weight[:, :, 0, 0], self.A.weight)
        with torch.no_grad():
            conv.weight += self.scale * delta
        for p in conv.parameters():
            p.requires_grad_(True)
        return conv


def _replace_children(module, fn):
    for name, child in module.named_children():
        new = fn(child)
        if new is not child:
            setattr(module, name, new)
        else:
            _replace_children(child, fn)


def apply_lora(model, rank, alpha=None, include_1x1=False):
    """Freeze everything, wrap every 3x3 (groups=1) conv with LoRA; return trainable count.

    include_1x1 also wraps 1x1 groups=1 convs. NAFNet's 3x3 convs are depthwise,
    so without it only intro/ending would adapt.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    def wrap(child):
        kernels = ((3, 3), (1, 1)) if include_1x1 else ((3, 3),)
        if isinstance(child, nn.Conv2d) and child.kernel_size in kernels and child.groups == 1:
            return LoRAConv2d(child, rank, alpha)
        return child

    _replace_children(model, wrap)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def merge_lora(model):
    """Return the model with every LoRAConv2d folded into a plain Conv2d (in place)."""
    _replace_children(model, lambda c: c.merged_conv() if isinstance(c, LoRAConv2d) else c)
    return model


def freeze_encoder(model):
    """Freeze the encoder path (model.encoder_modules()); return the frozen count."""
    frozen = 0
    for module in model.encoder_modules():
        for p in module.parameters():
            p.requires_grad_(False)
            frozen += p.numel()
    return frozen

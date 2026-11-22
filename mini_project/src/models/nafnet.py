"""NAFNet (Chen et al., "Simple Baselines for Image Restoration", ECCV 2022).

Architecture vendored from megvii-research/NAFNet
  https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/NAFNet_arch.py
  https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/arch_util.py
Copyright (c) 2022 megvii-model. Released under the MIT License.

Local changes: LayerNorm2d is written with native autograd (the original custom
autograd.Function uses ctx.saved_variables, removed from recent PyTorch); it
computes the same function and keeps the same parameter names, so official
checkpoints load with strict=True. The Local_Base / NAFNetLocal test-time
variant is not vendored. GrayNAFNet wraps the RGB model for grayscale pages.
"""
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

NAFNET_WEIGHTS = Path(os.environ.get(
    'NAFNET_WEIGHTS',
    Path(__file__).resolve().parents[2] / 'data' / 'external' / 'nafnet' / 'NAFNet-SIDD-width32.pth'))


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, padding=0, stride=1, groups=1, bias=True),
        )
        self.sg = SimpleGate()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, padding=0, stride=1, groups=1, bias=True)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=(), dec_blk_nums=()):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan = chan * 2
        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        x = self.ending(x)
        x = x + inp
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        return F.pad(x, (0, mod_pad_w, 0, mod_pad_h))


class GrayNAFNet(nn.Module):
    """[B,1,H,W] -> repeat to RGB -> NAFNet -> mean over channels -> [B,1,H,W].

    Pads to a multiple of 16 with replicate padding (instead of NAFNet's zero
    padding, which would add a black border on white pages), then crops back.
    """

    def __init__(self, width=32, enc_blks=(2, 2, 4, 8), middle_blk_num=12, dec_blks=(2, 2, 2, 2)):
        super().__init__()
        self.net = NAFNet(img_channel=3, width=width, middle_blk_num=middle_blk_num,
                          enc_blk_nums=list(enc_blks), dec_blk_nums=list(dec_blks))

    def forward(self, noisy):
        height, width = noisy.shape[-2:]
        m = self.net.padder_size
        x = F.pad(noisy, (0, (-width) % m, 0, (-height) % m), mode='replicate')
        y = self.net(x.repeat(1, 3, 1, 1)).mean(dim=1, keepdim=True)
        return y[..., :height, :width]


def load_nafnet_sidd(path=NAFNET_WEIGHTS):
    """GrayNAFNet with official SIDD width-32 weights ('params' or 'state_dict' format)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt.get('params', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
    model = GrayNAFNet()
    model.net.load_state_dict(state, strict=True)
    return model

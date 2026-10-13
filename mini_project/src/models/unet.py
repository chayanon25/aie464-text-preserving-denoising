"""Three-level small U-Net; padding/cropping preserves any page dimensions."""
import torch
from torch import nn
from torch.nn import functional as F


def block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU())


class SmallUNet(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.enc1, self.enc2, self.enc3 = block(1, width), block(width, 2*width), block(2*width, 4*width)
        self.bridge = block(4*width, 8*width)
        self.dec3, self.dec2, self.dec1 = block(12*width, 4*width), block(6*width, 2*width), block(3*width, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, noisy):
        height, width = noisy.shape[-2:]
        x = F.pad(noisy, (0, (-width) % 8, 0, (-height) % 8), mode='replicate')
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        x = self.bridge(F.max_pool2d(c, 2))
        for skip, decoder in ((c, self.dec3), (b, self.dec2), (a, self.dec1)):
            x = decoder(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='nearest'), skip], dim=1))
        return self.head(x).sigmoid()[..., :height, :width]

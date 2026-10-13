"""Small spatially preserving baseline for the Phase 2A sanity run."""

from torch import nn


class ShallowCNN(nn.Module):
    """Five 3x3 convolutions; grayscale [B,1,H,W] -> [B,1,H,W] in [0,1]."""

    def __init__(self, width=16):
        super().__init__()
        layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU()]
        for _ in range(3):
            layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.ReLU()])
        layers.extend([nn.Conv2d(width, 1, 3, padding=1), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, noisy):
        return self.net(noisy)

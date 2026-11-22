"""Small DnCNN variant: seven convolutions, residual noise prediction."""
from torch import nn


class DnCNNLite(nn.Module):
    def __init__(self, width=32, depth=7, norm='bn'):
        super().__init__()
        if norm not in ('bn', None):
            raise ValueError(f'Unsupported norm: {norm!r}')
        layers = [nn.Conv2d(1, width, 3, padding=1), nn.ReLU()]
        for _ in range(depth - 2):
            if norm == 'bn':
                layers += [nn.Conv2d(width, width, 3, padding=1, bias=False),
                           nn.BatchNorm2d(width), nn.ReLU()]
            else:
                # Without BN the conv needs its own bias.
                layers += [nn.Conv2d(width, width, 3, padding=1, bias=True), nn.ReLU()]
        layers += [nn.Conv2d(width, 1, 3, padding=1)]
        self.noise = nn.Sequential(*layers)

    def forward(self, noisy):
        # Do not clamp here: residual learning needs gradients outside [0,1].
        return noisy - self.noise(noisy)

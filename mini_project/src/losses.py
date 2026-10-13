"""L1 + optional normalized Sobel gradient L1; no SSIM training term."""
import torch
from torch import nn
from torch.nn import functional as F


class DenoisingLoss(nn.Module):
    def __init__(self, edge_weight=0.0):
        super().__init__()
        if edge_weight < 0:
            raise ValueError('edge_weight must be nonnegative')
        self.edge_weight = edge_weight
        x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8
        self.register_buffer('sobel', torch.stack([x, x.T]).unsqueeze(1))

    def forward(self, prediction, clean):
        if prediction.shape != clean.shape:
            raise ValueError('Loss requires aligned matching tensors')
        # Float32 loss, including Sobel, even when the forward uses AMP.
        with torch.amp.autocast(device_type=prediction.device.type, enabled=False):
            prediction, clean = prediction.float(), clean.float()
            l1 = F.l1_loss(prediction, clean)
            edge = l1.new_zeros(())
            if self.edge_weight:
                edges = [F.conv2d(F.pad(t, (1, 1, 1, 1), mode='reflect'), self.sobel)
                         for t in (prediction, clean)]
                edge = F.l1_loss(*edges)
            return {'total': l1 + self.edge_weight * edge, 'l1': l1, 'edge': edge}

"""Model registry shared by training, evaluation and the Colab notebook."""
from .dncnn import DnCNNLite
from .nafnet import GrayNAFNet
from .shallow_cnn import ShallowCNN
from .unet import SmallUNet

# Midterm experiment name -> architecture name used by build_model.
MIDTERM_EXPERIMENTS = {
    'cnn_l1': 'cnn',
    'dncnn_lite': 'dncnn_bn',
    'unet_l1': 'unet',
    'unet_edge': 'unet',
}


def build_model(name: str):
    """Build an architecture by name with the midterm hyperparameters."""
    if name == 'cnn':
        return ShallowCNN(width=16)
    if name == 'dncnn_bn':
        return DnCNNLite(width=32, depth=7, norm='bn')
    if name == 'dncnn_nobn':
        return DnCNNLite(width=32, depth=7, norm=None)
    if name == 'unet':
        return SmallUNet(width=16)
    if name == 'nafnet':
        return GrayNAFNet()   # random init; see nafnet.load_nafnet_sidd for weights
    raise ValueError(f'Unknown model: {name!r}')

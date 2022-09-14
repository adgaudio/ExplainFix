from explainfix.models.util import iter_conv2d
import torch as T


def fix_spatial_conv2d(model: T.nn.Module):
    """
    Set conv2d.requires_grad=False on all 2d spatial convolution filters so
    they are not learned during backprop.  This applies to both the .weight and
    .bias fields of T.nn.Conv2d.

    Modify model in-place.

    Note: A spatial convolution has a kernel size > 1 in all dimensions."""
    for module in iter_conv2d(model, include_spatial=True, include_1x1=False,):
        module.requires_grad_(False)

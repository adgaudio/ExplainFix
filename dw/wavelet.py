"""
"""
import torch as T
import torch.nn.functional as F
import pytorch_wavelets as pyw
import torchvision.transforms as tvt
from collections import OrderedDict


def wavelet_coefficients_as_tensorimage(approx, detail, normalize=False):
    norm11 = lambda x: (x / max(x.min()*-1, x.max()))  # into [-1,+1] preserving sign
    fixed_dims = detail[0][0].shape[:-3] # num images in minibatch, num channels, etc
    output_shape = fixed_dims + (
        detail[0][0].shape[-2]*2,  # input img height
        detail[0][0].shape[-1]*2)  # input img width
    im = T.zeros(output_shape)
    #  if normalize:
        #  approx = norm11(approx)
    im[..., :detail[-1].shape[-2], :detail[-1].shape[-1]] = approx if approx is not None else 0
    for level in detail:
        lh, hl, hh = level.unbind(-3)
        h,w = lh.shape[-2:]
        if normalize:
            lh, hl, hh = [norm11(x) for x in [lh, hl, hh]]
        #  im[:h, :w] = approx
        im[..., 0:h, w:w+w] = lh  # horizontal
        im[..., h:h+h, :w] = hl  # vertical
        im[..., h:h+h, w:w+w] = hh  # diagonal
    return im

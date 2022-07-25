from itertools import product
import dataclasses as dc
from typing import Optional, Callable, List, Tuple
import math
import numpy as np
import torch as T
from matplotlib import pyplot as plt
from explainfix.models.util import iter_conv2d
from explainfix.kernel import dct_basis_nd


@dc.dataclass
class ExplainSteerLayerwiseResult:
    """Stores result of an ExplainSteer applied to spatial convolution layers"""
    ed: List[T.Tensor]
    e1: List[T.Tensor]
    e0: List[T.Tensor]

    def __iter__(self):
        for layer_ed, layer_e1, layer_e0 in zip(self.ed, self.e1, self.e0):
            yield (layer_ed, layer_e1, layer_e0)


def explainsteer_layerwise_without_saliency(model) -> ExplainSteerLayerwiseResult:
    """Apply explainsteer without saliency to all spatial conv2d layers of a model"""
    spectra_e2, spectra_e1, spectra_e0 = [],[],[]
    for conv2d in iter_conv2d(model, include_spatial=True, include_1x1=False):
        O,I,H,W = conv2d.weight.shape
        device = conv2d.weight.device
        saliency_weight = T.ones(O*I, dtype=T.float) / (O*I)
        e_d,e_1,e_0 = get_spectra(conv2d.weight.reshape(O*I, H*W).detach(),
                                  saliency_weight, (H,W), device)
        spectra_e2.append(e_d)
        spectra_e1.append(e_1)
        spectra_e0.append(e_0)
    return ExplainSteerLayerwiseResult(spectra_e2, spectra_e1, spectra_e0)


YHat, Y, Scalar = T.Tensor, T.Tensor, T.Tensor  # for type checking


def explainsteer_layerwise_with_saliency(
        model:T.nn.Module, loader:T.utils.data.DataLoader,
        device:str, num_minibatches:int=float('inf'),
        grad_cost_fn:Callable[['YHat', 'Y'],'Scalar']=lambda yhat, y: (yhat*y).sum()

) -> ExplainSteerLayerwiseResult:
    """
    Apply explainsteer with saliency to all spatial conv2d layers of a model.
    This tells which horizontal and vertical components are most useful to the model.

    Args:
        model: a pytorch model or Module containing spatial 2d convolution layers  (T.nn.Conv2d)
        loader: pytorch data loader
        device: pytorch device, like 'cpu' or 'cuda:0'
        num_minibatches: over how many images to compute the gradients.  We
            think if the images are similar, then you don't actually need a large
            number at all.
        grad_cost_fn: a "loss" used to compute saliency.
            `yhat` is model output. `y` is ground truth.
            The default assumes `yhat=model(x)` and `y` are the same shape.
            Probably `lambda yhat, y: yhat.sum()` also works in many cases.

    Example Usage:
        spectra = explainsteer_layerwise_with_saliency(model, loader, device)
        for layer_idx, (e2, e1, e0) in enumerate(spectra):
            print(layer_idx, e0)
        plot_spectrum_ed(spectra.ed)
    """
    model.to(device, non_blocking=True)
    model.eval()
    # get the set of all 2d spatial filters for all layers of model
    filters_all_layers = [
        x.weight for x in iter_conv2d(
            model, include_spatial=True, include_1x1=False)]
    # set all filters to requires grad so we can get gradients on them
    [x.requires_grad_(True) for x in filters_all_layers]
    # prepare output as list of spectra (e_d, e_1, e_0)
    spectra = SumList()
    N = 0
    for n, (x,y) in enumerate(loader):
        if n >= num_minibatches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        batch_size = x.shape[0]
        N += batch_size
        yhat = model(x)
        # rescale yhat to all ones or zeros.
        with T.no_grad():
            yhat /= yhat
            #  yhat[yhat != 0] /= yhat[yhat!=0]

        # get gradients, making all predictions for correct classes correct
        #  (y*yhat).sum().backward(retain_graph=False)
        grads_all_layers = T.autograd.grad(
            grad_cost_fn(yhat, y), filters_all_layers, retain_graph=False)
        with T.no_grad():
            # get the spectra for every layer
            spectra_e2, spectra_e1, spectra_e0 = [], [], []
            #  for filters, grads in zip(filters_all_layers, grads_all_layers):
            for filters, grads_filters in zip(
                    filters_all_layers,
                    grads_all_layers,):

                weighted_sum = grads_filters.abs().sum((-1,-2), keepdims=True)
                O,I,H,W = filters.shape
                e_d, e_1, e_0 = get_spectra((filters).reshape(O*I,H*W), weighted_sum.reshape(1, O*I), (H,W), device)

                spectra_e2.append(e_d)
                spectra_e1.append(e_1)
                spectra_e0.append(e_0)
            spectra += (spectra_e2, spectra_e1, spectra_e0)
    e_2, e_1, e_0 = spectra.lists
    return ExplainSteerLayerwiseResult(e_2, e_1, e_0)


class SumList:
    """
    Helper code to sum list elements.  Best explained by example:
    >>> x = SumList()
    >>> x += ( [1,1,1], [10,10] )
    >>> x += ( [1,1,1], [10,10] )
    >>> x
        ( [2,2,2], [20,20] )
    >>> x[0]
        [2,2,2]
    >>> x[1]
        [20,20]
    """
    def __init__(self):
        self.lists = None

    def __iadd__(self, lists):
        if self.lists is None:
            self.lists = lists
        else:
            for add_this, to_this in zip(lists, self.lists):
                for i in range(len(to_this)):
                    to_this[i] += add_this[i]
        return self

    def __repr__(self):
        return repr(self.lists)

    def __getitem__(self, list_idx):
        return self.lists[list_idx]


def get_spectra(F:T.Tensor, w: T.Tensor, shape:Tuple[int], device:str) -> Tuple[T.Tensor,T.Tensor,Optional[T.Tensor]]:
    """Compute e^d, and then use back-projection to get e^1 and e^0.  For e^0, assume kernel shape is square.

    :F: a NxM matrix where each row is a flattened spatial convolution filter kernel of some shape, in any dimension.
        e.g. N is the count of spatial filters, and M=9 for when representing 3x3 filters.
    :w: (N, ) or (1,N) tensor of saliency weights for each spatial filter.  Assume w > 0 (untested with any w_i<0).
    :shape: the shape of the convolution filter kernel (H,W, ...) where both all dimensions H,W,... > 1
    :device: pytorch device like 'cpu' or 'cuda:0'
    :returns: three spectra e^d, e^1, e^0 of lengths (prod(shape), sum(shape), [None,shape[0]] ) where the last is None unless the shape is square.
    """
    # DCT-II basis, vectors on rows, constructed by itertools.product.
    B = T.from_numpy(
        dct_basis_nd(shape).reshape(math.prod(shape), math.prod(shape))
    ).float().to(device, non_blocking=True)
    w = w.reshape(1, F.shape[0]).to(device, non_blocking=True)

    # get the energies e^(d)
    e_d = w@(F@B.T).abs()  # dct-II basis
    e_d = e_d.squeeze()
    #  e_d = (F).abs().mean(0)  # canonical basis
    #  e_d = e_d/T.linalg.norm(e_d)
    assert len(e_d) == math.prod(shape), 'sanity check'
    # get the energies e^(1)
    e1_dims = [T.zeros(x).to(device, non_blocking=True)
               for x in shape]
    d = len(shape)
    # --> note: assumes basis constructed from 1d vectors, in same order output by itertools.product(...)
    for e_j, dims in zip(e_d, product(*(range(x) for x in shape))):
        tmp = e_j**(1/d)
        for spatial_dimension, dim_idx in enumerate(dims):
            # e.g. in d=2, spatial_dimensions are "rows" or "columns" of a 2d kernel
            # dim_idx is the nth 1-d basis vector in that spatial dimension.
            e1_dims[spatial_dimension][dim_idx] += tmp
    e_1 = T.hstack(e1_dims)
    # --> the simpler 2-d version for getting e^1:
    #  e1_dim1 = T.zeros(H).to(device, non_blocking=True)
    #  e1_dim2 = T.zeros(W).to(device, non_blocking=True)
    #  assert len(e_d) == H*W, 'sanity check'
    #  for e_j, (dim1_idx, dim2_idx) in zip(e_d, product(range(H), range(W))):
    #      d = 2
    #      tmp = e_j**(1/d)
    #      e1_dim1[dim1_idx] += tmp
    #      e1_dim2[dim2_idx] += tmp
    # get the energies e^(0)
    #  e0 = T.vstack([e1_dim1, e1_dim2]).sum(0)  # assume square filter
    if all(x == shape[0] for x in shape):
        e_0 = T.vstack(e1_dims).sum(0)  # assume square filter
    else:
        e_0 = None

    # Re-order the e^d spectrum so it has a diagonal traversal ordering (order
    # basis filters from highest to lowest frequency).  Currently, it has a
    # raster-scan order (reshape will just concat rows together).
    idx_order = dravel(np.arange(B.shape[0]).reshape(int(np.sqrt(B.shape[0])), int(np.sqrt(B.shape[0]))))
    e_d = e_d[idx_order]

    return e_d, e_1, e_0


def dravel(M, edges_first=True):
    """
    Diagonal Traversal of a matrix, in lines parallel to top right to bottom
    left.  Optionally, sample the edges of the diagonal lines before the
    middles in stable order.  In this optional case, the matrix must be square.

    Outputs a vector of values of M in this diagonal scan order.

    (Used by ExplainSteer to assign numbers to the DCT-II basis filters
    ordering them by lowest to highest frequency)

    :M: matrix
    :edges_first: if True, sample the edges of the traversed diagonal lines
        before middles
    :returns: a flattened vector of M, in diagonal traversal ordering
    """
    z = np.zeros(M.shape)
    # invent a matrix with highest values (least important) in bottom right and
    # lowest (most important) in top right.  This naturally orders the matrix
    # in a diagonal line scan ordering, with lines traveling in the direction top
    # right to bottom left.
    max_ = 2*sum(x-1 for x in M.shape)
    z = 2*np.mgrid[:M.shape[0], :M.shape[1]].sum(0)
    assert z.max() == max_
    # augment the matrix to make values closer to the diagonal less important
    # but guarantee the difference is small enough that for each diagonal scan line:
    # the max value is less than the min of the line to the right;
    # the min value is greater than the max of the line to the left.
    if edges_first:
        # --> THIS part requires a square matrix
        if M.shape[0] != M.shape[1]:
            raise Exception("non-square matrix")
        z = z * max_/2-np.fliplr(np.abs(z-max_/2))
    # do the diagonal traversal
    return M.ravel()[z.ravel().argsort(kind='stable')]


def plot_spectrum_ed(ed, fig:plt.Figure=None, loglog=False) -> plt.Figure:
    """
    :ed: list of the e^d energy spectra, with one spectrum for each spatial
        convolution layer of the model.
    :fig: A matplotlib figure (or subfigure) in which to generate the visuals.
    """
    if fig is None:
        fig = plt.figure()
    # --> convert to matrices for viewing
    E = ragged_to_matrix(ed).T.cpu().numpy()
    # set up the figure (or subfigure)
    gs1 = fig.add_gridspec(nrows=4, ncols=4)#, left=0.05, right=0.48, wspace=0.05)
    fig.add_subplot(gs1[1:, :3])
    fig.add_subplot(gs1[1:, 3], sharey=fig.axes[-1])
    fig.add_subplot(gs1[0, :3], sharex=fig.axes[-2])
    axes = fig.axes
    # plot main heatmap
    axes[0].set_xlabel('Spatial Layer Index')
    axes[0].set_ylabel('Basis Vector Index')
    if loglog:
        Emain = np.log1p(np.log1p(E))
        axes[0].set_ylabel(f'{axes[0].get_ylabel()} (Log Log Scale)')
    else:
        Emain = E
    axes[0].imshow(Emain, vmin=0, aspect='auto', cmap='Greens')
    # --> ensure yticks are integers
    axes[0].yaxis.get_major_locator().set_params(integer=True)
    # bar plots
    axes[1].barh(np.arange(E.shape[0]), E.sum(1))
    axes[2].bar(np.arange(E.shape[1]), E.sum(0))
    # ensure bar plots have no labels or ticks
    [ax.axis('off') for ax in axes[1:]]
    return fig


def ragged_to_matrix(ragged_vectors:List[T.Tensor]):

    """Convert list of ragged 1-d vectors by padding zeros to the right
    for any vector that is too small.

    :ragged_vectors: list of N 1-d vectors of varying length.

    :return: (N,M) matrix where M is max length of all vectors"""
    N = max(len(x) for x in ragged_vectors)
    T.nn.functional.pad(T.tensor([1]), [0,1])
    M = T.stack([T.nn.functional.pad(x, [0, N-len(x)])
                 for x in ragged_vectors])
    return M

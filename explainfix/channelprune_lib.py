import torch as T
import numpy as np
import math
from typing import Callable, List
from torch.utils.data import DataLoader
from explainfix.models import prune_model, iter_conv2d


YHat = Y = Scalar = T.Tensor  # for type checking


def channelprune(
        model:T.nn.Module, pct:float,
        grad_cost_fn:Callable[['YHat', 'Y'], 'Scalar'],
        loader:DataLoader, device:str, num_minibatches:int=float('inf'),
        zero_fill_method:str='none'
):
    """
    ChannelPrune makes models smaller by spatial filter weight saliency.
    Updates the given `model` in-place.
    You should fine-tune the model after pruning.
    Works with completely random models, and with pre-trained models.

    Args:
        model: pytorch model to remove weights and connections from
        pct: The percentage of least salient weights to zero.  In [0,100].
            Larger numbers result in more computational efficiency.  Small numbers
            may improve predictive performance of the input model (with
            fine-tuning).  For more computationally efficient models, choose the
            largest number you can.  Note that setting `pct` way too large (e.g.
            `pct=100`) will cause an error.  Setting it too large will hurt model
            performance.
        grad_cost_fn: The saliency score uses a gradient.  This function
            determines the loss to use for the gradient.  `yhat = model(X)` is the
            model's predictions.  Y is the ground truth label.
            Examples:
                grad_cost_fn=lambda yhat, y: yhat[:, y].sum()  # get only the correct class (assuming y contains an index value for each sample)
                grad_cost_fn=lambda yhat, y: (yhat * y).sum()  # get only the correct class (assuming y contains a vector over classes for each sample)
        loader: A pytorch data loader used to compute spatial filter saliency
            scores.  It should generate tuples of form (X, y) where X is model
            input, `yhat = model(X)` and `y` is ground truth.
        device: A pytorch device e.g. "cpu" or "cuda".
        num_minibatches: the number of minibatches from the data loader to use.
        zero_fill_method: Whether to re-initialize the spatial filters
            that were zeroed but not pruned.  'none' or 'FillZero' or 'reset'
    Returns:
        None.  It modifies the model in-place.

    Algorithm:
        1. Assign each spatial filter in the model a scalar saliency score.
        2. Zero out the XX percent least salient (lowest score) filters.
        3. Remove rows and columns of spatial filters that are entirely zero,
           and remove neighboring connections to fix the network.
        4. For any spatial filters that were zeroed and not pruned, optionally
           re-initialize them via FillZero initialization or reset them to
           their original values before zeroing.
    """
    saliency_scores = get_spatial_filter_saliency(
        model=model, loader=loader, device=device,
        num_minibatches=num_minibatches, grad_cost_fn=grad_cost_fn)
    # store original values
    if zero_fill_method == 'reset':
        dct = {f'{name}.weight': conv2d.weight.data.clone() for name, conv2d in iter_conv2d(
            model, include_spatial=True, include_1x1=False, return_name=True)}

    zero_least_salient_spatial_filters(
        model=model, saliency_scores=saliency_scores, pct=pct)

    # restore the zeroed filters that can't be pruned
    if zero_fill_method == 'reset':
        for name, conv2d in iter_conv2d(model, include_spatial=True, include_1x1=False, return_name=True):
            f = conv2d.weight.data
            mask = (f != 0).sum((-1,-2))
            prunable_cols = mask.sum(0) == 0
            prunable_rows = mask.sum(1) == 0
            o = (~prunable_rows).outer(~prunable_cols)
            f[o] = dct[f'{name}.weight'][o]

    prune_model(model)

    if zero_fill_method == 'FillZero':
        for conv2d in iter_conv2d(model, include_spatial=True, include_1x1=False):
            fill_zero_initialization(conv2d)


def fill_zero_initialization(conv2d:T.nn.Conv2d):
    """In-place modify the spatial weights of given conv2d layer.
    Basically, re-initialize any values that are approximately zero.

    How:  A spatial filter kernel is a matrix.  If that matrix is entirely zero, use
    kaiming uniform initialization (standard pytorch).  If that matrix is
    sparse but not entirely zeros, replace zeros by sampling from a gaussian of
    mean and std of the values in the matrix.
    """
    # replace zeroed spatial weights with non-zero values using two methods.
    # first, for kernels that have some values zero, just sample the mean
    # and variance of that kernel and replace those missing values with gaussian noise.
    # second, for the kernels that are entirely zero, reinitialize weights with kaiming uniform values.
    tmp = conv2d.weight.detach().clone()
    # 1. fill the zeroed weights of kernels that are not entirely zero with random values
    tmp2 = conv2d.weight.detach().clone()
    tmp2 = T.nn.init.normal_(tmp2)*tmp.std((-1,-2), keepdim=True) + tmp.mean((-1,-2), keepdim=True)
    mask = T.isclose(tmp, T.tensor(0., dtype=tmp.dtype, device=tmp.device))
    tmp[mask] = tmp2[mask]
    #  2. fill the kernels that are entirely zero with kaiming uniform values
    mask = T.isclose(tmp, T.tensor(0., dtype=tmp.dtype, device=tmp.device))
    if mask.sum():
        tmp2 = T.nn.init.kaiming_uniform_(tmp2, a=math.sqrt(5))
        tmp[mask] = tmp2[mask]
    conv2d.weight.data[:] = tmp


def get_spatial_filter_saliency(
        model:T.nn.Module, loader:DataLoader, device:str,
        num_minibatches:int=float('inf'),
        grad_cost_fn:Callable[['YHat', 'Y'], 'Scalar']=lambda y,yhat: (y*yhat).sum()
        ) -> List[T.Tensor]:
    model.eval().to(device, non_blocking=True)
    # --> list of spatial filters in model
    filters_all_layers = [
        x.weight for x in iter_conv2d(model, include_spatial=True, include_1x1=False)]
    # --> remember requires_grad setting for each weight, and (temporarily) make all requires_grad=True
    rg = [x.requires_grad for x in filters_all_layers]
    [x.requires_grad_(True) for x in filters_all_layers]
    # --> get saliency scores
    saliency_scores = [f.new_zeros(f.shape[:2])  # initialize to zeros
                       for f in filters_all_layers]
    for n, minibatch in enumerate(loader):
        if n >= num_minibatches:
            break
        X = minibatch[0].to(device, non_blocking=True)
        y = minibatch[1].to(device, non_blocking=True)
        yhat = model(X)
        # --> rescale gradients (multiply partial derivatives by a scalar)
        with T.no_grad():
            yhat /= yhat
        # --> compute gradients
        # note: autograd implicitly accumulated gradients for each minibatch sample
        grads_all_layers = T.autograd.grad(  # might want to scale classes so we backprop only ones and zeros. or use deep taylor decomposition or something.  would be equivalent to scaling the gradients directly.
            grad_cost_fn(yhat, y), filters_all_layers, retain_graph=False)
        with T.no_grad():
            assert len(grads_all_layers) == len(filters_all_layers) == len(saliency_scores), 'code bug'
            # compute saliency by reducing partial derivatives to a single
            # number for each spatial filter.
            for idx, (filter, grad) in enumerate(zip(filters_all_layers, grads_all_layers)):
                assert filter.shape == grad.shape, 'code bug'
                _spatial_dims = list(range(-1*(filter.ndim-2), 0))
                #  tmp = (filter*grad).abs().sum(_spatial_dims)
                tmp = grad.abs().sum(_spatial_dims)  # alternative
                saliency_scores[idx] += (filter*grad).abs().sum(_spatial_dims)
    # --> restore the original requires_grad settings
    [x.requires_grad_(y) for x, y in zip(filters_all_layers, rg)]
    return saliency_scores


def zero_least_salient_spatial_filters(
        model:T.nn.Module, saliency_scores:List[T.Tensor], pct:float):
    """
    Modify a model in-place by zero-ing out the least salient spatial filters.

    Args:
        model: a pytorch model
        pct: the percent of spatial filters to zero out.  pct in [0,100]
        scores: a scalar saliency score number for each spatial filter in the model.
    """
    # push to cpu
    saliency_scores = [x.cpu() for x in saliency_scores]
    # boilerplate: associate a spatial filter to a layer index and global index
    tmp = T.hstack([x.reshape(-1).cpu() for x in saliency_scores])
    idxs = tmp.argsort()
    layer_idxs = T.hstack([
        T.ones_like(x.reshape(-1))*n for n,x in enumerate(saliency_scores)])
    cum_filters_per_layer = [0] + list(np.cumsum([x.numel() for x in saliency_scores]))
    model_layers = list(x.weight for x in iter_conv2d(model))
    # zero out XX percent of least salient filters using `saliency_scores`
    chunk_start, chunk_end = 0, int((pct/100)*len(tmp))
    reinitialize_these = idxs[chunk_start:chunk_end]
    for global_filter_idx in reinitialize_these:
        layer_idx = layer_idxs[global_filter_idx].int().item()
        within_layer_idx = global_filter_idx.item() - cum_filters_per_layer[layer_idx]
        o, i = np.unravel_index(within_layer_idx, saliency_scores[layer_idx].shape[:2])
        # --> actually update this spatial filter in the model
        with T.no_grad():
            model_layers[layer_idx].data[o,i] = 0

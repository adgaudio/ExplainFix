"""
correlate the spectra from gradients to spectra of filters.
"""
import re
import os
import pandas as pd
from collections import defaultdict
from typing import Optional
from itertools import product
import math
import torch as T
import numpy as np
from matplotlib import pyplot as plt
#  from simplepytorch.metrics import distance_correlation
from dw2.models import iter_conv2d
from dw2.kernel import dct_basis_nd
from dw2.configs.fixed_filters import load_cfg_from_checkpoint
from dw2.datasets.dsets_for_fixed_filters_paper import get_datasets_and_loaders

plt.rcParams.update({'font.size': 20,
                     'legend.title_fontsize': 16,
                     'legend.fontsize': 15,
                     'axes.labelsize': 20,
                     'lines.markersize': 18,
                     "text.usetex": True,
                     })



def dravel(M, edges_first=True):
    """
    Diagonal Traversal of a matrix, in lines parallel to top right to bottom
    left.  Optionally, sample the edges of the diagonal lines before the
    middles in stable order.  In this optional case, the matrix must be square.

    Outputs a vector of values of M in this diagonal scan order.

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


def get_spectra(F:T.Tensor, w: T.Tensor, shape:tuple[int], device:str) -> tuple[T.Tensor,T.Tensor,Optional[T.Tensor]]:
    """Compute e^d, and then use back-projection to get e^1 and e^0.  For e^0, assume kernel shape is square.

    :F: a NxM matrix where each row is a flattened spatial convolution filter kernel of some shape, in any dimension.
    :w: (N, ) or (1,N) tensor of saliency weights for each spatial filter
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


dct_bases = defaultdict(
    lambda n,m: T.from_numpy(dct_basis_nd((n,m)).reshape(n*m,n*m)).float())


class GetGrads:
    """Get the gradients w.r.t. to pre-activation and activation
    of each given layer.  Assume each module only has a single input and output."""
    def __init__(self, modules:list[T.nn.Module]):
        self.inputs = []
        self.outputs = []
        self._handles = [
            module.register_full_backward_hook(self._receive_hook)
            for module in modules]

    def _receive_hook(self, module, grad_input, grad_output):
        assert len(grad_input) == 1
        assert len(grad_output) == 1
        if grad_input[0] is None:
            return
        with T.no_grad():
            self.inputs.append(grad_input[0])
            self.outputs.append(grad_output[0])

    def remove_handles(self):
        [x.remove() for x in self._handles]
        del self._handles[:]

    def clear(self):
        del self.inputs[:], self.outputs[:]


def get_spectra_from_grads_2d(model:T.nn.Module, loader: T.utils.data.DataLoader, device:str,
                              num_minibatches:int=float('inf')):
    """Get e^d, e^1 and e^0 from gradients of 2d convolution filters."""
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
    getgrads = GetGrads(iter_conv2d(model, include_spatial=True, include_1x1=False))
    for n, (x,y) in enumerate(loader):
        if n >= num_minibatches:
            break
        batch_size = x.shape[0]
        N += batch_size
        yhat = model(x)
        # rescale yhat to all ones or zeros.
        with T.no_grad():
            yhat /= yhat
            #  yhat[yhat != 0] /= yhat[yhat!=0]

        # get gradients, making all predictions for correct classes correct
        #  (y*yhat).sum().backward(retain_graph=False)
        grads_all_layers = T.autograd.grad(  # might want to scale classes so we backprop only ones and zeros. or use deep taylor decomposition or something.  would be equivalent to scaling the gradients directly.
            ((y)*(yhat)).sum(), filters_all_layers, retain_graph=False)
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
        getgrads.clear()
    e_2, e_1, e_0 = spectra.lists
    return e_2, e_1, e_0


def ragged_to_matrix(ragged_vectors:list[T.Tensor]):

    """Convert list of ragged 1-d vectors by padding zeros to the right
    for any vector that is too small.

    :ragged_vectors: list of N 1-d vectors of varying length.

    :return: (N,M) matrix where M is max length of all vectors"""
    N = max(len(x) for x in ragged_vectors)
    T.nn.functional.pad(T.tensor([1]), [0,1])
    M = T.stack([T.nn.functional.pad(x, [0, N-len(x)])
                 for x in ragged_vectors])
    return M


def get_spectra_from_model(model):
    spectra_e2, spectra_e1, spectra_e0 = [],[],[]
    for conv2d in iter_conv2d(model, include_spatial=True, include_1x1=False):
        O,I,H,W = conv2d.weight.shape
        saliency_weight = T.ones(O*I, dtype=T.float) / (O*I)
        e_d,e_1,e_0 = get_spectra(conv2d.weight.reshape(O*I, H*W).detach(),
                                  saliency_weight, (H,W), device)
        spectra_e2.append(e_d)
        spectra_e1.append(e_1)
        spectra_e0.append(e_0)
    return spectra_e2, spectra_e1, spectra_e0


def plot_spectrum_ed(ed, fig, loglog=False):
    """
    :ed: list of the e^d energy spectra, with one spectrum for each spatial
        convolution layer of the model.
    :fig: A figure (or subfigure) in which to generate the visuals.

    """
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


def plot_ed_with_and_without_saliency(model, data_loader, device, figsize):
    """Compare the e^d spectrum of weights of model to spectrum of the gradients
    on a sample of images
    """
    # --> get spectra using saliency weights
    e2, e1, e0 = get_spectra_from_grads_2d(model, data_loader, device, num_minibatches=15)
    # --> get spectra without saliency weights
    e2f, e1f, e0f = get_spectra_from_model(model)
    # --> HACK / Workaround for DenseNet:  skip the 7x7 filter at beginning since
    #  it has 49 bases and makes visualization funky
    print(e2[0].shape)
    if e2[0].shape[0] == 49:
        e2 = e2[1:]
        e2f = e2f[1:]

    # --> visualize
    #  fig, axs = plt.subplots(1,2, figsize=figsize, sharex=True, sharey=True)
    #  axs = axs.ravel()
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    subfigs = fig.subfigures(1,2, wspace=.07)
    # plot spectra (img)
    subfigs[0].suptitle('$\mathbf{e}^{(2)}$ with Saliency')
    subfigs[1].suptitle('$\mathbf{e}^{(2)}$ without Saliency')
    # bar plots of total energy across layers, ax3 and ax4
    for ed, _sfig in zip((e2, e2f), subfigs):
        # main plot
        plot_spectrum_ed(ed, _sfig, loglog=False)
    return fig, e2, e1, e0


def plot_e1_e0(e1_e0_data):
    e1_e0_data = e1_e0

    #e1
    df = pd.DataFrame(e1_e0_data)
    df2 = pd.DataFrame(
        df['e1'].apply(lambda xx: sum(x for x in xx if len(x) == 3+3)
                       ).tolist(),
        columns=pd.Index([r'$e_%s^{(1)}$'%(i) for _ in [0,1] for i in range(3)],
                         name=r'Energy Spectrum $\mathbf{e}^{(1)}$'),
        index=df.set_index(['model_name', 'fixed_filter_method']).index)
    df2 = df2.T
    df2 = (df2 - df2.min(0)) / (df2.max(0) - df2.min(0))
    df2 = df2.T
    #e0
    mask = df['model_name'].str.contains('EfficientNet')
    df = df.copy()
    df3_3 = df.loc[~mask, 'e0'].apply(lambda xx: sum(x for x in xx if len(x) == 3))
    df3_5 = df.loc[mask, 'e0'].apply(lambda xx: sum(x for x in xx if len(x) == 5))
    df3_3 = pd.DataFrame(df3_3.tolist(),
        columns=pd.Index([r'$e_%s^{(0)}$ %s'%(i,n) for i,n in zip(range(3), ['Low Freq', 'Med Freq', 'High Freq'])],
                         name=r'Energy Spectrum $\mathbf{e}^{(0)}$'),
        index=df.loc[~mask].set_index(['model_name', 'fixed_filter_method']).index)
    df3_5 = pd.DataFrame(df3_5.tolist(),
        columns=pd.Index([r'$e_%s^{(0)}$ %s'%(i,n) for i,n in zip(range(5), ['Low Freq', 'Med-Low Freq', 'Med Freq', 'Med-High Freq', 'High Freq'])],
                         name=r'Energy Spectrum $\mathbf{e}^{(0)}$'),
        index=df.loc[mask].set_index(['model_name', 'fixed_filter_method']).index)
    df3_3 = df3_3.T
    df3_3 = (df3_3 - df3_3.min(0)) / (df3_3.max(0) - df3_3.min(0))
    df3_3 = df3_3.T
    df3_5 = df3_5.T
    df3_5 = (df3_5 - df3_5.min(0)) / (df3_5.max(0) - df3_5.min(0))
    df3_5 = df3_5.T

    # plot fully learned baselines of pretrained models
    fig_e1, ax0 = plt.subplots(figsize=(6,4))
    fig_e0_3, ax1 = plt.subplots(figsize=(3,4))
    fig_e0_5, ax2 = plt.subplots(figsize=(3,4))
    for df23, d, k, ax in [(df2, 1, 3, ax0), (df3_3, 0, 3, ax1), (df3_5, 0, 5, ax2)]:
        z = df23.xs('Fully Learned Baseline', level=1).sort_index()
        z.plot.barh(stacked=True, ax=ax, color=np.vstack([
                plt.cm.RdYlGn_r(np.linspace(0, 1, k)),
                plt.cm.RdYlGn_r(np.linspace(0, 1, k))]))
        ax.set_ylabel(None)
        ax.set_xlabel('Normalized energies')
        ax.set_title(r'$%s \times %s$ Kernels' % (k,k))
        ax.legend(
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    fig_e0_3.tight_layout()# ; fig_e0_3.subplots_adjust(top=.8)
    fig_e0_5.tight_layout()# ; fig_e0_5.subplots_adjust(top=.8)
    for fig, d, k in [(fig_e1, 1, 3), (fig_e0_3, 0, 3), (fig_e0_5, 0, 5)]:
        #  fig.suptitle(r'$\mathbf{e}^{(%s)}$ Spectra of Fully Learned Baselines' % d)
        fig.subplots_adjust(left=.6, right=.8, bottom=.21)
        fig.savefig(f'explainsteer_plots/e{d}_{k}_baselines.png', bbox_inches='tight', pad_inches=.01)

    # plot all methods for a given model
    #  model_names = df2.index.levels[0]
    #  for name in model_names:
    #      fig, ax = plt.subplots(clear=True)
    #      df2.loc[name].reindex([
    #          'Fully Learned Baseline', 'Unchanged', 'GuidedSteer', 'GHaar', 'Psine', 'DCT2'
    #      ]).plot.barh(ax=ax, color=np.vstack([
    #          plt.cm.Blues(np.linspace(.5, 1, 3)),
    #          plt.cm.Purples(np.linspace(.5, 1, 3))]))
    #      ax.set_ylabel(None)
    #      ax.set_xlabel('Normalized energies')
    #      fig.suptitle(r'$\mathbf{e}^{(1)}$ Spectrum, ' + name)
    #      ax.legend(
    #            bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    #      fig.subplots_adjust(left=.48, right=.8)
    #      fig.savefig(f'explainsteer_plots/e1_{name}.png', bbox_inches='tight')
    #      plt.close(fig)


if __name__ == "__main__":
    def get_test_perf(dir):
        """Find the avg test set performance for a given model"""
        fp = f'{dir}/eval_CheXpert_Small_L_valid.csv'
        with open(fp, 'r') as fin:
            scores = [float(x.split(',')[1]) for x in fin.readlines() if 'test_roc_auc_MEAN' in x]
            if len(scores) > 1:
                print('WARNING', fp, len(scores), '   --    ', scores)
            z = np.mean(scores)
        return z


    def load_model(dir, device):
        # TODO: remove hack shortcut
        #  _, dir = ('baseline (densenet fromscratch)', 'results/C8-densenet121:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('baseline (efficientnet fromscratch)', 'results/C8-efficientnet-b0:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('baseline (efficientnet pretrained)', 'results/C8-efficientnet-b0:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        model = load_cfg_from_checkpoint(dir, 'epoch_40.pth', load_random_state=False, device=device).model
        return model
        #  model_name, dir = ('Psine (densenet)', 'results/C8-densenet121:5:fromscratch-spatial_100%_psine-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('GHaar (densenet)', 'results/C8-densenet121:5:fromscratch-spatial_100%_ghaarA-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('baseline (densenet pretrained)', 'results/C8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('ImageNet filters (densenet)', 'results/C8-densenet121:5:pretrained-spatial_100%_unchanged-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('Random filters (densenet)', 'results/C8-densenet121:5:fromscratch-spatial_100%_unchanged-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('DCT2 (densenet)', 'results/C8-densenet121:5:fromscratch-spatial_100%_DCT2-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')

        #  model_name, dir = ('baseline (efficientnet fromscratch)', 'results/C8-efficientnet-b0:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('baseline (efficientnet pretrained)', 'results/C8-efficientnet-b0:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('ImageNet filters (efficientnet)', 'results/C8-efficientnet-b0:5:pretrained-spatial_100%_unchanged-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('DCT2 (efficientnet)', 'results/C8-efficientnet-b0:5:fromscratch-spatial_100%_DCT2-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('Random filters (efficientnet)', 'results/C8-efficientnet-b0:5:fromscratch-spatial_100%_unchanged-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('baseline (resnet fromscratch)', 'results/C8-resnet50:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('baseline (resnet pretrained)', 'results/C8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model_name, dir = ('ImageNet filters (resnet)', 'results/C8-resnet50:5:pretrained-spatial_100%_unchanged-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
        #  model = get_model('densenet121:5:fromscratch', 'unmodified_baseline', device)
        #  return model


    def prepare_chexpert_dataloader(loader, device):
        for x,y in loader:
            y = y.clone()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # CheXpert specific things:
            y[y == 3] = 0  # remap missing values to negative
            y[y == 2] = 0  # ignore uncertain values
            if y.sum() == 0:
                continue
            yield x,y


    import glob

    device = 'cuda:0'
    #  device = 'cpu'
    dset_name = 'CheXpert_Small_L_150k'
    #  dset_name = 'CheXpert_Small_L_debug'
    dset = get_datasets_and_loaders(dset_name, 'none')
    loader = prepare_chexpert_dataloader(dset['train_loader'], device)

    dirs = glob.glob('results/C8-*')
    e1_e0 = []
    os.makedirs('explainsteer_plots', exist_ok=True)
    for dir in dirs:
        tmp = re.match(r'results/C8-(.*?:5:(pretrained|fromscratch))-(spatial_100%_)?(unmodified_baseline|.*?)-.*', dir)

        # helpful HACKs for testing:
        #  if 'fromscratch' not in dir: continue
        #  if 'densen' not in dir: continue
        if 'unmodified' not in dir: continue
        #  if 'pretrained' not in dir: continue
        #  if 'unchanged' not in dir: continue
        #  if 'ghaarA' not in dir: continue

        if not any(x in dir for x in {'ghaarA', 'psine', 'unmodified', 'unch', 'GuidedSteer-', 'DCT2-'}):
            continue
        if tmp is None:
            print('SKIP', dir)
            continue

        print(dir)
        model_name = tmp.group(1).split(':')[0].replace('resnet', 'ResNet').replace('densenet', 'DenseNet').replace('efficientnet', 'EfficientNet')
        fixed_filter_method = tmp.group(4).replace('ghaarA', 'GHaar')
        for k,v in {'ghaarA': 'GHaar', 'psine': 'Psine', 'unchanged': 'Unchanged',
                    'unmodified_baseline': 'Fully Learned Baseline' }.items():
            fixed_filter_method = fixed_filter_method.replace(k,v)
        is_pretrained = tmp.group(2)
        try:
            model = load_model(dir, device)
        except FileNotFoundError:
            print("\n\nSKIP, no pre-trained model found!!\n", dir, '\n\n')
            continue
        if 'DenseNet' in model_name:
            figsize = (12,3)
        elif 'ResNet' in model_name:
            figsize = (12,4)
        elif 'EfficientNet' in model_name:
            figsize = (12,8)
        fig, e2, e1, e0 = plot_ed_with_and_without_saliency(model, loader, device, figsize)
        if 'DenseNet' in model_name and 'GHaar' in fixed_filter_method and is_pretrained == 'fromscratch':
            _fig = plt.figure(figsize=(12,4))
            plot_spectrum_ed(e2, _fig, loglog=False)
            #  _fig.suptitle(f'{model_name}, Initialization: {fixed_filter_method}')
            _fig.savefig(f'explainsteer_plots/ed_onlywith_saliency__{model_name}_{is_pretrained}_{fixed_filter_method.replace("Fully Learned Baseline", "Baseline")}.png',
                         bbox_inches='tight', pad_inches=0.02)
        e1_e0.append({
            'e1': [x.cpu().numpy() for x in e1],
            'e0': [x.cpu().numpy() for x in e0],
            'model_name': f'{model_name}:{is_pretrained}',
            'fixed_filter_method': fixed_filter_method})
        fig.suptitle(f'{model_name}, Initialization: {fixed_filter_method}')
        #  fig.subplots_adjust(wspace=.1, hspace=.0, left=.05, right=.995, top=.95,bottom=0)
        fig.savefig(f'explainsteer_plots/ed_with_without_saliency__{model_name}_{is_pretrained}_{fixed_filter_method.replace("Fully Learned Baseline", "Baseline")}.png',
                    bbox_inches='tight', pad_inches=0.02)

    plot_e1_e0(e1_e0)



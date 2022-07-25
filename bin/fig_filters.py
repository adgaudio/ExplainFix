from explainfix import ghaar2d, kernel
import math
import matplotlib.pyplot as plt
import torch as T
import numpy as np
from dw.plotting import plot_img_grid
from matplotlib import rcParams
rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'lines.linewidth': 3,
    'lines.markersize': 10,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})


def norm01(x):
    return (x-x.min()) / (x.max()-x.min())


def sumto1(x):
    return x/x.sum()


def make_ghaar():
    ch = 5
    kernel_size = 3
    weights = np.random.rand(ch, 3)*2-1
    freqs = np.linspace(1/ch, 2*(kernel_size-1)-1/ch, ch)
    filters_3x3 = [
        ghaar2d(3, *hvd, f=f, norm=False)
        for (hvd, f) in zip(weights, freqs)]
    filters_100x100 = [
        ghaar2d(100, *hvd, f=f, norm=False)
        for (hvd, f) in zip(weights, freqs)]
    fig = plot_img_grid(filters_3x3 + filters_100x100, rows_cols=(2,-1,),
                        norm=plt.cm.colors.CenteredNorm(0), cmap='PuOr_r')
    for ax, f in zip(fig.axes, freqs):
        ax.set_title(f'$f={f:.02g}$')
    for ax, f in zip(fig.axes[len(freqs):], freqs):
        ax.set_title(f' ')
    fig.subplots_adjust()
    fig.savefig('ghaar.png')


def make_guidedsteer():
    dct = T.load('svdsteering_noncentered/densenet121.pth', map_location='cpu')
    layer_idx = 1
    dct = dct['layer_params'][layer_idx]

    O, I, H, W = 10, 1, 3, 3
    a,b = 0,dct['Bu'].shape[0]

    filters = (T.from_numpy(dct['kde'].resample(O*I).T).float().to('cpu')[:,a:b] @ dct['Bu'][a:b] + dct['Fmean']).reshape(O,I,H,W)
    fig = plot_img_grid(filters.reshape(O, 3, 3), rows_cols=(2, -1), norm=plt.cm.colors.CenteredNorm(0), cmap='PuOr_r')
    fig.savefig('guidedsteer.png', bbox_inches='tight')
                

def make_psine():
    ch = 5
    _, N, M = ..., 2, ...
    device='cpu'
    make_params = lambda L, fmax: dict(
        f=T.rand((L, N), device=device)*fmax,
        t=T.rand((L, N), device=device) * math.pi*2,
        p=T.rand((L,), device=device)*((L-1)/2-1)+1,
        w=sumto1(T.rand((L, ), device=device))*2-1,
        # M=M,
    )
    _L = (2**(6*T.rand((ch,))))  # just sample in log space from 0 to 64, roughly.
    params = [make_params(L=int(3+L), fmax=np.random.uniform(1, 5)) for L in _L]
    filters = [kernel.polynomial_sin_ND(M=3, **pdict)  for pdict in params]
    filters += [kernel.polynomial_sin_ND(M=100, **pdict) for pdict in params]

    filters = [norm01(x)*2-1 for x in filters]
    # filters = [ x/(((x**2).sum())**.5) for x in filters]
    fig = plot_img_grid(filters, rows_cols=(2, -1), norm=plt.cm.colors.CenteredNorm(0), cmap='PuOr_r')
    fig.savefig('psine.png', bbox_inches='tight')

def make_unchanged():
    import torchvision.models as tvm
    mod = tvm.densenet121(pretrained=False)
    filters = mod.features.denseblock1.denselayer1.conv2.weight[np.random.randint(0, 32, 5), np.random.randint(0, 128, 5)].detach() 
    fig = plot_img_grid(filters, norm=plt.cm.colors.CenteredNorm(0), cmap='PuOr_r', rows_cols=(1, -1))
    fig.savefig('unchanged_random.png', bbox_inches='tight')

    mod = tvm.densenet121(pretrained=True)
    filters = mod.features.denseblock1.denselayer1.conv2.weight[np.random.randint(0, 32, 5), np.random.randint(0, 128, 5)].detach()
    fig = plot_img_grid(filters, norm=plt.cm.colors.CenteredNorm(0), cmap='PuOr_r', rows_cols=(1, -1))
    fig.savefig('unchanged_imagenet.png', bbox_inches='tight')


make_ghaar()
make_guidedsteer()
make_psine()
make_unchanged()

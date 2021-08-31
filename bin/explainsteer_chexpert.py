"""
correlate the spectra from gradients to spectra of filters.
"""
import re
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#  from simplepytorch.metrics import distance_correlation
from dw2.configs.fixed_filters import load_cfg_from_checkpoint
from dw2.datasets.dsets_for_fixed_filters_paper import get_datasets_and_loaders
from dw2.explainsteer import (
    explainsteer_layerwise_with_saliency,
    explainsteer_layerwise_without_saliency,
    plot_spectrum_ed
)

plt.rcParams.update({'font.size': 20,
                     'legend.title_fontsize': 16,
                     'legend.fontsize': 15,
                     'axes.labelsize': 20,
                     'lines.markersize': 18,
                     "text.usetex": True,
                     })



def plot_ed_with_and_without_saliency(model, data_loader, device, figsize):
    """Compare the e^d spectrum of weights of model to spectrum of the gradients
    on a sample of images
    """
    # --> get spectra using saliency weights
    e2, e1, e0 = explainsteer_layerwise_with_saliency(model, data_loader, device, num_minibatches=15)
    # --> get spectra without saliency weights
    e2f, e1f, e0f = explainsteer_layerwise_without_saliency(model)
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

    # e1
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
    # e0
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

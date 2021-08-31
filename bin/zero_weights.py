import torch as T
import numpy as np
import re
from os import path
import gc
import pandas as pd
from dw2 import trainlib as TL
from dw2.models import iter_conv2d, prune_model, convert_all_spatial_conv2d
from dw2.configs.fixed_filters import load_cfg_from_checkpoint
import dw2.models.models_for_fixed_filters_paper as M
from dw2.datasets.dsets_for_fixed_filters_paper import get_datasets_and_loaders
from explainsteer_chexpert import GetGrads, SumList, get_spectra


def get_spectra_from_grads_2d(model:T.nn.Module, loader: T.utils.data.DataLoader, device:str,
                              num_minibatches:int=float('inf')):
    """Get e^d, e^1 and e^0 from gradients dy/dw of 2d convolution filters."""
    model.eval()
    # --> list of spatial filters in model
    filters_all_layers = [
        x.weight for x in iter_conv2d(
            model, include_spatial=True, include_1x1=False)]
    [x.requires_grad_(True) for x in filters_all_layers]
    # obtain spectra using gradients dy/dw as weights
    # --> get gradients
    spectra = SumList()
    N = 0
    getgrads = GetGrads(iter_conv2d(model, include_spatial=True, include_1x1=False))
    for n, (x,y) in enumerate(loader):
        if n >= num_minibatches:
            break
        batch_size = x.shape[0]
        N += batch_size
        yhat = model(x)
        #  with T.no_grad():
            #  yhat /= yhat
        grads_all_layers = T.autograd.grad(  # might want to scale classes so we backprop only ones and zeros. or use deep taylor decomposition or something.  would be equivalent to scaling the gradients directly.
            (y*yhat).sum(), filters_all_layers, retain_graph=False)
        # --> get spectra of filters, weighted by gradients
        # --> (basically input * mag of gradient saliency)
        with T.no_grad():
            w, spectra_e2, spectra_e1, spectra_e0 = [], [], [], []
            for filters, grads_filters in zip(filters_all_layers, grads_all_layers,):
                O,I,H,W = filters.shape
                saliency_weight = grads_filters.abs().sum((-1,-2), keepdims=True)
                #  saliency_weight = (filters*grads_filters).abs().sum((-1,-2), keepdims=True)
                e_d, e_1, e_0 = get_spectra(
                    filters.reshape(O*I,H*W), saliency_weight.reshape(1, O*I),
                    (H,W), device)
                spectra_e2.append(e_d)
                spectra_e1.append(e_1)
                spectra_e0.append(e_0)
                w.append(saliency_weight)
            spectra += (spectra_e2, spectra_e1, spectra_e0, w)
        getgrads.clear()
    e_2, e_1, e_0, w = spectra.lists
    return e_2, e_1, e_0, w


def eval_zero_weights(scores, cfg, remove_most_salient_first:bool):
    """Evaluate effects of progressively removing weights in the network"""
    tmp = T.hstack([x.reshape(-1) for x in scores])
    #
    idxs = tmp.argsort(descending=remove_most_salient_first)  # side note: reversing this does kill performance, confirming that scores is meaningful.
    #
    # modify model and evaluate: re-initialize low scoring filters to 0, get performance
    model_layers = list(x.weight for x in iter_conv2d(cfg.model))
    perf = []
    # --> do some book keeping to find the smallest filters across all layers in model
    layer_idxs = T.hstack([T.ones_like(x.reshape(-1))*n for n,x in enumerate(scores)])
    cum_filters_per_layer = [0] + list(np.cumsum([x.numel() for x in scores]))
    # --> iteratively zero out the filters, smallest first, and evaluate model.
    inc = .01  # remove bottom 1\% of filters per iter
    pct = -inc
    while pct < 1:
        chunk_start, chunk_end = int(max(0,pct) * len(tmp)), int((pct+inc)*len(tmp))
        pct += inc
        reinitialize_these = idxs[chunk_start:chunk_end]
        for global_filter_idx in reinitialize_these:
            layer_idx = layer_idxs[global_filter_idx].int().item()
            within_layer_idx = global_filter_idx.item() - cum_filters_per_layer[layer_idx]
            o, i = np.unravel_index(within_layer_idx, scores[layer_idx].shape[:2])
            # --> actually update this spatial filter in the model
            with T.no_grad():
                model_layers[layer_idx][o,i,:,:] = 0
        print('eval mdl', len(reinitialize_these))
        # --> evaluate
        test_loader = get_datasets_and_loaders(
            'CheXpert_Small_L_valid', 'none')['test_loader']
        perf.append({
            'Percentage Filters Zeroed Out': np.round(pct, 2)*100,
            'Num Filters Zeroed Out': chunk_end,
            'Test Set Mean ROC AUC': TL.evaluate_perf(cfg, loader=test_loader).roc_auc['roc_auc_MEAN']
            })
        print(perf[-1])
    df = pd.DataFrame(perf)
    return df


def traineval_zero_weights(scores, cfg, pct, prune:bool, model_name, prefix, n_epochs):
    """Zero out `pct` of spatial filters.  Then, train model for 5 epochs.  Get perf after every epoch.
    """
    tmp = T.hstack([x.reshape(-1).cpu() for x in scores])
    idxs = tmp.argsort()
    model_layers = list(x.weight for x in iter_conv2d(cfg.model))
    # --> do some book keeping to find the smallest filters across all layers in model
    layer_idxs = T.hstack([T.ones_like(x.reshape(-1))*n for n,x in enumerate(scores)])
    cum_filters_per_layer = [0] + list(np.cumsum([x.numel() for x in scores]))
    # --> evaluate model
    perf = []
    test_loader = get_datasets_and_loaders(
        'CheXpert_Small_L_valid', 'none')['test_loader']

    perf.append({
        'Percentage Filters Zeroed Out': 0,
        'Epoch': cfg.epochs,  # bug: should be 1 for from-scratch models.  leave this because corrected in plots and have data with it.
        'Num Filters Zeroed Out': 0,
        'Test Set Mean ROC AUC': TL.evaluate_perf(cfg, loader=test_loader).roc_auc['roc_auc_MEAN']
        })
    print(perf[-1])
    # --> zero out XX percent of least salient filters using `scores`
    chunk_start, chunk_end = 0, int((pct/100)*len(tmp))
    reinitialize_these = idxs[chunk_start:chunk_end]
    for global_filter_idx in reinitialize_these:
        layer_idx = layer_idxs[global_filter_idx].int().item()
        within_layer_idx = global_filter_idx.item() - cum_filters_per_layer[layer_idx]
        o, i = np.unravel_index(within_layer_idx, scores[layer_idx].shape[:2])
        # --> actually update this spatial filter in the model
        with T.no_grad():
            model_layers[layer_idx].data[o,i,:,:] = 0
    # --> save space
    del scores, tmp, idxs, model_layers, layer_idxs, cum_filters_per_layer, test_loader, reinitialize_these
    # --> Prune the model if requested to do so.
    if prune:
        print("PRUNING MODEL")
        a = num_weights_before_pruning = sum([x.numel() for x in cfg.model.parameters()])
        prune_model(cfg.model)
        cfg.model.to(cfg.device)
        b = num_weights_after_pruning = sum([x.numel() for x in cfg.model.parameters()])
        pct_weights_pruned = 1-b/a
        print(f"  PRUNED {a-b} ({1-b/a}\%) parameters")
        # re-initialize the pruned model so it isn't sparse
        for name, name2 in [('ghaar', 'ghaarA'),
                            ('psine', 'polynomial_sin_ND'),
                            ('dct2', 'DCT2'),
                            ('guidedsteer', 'GuidedSteer'),
                            ('imagenet', 'unchanged'),
                            ('krandom', 'kaiming_uniform'),
                            ]:
            if name in model_name.lower():
                print("  Re-initialize weights with method ", name2)
                convert_all_spatial_conv2d(cfg.model, False, name2)
                break
        # then, ensure that all sparse values are filled regardless of
        # initialization in many some cases the model performance was dropping.
        # I think sparsity from the zeroing step is the problem.  This is my
        # attempt to fix it.
        if 'baseline' not in model_name.lower():
            print("  (pruning) Extra re-initialization method: FILL ZERO")
            convert_all_spatial_conv2d(cfg.model, False, 'fill_zero')
        else:
            print("  (pruning) Not running FILL ZERO initialization on baseline")
        prune_stats = {
            'Num Weights Before Pruning': num_weights_before_pruning,
            'Num Weights After Pruning': num_weights_after_pruning,
            'Pct Weights Pruned': pct_weights_pruned
        }
        # reset optimizer
        _tmp = cfg.optimizer.defaults
        # --> double the learning rate
        #  newlr = _tmp['lr']*2
        #  print(f'double learning rate from,to:  {_tmp["lr"]}, {newlr}')
        #  _tmp['lr'] = newlr
        del cfg.optimizer
        cfg.optimizer = T.optim.Adam(cfg.model.parameters(), **_tmp)
    else:
        prune_stats = {}

    # --> fix all spatial filters of the model so no learning happens
    if 'baseline' in model_name.lower():
        print('LEARNED SPATIAL FILTERS... Not fixed!')
        [x.weight.requires_grad_(True) for x in iter_conv2d(cfg.model)]
    else:
        print('FIXED SPATIAL FILTERS')
        [x.weight.requires_grad_(False) for x in iter_conv2d(cfg.model)]
    # --> check the number of filters per layer
    print('filters per layer', [(x.weight != 0).sum().item() for x in iter_conv2d(cfg.model)])
    if any((x.weight != 0).sum()==0 for x in iter_conv2d(cfg.model)):
        print('WARNING: model has layers with zero weights!!  Will fail to forward or backpropagate.')
        print('NOT continuing with this model', model_name, 'pct', pct)
        return
    gc.collect()
    T.cuda.empty_cache()
    # --> run experiment
    for cur_epoch in range(cfg.epochs+1, cfg.epochs+1+n_epochs):
        # --> train model
        with TL.timer() as seconds:
            cfg.train_one_epoch(cfg)
        # --> evaluate
        test_loader = get_datasets_and_loaders(
            'CheXpert_Small_L_valid', 'none')['test_loader']
        perf.append({
            'Percentage Filters Zeroed Out': pct,
            'Epoch': cur_epoch,
            'Num Filters Zeroed Out': chunk_end,
            'Test Set Mean ROC AUC': TL.evaluate_perf(cfg, loader=test_loader).roc_auc['roc_auc_MEAN'],
            'seconds_training_epoch': seconds(),
            **prune_stats
            })
        print(perf[-1])
        if cur_epoch % 10 == 0:
            df = pd.DataFrame(perf)
            df.to_csv(f'zero_weights/{prefix}zero_weights_v6_pct{pct}_{"pruned" if prune else "notpruned"}_traineval_{model_name.replace(" ","_")}_epoch{cur_epoch}.csv', index=False)
            cfg.save_checkpoint(save_fp=f'zero_weights/checkpoints/{prefix}v6_{model_name.replace(" ","_")}_pct{pct}_{"pruned" if prune else "notpruned"}_epoch{cur_epoch}.pth', cfg=cfg, cur_epoch=cur_epoch, save_model_architecture=True)
    df = pd.DataFrame(perf)
    df.to_csv(f'zero_weights/{prefix}zero_weights_v6_pct{pct}_{"pruned" if prune else "notpruned"}_traineval_{model_name.replace(" ","_")}.csv', index=False)
    cfg.save_checkpoint(save_fp=f'zero_weights/checkpoints/{prefix}v6_{model_name.replace(" ","_")}_pct{pct}_{"pruned" if prune else "notpruned"}_epoch{cur_epoch}.pth', cfg=cfg, cur_epoch=cur_epoch, save_model_architecture=True)
    return df


def experiment1(device, loader, try_these, remove_most_salient_first=False):
    """Zero out weights in a trained fixed filter model and then evaluate perf"""
    for model_name, dir in try_these:
        cfg = load_cfg_from_checkpoint(dir, 'epoch_40.pth', load_random_state=False, device=device)
        e2, e1, e0, w = get_spectra_from_grads_2d(cfg.model, loader, device, num_minibatches=15)
        del e2, e1, e0
        #  E2 = ragged_to_matrix(e2)
        #
        df = eval_zero_weights(scores=w, cfg=cfg, remove_most_salient_first=remove_most_salient_first)
        #  df.to_csv(f'zero_weights/zero_weights_REVERSE_{model_name.replace(" ","_")}.csv', index=False)
        df.to_csv(f'zero_weights/zero_weights_{"REVERSE_" if remove_most_salient_first else ""}{model_name.replace(" ","_")}.csv', index=False)
        fig = df.plot.scatter('Percentage Filters Zeroed Out', 'Test Set Mean ROC AUC').figure
    del cfg, df, fig, w


def experiment2(device, loader, try_these, pcts, prune:bool, prefix:str, n_epochs:int):
    """Zero out weights in a random network, then train and evaluate perf"""
    for model_name, dir in try_these:
        for pct in pcts:
            if 'Z8' in dir:
                checkpoint_fn = 'epoch_0.pth'
            else:
                checkpoint_fn = 'epoch_40.pth'
            cfg = load_cfg_from_checkpoint(dir, checkpoint_fn, load_random_state=False, device=device)
            e2, e1, e0, w = get_spectra_from_grads_2d(cfg.model, loader, device, num_minibatches=50)
            del e2, e1, e0, cfg
            w = [x.cpu() for x in w]
            # --> reload cfg because getting grads called requires_grad_ on all spatial weights and otherwise there is a memory leak.  don't fully understand why
            cfg = load_cfg_from_checkpoint(dir, checkpoint_fn, load_random_state=False, device=device)
            # ... workarounds for the random fixed models and their baselines to set config for a completely untrained model but still use all other settings.
            is_random_fixed_baseline = model_name.startswith('baseline') and model_name.endswith('fromscratch)')
            if model_name.startswith('Random Fixed') or is_random_fixed_baseline:
                if 'efficientnet' in model_name:
                    name = 'efficientnet-b0:5:fromscratch'
                elif 'densenet' in model_name:
                    name = 'densenet121:5:fromscratch'
                elif 'resnet' in model_name:
                    name = 'resnet50:5:fromscratch'
                print("NOTE: Random Fixed model.  initializing config with a random untrained model", name)
                _tmp = cfg.optimizer.defaults
                del cfg.model
                del cfg.optimizer
                cfg.start_epoch = 1
                cfg.model = M.get_model(name=name, mode='unmodified_baseline', device=cfg.device)
                cfg.optimizer = T.optim.Adam(cfg.model.parameters(), **_tmp)
                # note: model gets fixed appropriately in traineval_zero_weights.
            # --> get and save results
            traineval_zero_weights(scores=w, cfg=cfg, pct=pct, prune=prune, model_name=model_name, prefix=prefix, n_epochs=n_epochs)


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


def main(ns):
    print(ns)
    device = 'cuda:0'
    #  device = 'cpu'
    dset_name = 'CheXpert_Small_L_15k_per_epoch'
    #  dset_name = 'CheXpert_Small_L_debug'
    dset = get_datasets_and_loaders(dset_name, 'none')
    loader = prepare_chexpert_dataloader(dset['train_loader'], device)

    if ns.experiment1:
        try_these = [
            (f'{i}_baseline (densenet fromscratch)', f'results/CA8-{i}-densenet121:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
            for i in range(1, 5) ] + [
            (f'{i}_baseline (densenet pretrained)', f'results/CA8-{i}-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
            for i in range(1, 5) ] + [
            (f'{i}_baseline (resnet fromscratch)', f'results/CA8-{i}-resnet50:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
            for i in range(1, 5) ] + [
            (f'{i}_baseline (resnet pretrained)', f'results/CA8-{i}-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
            for i in range(1, 5) ] + [
            (f'{i}_baseline (efficientnet fromscratch)', f'results/CA8-{i}-efficientnet-b0:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
            for i in range(1, 5) ] + [
            (f'{i}_baseline (efficientnet pretrained)', f'results/CA8-{i}-efficientnet-b0:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1')
            for i in range(1, 5) ] + [

            ('baseline (densenet fromscratch)', 'results/C8-densenet121:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('baseline (densenet pretrained)', 'results/C8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('baseline (resnet fromscratch)', 'results/C8-resnet50:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('baseline (resnet pretrained)', 'results/C8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('baseline (efficientnet fromscratch)', 'results/C8-efficientnet-b0:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('baseline (efficientnet pretrained)', 'results/C8-efficientnet-b0:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
        ]
        try_these = [x for x in try_these if re.search(ns.experiment1, x[0])]
        assert all([path.exists(x[1]) for x in try_these])
        print(try_these)
        experiment1(device, loader, try_these, remove_most_salient_first=ns.e1_remove_most_salient_first)
    if ns.experiment2:
        try_these = [
            ('baseline (densenet fromscratch)', 'results/C8-densenet121:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('baseline (resnet fromscratch)', 'results/C8-resnet50:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('baseline (efficientnet fromscratch)', 'results/C8-efficientnet-b0:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),

            ('baseline (densenet pretrained)', 'results/C8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('baseline (resnet pretrained)', 'results/C8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('baseline (efficientnet pretrained)', 'results/C8-efficientnet-b0:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),

            #  ('FixedBaseline (densenet pretrained)', 'results/C8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            #  ('FixedBaseline (resnet pretrained)', 'results/C8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            #  ('FixedBaseline (efficientnet pretrained)', 'results/C8-efficientnet-b0:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),

            ('Unchanged (densenet pretrained)', 'results/C8-densenet121:5:pretrained-spatial_100%_unchanged-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('Unchanged (efficientnet pretrained)', 'results/C8-efficientnet-b0:5:pretrained-spatial_100%_unchanged-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('Unchanged (resnet pretrained)', 'results/C8-resnet50:5:pretrained-spatial_100%_unchanged-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),

            ('Random Fixed (densenet fromscratch)', 'results/C8-densenet121:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('Random Fixed (resnet fromscratch)', 'results/C8-resnet50:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('Random Fixed (efficientnet fromscratch)', 'results/C8-efficientnet-b0:5:fromscratch-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
        ] + [
            #  ('Psine (densenet fromscratch)', 'results/C8-densenet121:5:fromscratch-spatial_100%_psine-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            #  ('GuidedSteer (densenet fromscratch)', 'results/C8-densenet121:5:fromscratch-spatial_100%_GuidedSteer-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            #  ('DCT2 (densenet fromscratch)', 'results/C8-densenet121:5:fromscratch-spatial_100%_DCT2-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            #  ('GHaar (densenet fromscratch)', 'results/C8-densenet121:5:fromscratch-spatial_100%_ghaarA-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),

            #  ('Psine (resnet fromscratch)', 'results/C8-resnet50:5:fromscratch-spatial_100%_psine-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            #  ('GuidedSteer (resnet fromscratch)', 'results/C8-resnet50:5:fromscratch-spatial_100%_GuidedSteer-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            #  ('DCT2 (resnet fromscratch)', 'results/C8-resnet50:5:fromscratch-spatial_100%_DCT2-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            #  ('GHaar (resnet fromscratch)', 'results/C8-resnet50:5:fromscratch-spatial_100%_ghaarA-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
        ] + [  # Pruning experiments.  this baseline model architecture is never fine-tuned.  after pruning, it's re-initialized by the initialization method.
            ('Psine (densenet pretrained)', 'results/Z8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('GuidedSteer (densenet pretrained)', 'results/Z8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('DCT2 (densenet pretrained)', 'results/Z8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('GHaar (densenet pretrained)', 'results/Z8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('ImageNet (densenet pretrained)', 'results/Z8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('KRandom (densenet pretrained)', 'results/Z8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),

            ('Psine (resnet pretrained)', 'results/Z8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('GuidedSteer (resnet pretrained)', 'results/Z8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('DCT2 (resnet pretrained)', 'results/Z8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('GHaar (resnet pretrained)', 'results/Z8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('ImageNet (resnet pretrained)', 'results/Z8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('KRandom (resnet pretrained)', 'results/Z8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),

             # --> to compare pruning fixed models against pruning learned models.
            ('Pruned Baseline (densenet pretrained)', 'results/Z8-densenet121:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
            ('Pruned Baseline (resnet pretrained)', 'results/Z8-resnet50:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),

            ('ImageNet (efficientnet pretrained)', 'results/Z8-efficientnet-b0:5:pretrained-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'),
        ]

        try_these = [x for x in try_these if re.search(ns.experiment2, x[0])]
        #  experiment2(device, loader, try_these, [0, 99.9])
        # NOTE: the 0pct baseline was done with the default setting of requires_grad_(True)
        # but code is currently hardcoded to requires_grad_(False) to test the fixed filter models
        experiment2(device, loader, try_these, ns.e2_pct, ns.e2_prune, ns.savefile_prefix, ns.e2_epochs)


if __name__ == "__main__":
    import argparse as ap
    P = ap.ArgumentParser()
    P.add_argument('--experiment1', '--e1', help='regex to select models for experiment 1')
    P.add_argument('--experiment2', '--e2', help='regex to select models for experiment 2')
    P.add_argument('--e2-pct', help='percent filters to remove as list of floats.  ', type=float, nargs='+', default=[99.9])
    P.add_argument('--e1-remove-most-salient-first', help='if supplied, remove most salient filter first.  otherwise, remove least salient first.', action='store_true')
    P.add_argument('--e2-prune', action='store_true', help='If supplied, will prune the network after zeroing, before training.  Pruned networks may have sparse spatial convs.  Re-initialize any zeroed weights in the pruned model.') 
    P.add_argument('--savefile-prefix', default='', help="prefix to filename of saved csv and checkpoints")
    P.add_argument('--e2-epochs', default=80, type=int)
    main(P.parse_args())

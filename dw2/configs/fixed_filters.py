#!/usr/bin/env python
from matplotlib import pyplot as plt
from pampy import match
from textwrap import dedent
from typing import Dict, Tuple, List
import dataclasses as dc
import argparse as ap
import datetime
import numpy as np
import os.path
import pandas as pd
import sklearn.metrics
import simplepytorch.api as api
import sys
import torch as T


from dw2.datasets import dsets_for_fixed_filters_paper as D
from dw2.models import models_for_fixed_filters_paper as M
from dw2 import trainlib as TL
from dw2 import losses


@dc.dataclass
class CheXpertResult(TL.Result):
    """Report results predicting CheXpert labels for each of the diagnostic tasks.

    Consider ONLY positive and negative in the result scores.
    Ignore missing or uncertain labels.
    Assume model outputs scalar for each image for each diagnostic classes,
      where output x is classified negative if x<=0 and positive if x>0.
    """
    #  task_names = D.CheXpert.LABELS_DIAGNOSTIC
    #  task_names: Tuple[str] = tuple(D.CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD)
    task_names: Tuple[str]

    # don't adjust these at initialization
    loss: float = 0
    num_samples: int = 0
    metrics: Tuple[str] = ('loss', 'mcc', 'roc_auc', 'confusion_matrix')
    cms: Dict[str,T.Tensor] = dc.field(init=False)
    yhats: List[T.Tensor] = dc.field(default_factory=list)
    ys: List[T.Tensor] = dc.field(default_factory=list)

    def __post_init__(self):
        self.cms = {task: T.zeros(2,2, dtype=T.float)
                    for task in self.task_names}

    def update(self,
               yhat: Dict[str, T.Tensor],
               y: T.Tensor,
               loss: T.Tensor,
               ) -> None:
        """
        update results

        :yhat:  model outputs, shape (b,c)
        :y:  ground truth, shape (b,c)
        :loss:  scalar tensor
        """
        assert y.shape[1] == len(self.cms) and y.max() < 4, 'configuration error:  ground truth should be a vector of index values over the 4 possible classes (neg, pos, uncertain, missing)'

        y = y.clone()
        y[y == 3] = 0
        #  assert y.max() < 3, 'bug: at this point, the nans (missing data) in ground truth should have been re-assigned to 0 (negative)'

        with T.no_grad():  # store this for the roc curve
            self.ys.append(y.to('cpu', non_blocking=True))
            self.yhats.append(yhat.to('cpu', non_blocking=True))


        minibatch_size = y.shape[0]
        # update loss
        self.loss += loss.item()
        self.num_samples += minibatch_size

        # model final layer: just return the highest predicted class (neg or
        # pos) for each task. assume model makes no uncertainty or missing value predictions

        assert yhat.shape == (minibatch_size, len(self.task_names)), f'configuration error: model output should be a vector of {len(self.task_names)} numbers per image, predicting neg or pos outcome'
        yhat = (yhat > 0).long()

        # update confusion matrix
        assert len(self.task_names) == y.shape[1]
        assert len(self.task_names) == yhat.shape[1]
        for idx, task in enumerate(self.task_names):
            target = y[:, idx].reshape(-1)

            # --> select only images with ground truth that is not missing or uncertain
            select_some_imgs = ((target == 0) | (target == 1))
            if select_some_imgs.sum() == 0:
                continue  # this minibatch does not have sufficient ground truth for this task.
            # --> update confusion matrix for the task
            self.cms[task] = self.cms[task].to(y.device, non_blocking=True) + api.metrics.confusion_matrix(
                yhat=yhat[:,idx].reshape(-1)[select_some_imgs],
                y=target[select_some_imgs], num_classes=2)

    @property
    def confusion_matrix(self):
        return {
            f'cm_{task}': self.cms[task].cpu().numpy()
            for task in self.task_names}

    @property
    def mcc(self):
        dct = {}
        for task, cm in self.cms.items():
            dct[f'mcc_{task}'] = api.metrics.matthews_correlation_coeff(cm).item()
        return dct

    @property
    def roc_auc(self):
        if len(self.ys) == 0:  # basecase: no data yet
            return {f'roc_auc_{task}': 0 for task in self.task_names}

        ys = T.cat(self.ys, 0)
        yhats = T.cat(self.yhats, 0)
        masks = ys != 2  # ignore uncertain ground truth
        dct = {}
        for idx, task in enumerate(self.task_names):
            y_true = ys[:, idx].numpy()
            mask = masks[:, idx].numpy()
            if y_true[mask].var() == 0:  # basecase: incomplete data so far
                roc_score = 0
            else:
                y_score = yhats[:, idx].numpy()
                roc_score = sklearn.metrics.roc_auc_score(
                    y_true=y_true[mask], y_score=y_score[mask])
            dct[f'roc_auc_{task}'] = roc_score
        dct['roc_auc_MEAN'] = sum(dct.values()) / len(dct)
        return dct


def get_chexpert_labels(model_name):
    """Infer the labels we are training with from the number of classes in the model name"""
    return match(
        model_name.rsplit(':', 2)[1],
        '5', lambda _: D.CheXpert_Small.LABELS_DIAGNOSTIC_LEADERBOARD,
        '14', lambda _: D.CheXpert_Small.LABELS_DIAGNOSTIC,
    )


def train_config(args: Dict) -> TL.TrainConfig:
    """
    Generate configurations to train a variety of models on different datasets

    :args:  Dict of arguments from commandline arg parser

    Initialize PyTorch:  Model, DataLoaders, Loss, Optimizer, ...
    Return a trainlib.TrainConfig that can be used to train the model
    """
    if args['experiment_id']:
        experiment_id = args['experiment_id']
    else:
        experiment_id = '-'.join([args['model'], args['model_mode'], args['dset']])
    device = T.device(args['device'])
    model = M.get_model(name=args['model'], mode=args['model_mode'], device=device)

    # config values for training with various datasets
    if args['dset'].startswith('CheXpert'):
        assert args['dset'].startswith('CheXpert')
        assert args['model'].rsplit(':', 2)[1] in {'5', '14'}, '--model must define the num classes as 14 or 5'
        some_cfg_params = dict(
            result_factory=lambda: CheXpertResult(task_names=get_chexpert_labels(args['model'])),
            optimizer=T.optim.Adam(model.parameters(), lr=.0001)
        )

    elif args['dset'].startswith('BBBC038v1'):
        some_cfg_params = dict(
            result_factory = lambda: TL.SegmentationResult(
                model_final_layer=lambda x: (x > 0).long()),  #T.sigmoid(x).round().long()),
                optimizer = T.optim.Adam(model.parameters(), lr=.008),#, eps=1),  # DCT2 works better
                #  optimizer = T.optim.Adam(model.parameters(), lr=.02),#, eps=1),  # DCT2 doesn't work
        )
    else:
        raise NotImplementedError('Need to define some configuration in order to train on this dataset')

    _log_header = list(some_cfg_params['result_factory']().asdict('train_')) + list(some_cfg_params['result_factory']().asdict('val_'))

    cfg = TL.TrainConfig(
        experiment_id = experiment_id,
        model = model,
        #  loss_fn = T.nn.BCELoss(),  # for pytorch unet because it includes sigmoid
        #  loss_fn = T.nn.BCEWithLogitsLoss(),
        loss_fn = losses._LOSSES[args['loss']],
        device = device,
        epochs = args['epochs'],
        #  checkpoint_if = TL.CheckpointBestOrLast(
        #      'val_loss', mode='min', last_filename='latest.pth',
        #      best_filename=None,
        #  ),
        logger_factory = lambda cfg: api.MultiplexedLogger(
            api.LogRotate(api.CsvLogger)(f'{cfg.base_dir}/perf.csv', ['epoch', 'seconds_training_epoch'] + [x for x in _log_header if '_cm_' not in x]),
            api.LogRotate(api.HDFLogger)(f'{cfg.base_dir}/perf_tensors.h5', [x for x in _log_header if '_cm_' in x])),
        **some_cfg_params,
        **D.get_datasets_and_loaders(args['dset'], args['dset_augmentations_train'])
    )
    # post-processing hack:  initialize chexpert focal loss if necessary.  its parameters are dataset specific.
    if args['loss'].startswith('chexpert_focal'):
        cfg.loss_fn = losses.loss_chexpert_focal_bce(
            chexpert_labels_csv=cfg.train_dset.labels_csv, device=cfg.device,
            gamma=float(args['loss'].rsplit(':', 1)[1]),
            chexpert_task_names=get_chexpert_labels(args['model']))
    # override loss fn by adding a prior to it.
    if args['loss2']:
        _, weight = args['loss2'].split(':')
        cfg.loss_fn = losses.loss_with_prior(
            loss_fn=cfg.loss_fn, prior_fn=losses._PRIORS[args['loss2']],
            prior_attention=float(weight), model=cfg.model)

    if args['restore_checkpoint']:
        fp = args['restore_checkpoint']
        if not os.path.exists(fp):
            fp = f'{cfg.base_dir}/checkpoints/{fp}'
        TL.load_checkpoint(fp, cfg)
    return cfg


def visualize_masks(cfg, n=15, postprocess_fn=T.sigmoid, fname_prefix='img_'):
    cfg.model.eval()
    with T.no_grad():
        for i in range(n):
            x,y = cfg.val_dset[i]
            if y.ndim == 1:  # assume chexpert
                task_names = cfg.result_factory().task_names
                Nclasses = len(task_names)
                y[y>1] -= 4
                yh = postprocess_fn(cfg.model(x.unsqueeze_(0).to(cfg.device)))

                fig, axs = plt.subplots(1,3, figsize=(16,4), num=1)
                axs[0].vlines(.5, -1, Nclasses, alpha=.4)
                axs[0].hlines(np.arange(Nclasses), -2, 1, colors='gray', linestyles='dotted')
                axs[0].set_xticks([-2,-1,0,1])
                axs[0].set_xticklabels(['uncertain', 'missing', 'neg', 'pos', ])
                axs[0].set_yticks(range(0, Nclasses))
                axs[0].set_yticklabels(task_names)
                axs[0].barh(np.arange(len(y)), yh.squeeze().cpu()-.5, left=.5, alpha=.5, label='predicted probability')
                axs[0].scatter([-2]*Nclasses, range(len(y)), color='gray', alpha=.4)
                axs[0].scatter([-1]*Nclasses, range(len(y)), color='gray', alpha=.4)
                axs[0].scatter([0]*Nclasses, range(len(y)), color='gray', alpha=.4)
                axs[0].scatter([1]*Nclasses, range(len(y)), color='gray', alpha=.4)
                axs[0].scatter(y, np.arange(len(y)), label='ground truth')
                axs[0].legend(loc='upper left')
                axs[-1].axis('off')
                axs[-1].imshow(x.squeeze().cpu().numpy())
                axs[-1].set_title('x, input img')

            elif y.ndim == 3 and y.shape[0] == 1:

                fig, axs = plt.subplots(1,3, figsize=(16,4), num=1)
                [ax.axis('off') for ax in axs.ravel()]
                yh = postprocess_fn(cfg.model(x.unsqueeze_(0).to(cfg.device)))
                gt = (y.squeeze().cpu().numpy()>.5)*1.
                p = (yh.squeeze().cpu().numpy()>.5)*1.
                rgb = np.dstack([gt+p -2*gt*p, p, np.zeros_like(p)])
                axs[2].imshow(rgb, vmin=0, vmax=1)
                #  axs[3].imshow(y.squeeze().cpu().numpy(), vmin=0, vmax=1)
                #  axs[2].imshow((yh.squeeze().cpu().numpy()>.5)*1., vmin=0, vmax=1)
                axs[1].imshow(yh.squeeze().cpu().numpy(), vmin=0, vmax=1)
                axs[0].imshow(x.squeeze().permute(1,2,0).cpu().numpy())
                #  axs[3].set_title('ground truth,  y')
                #  axs[2].set_title('prediction thresholded,  f(x) > .5')
                axs[2].set_title('Ground Truth (red) vs Binarized Prediction (yellow).\nAgreement in green.')
                axs[1].set_title('Prediction Probability,  f(x)')
                axs[0].set_title('Input Img, x')

            else:
                raise NotImplementedError()
            os.makedirs(cfg.base_dir, exist_ok=True)
            fig.savefig(
                f'{cfg.base_dir}/{fname_prefix}{i}.png', bbox_inches='tight')
            plt.close(fig)
    #  plt.show()


def start_ipython(cfg):
    import IPython
    dct = dict(locals())
    globals_ = {k: v for k, v in globals().items() if not k.startswith('__')}
    dct.update(globals_)
    print("\n==== Global Namespace: ====\n{1}\n\n==== Local Namespace: ====\n{0}\n".format(
        '\n'.join([x for x in locals().keys() if not x.startswith('__')]),
        ', '.join(globals().keys())))
    IPython.start_ipython(['--no-banner', ], user_ns=dct)


def train(CFG):
    start = datetime.datetime.utcnow()
    with TL.timer() as seconds:
        TL.train(CFG)
    with open(f'{CFG.base_dir}/trained', 'w') as fout:
        fout.write(f"Start training at: {start}\n" )
        fout.write(f"Finished training at: {datetime.datetime.utcnow()}\n" )
        fout.write(f"Training wall time (total seconds, start to finish): {seconds()}\n" )


def numel(cfg):
    from dw2.models import iter_conv2d
    a = sum(x.numel() for x in cfg.model.parameters())
    b = sum(x.numel() if not x.requires_grad else 0 for x in cfg.model.parameters())
    c = sum(x.numel() if x.requires_grad else 0 for x in cfg.model.parameters())
    d = sum(x.weight.numel() for x in iter_conv2d(cfg.model, include_spatial=True, include_1x1=False))
    e = sum(x.weight.numel() for x in iter_conv2d(cfg.model, include_spatial=False, include_1x1=True))
    print('name,num parameters,num fixed,num learned,spatial conv,1x1 conv,non-conv params,')
    print(','.join(str(x) for x in (cfg.experiment_id, a, b, c, d, e, a-d-e,)))
        #  f'{b} ({b/a*100:02g}%)'))


def eval(cfg, eval_dset_name, labels):
    if eval_dset_name not in {'CheXpert_Small_L_valid', 'CheXpert_Small_D_valid'}:
        raise NotImplementedError()
    dct = D.get_datasets_and_loaders(eval_dset_name, 'none')
    # get results
    test_result_dct = TL.evaluate_perf(cfg, loader=dct['test_loader']).asdict('test_')
    out_fp = f'{cfg.base_dir}/eval_{eval_dset_name}.csv'
    # append results to csv (append enables aggregation across multiple training runs)
    ser = pd.Series(test_result_dct).sort_index()
    ser.to_csv(out_fp, mode='w', header=False)
    print(ser.to_string())

    if eval_dset_name == 'CheXpert_Small_D_valid':
        return  # don't do maxpooling because it fails on some tasks!  Leaderboard people didn't look at this dataset.

    # also evaluate by maxpooling the predictions for all views in a study.
    # --> first, compute the prediction for each image in study.
    outputs = {'yhat': [], 'y': []}
    cfg.model.eval()
    with T.no_grad():
        for X, y in dct['test_loader']:
            with T.no_grad():
                outputs['y'].append(y)
                X = X.to(cfg.device, non_blocking=True)
                outputs['yhat'].append(cfg.model(X).to('cpu', non_blocking=True))
    outputs['yhat'] = T.cat(outputs['yhat'], 0)
    outputs['y'] = T.cat(outputs['y'], 0)
    # --> next, aggregate results per study by maxpooling predictions of different views
    maxpooled = {'yhat': [], 'y': []}
    for patient, study in dct['test_dset'].labels_csv.groupby(['Patient', 'Study'])['View'].groups.keys():
        view_indexes = dct['test_dset'].get_other_views_in_study(patient=patient, study=study)['index'].values
        # assume the loader outputs images in index order.  (if it didn't performance should be obviously bad)
        if len(view_indexes) > 1:
            assert (outputs['y'][view_indexes].float().var(0) == 0).all(), 'sanity check: lateral and frontal imgs of the same study should have same ground truth'
        assert len(view_indexes) <= 3, "code bug: should be only max of three views per study"
        maxpooled['yhat'].append(outputs['yhat'][view_indexes].max(0).values)
        maxpooled['y'].append(outputs['y'][view_indexes].max(0).values)
    maxpooled['yhat'] = T.stack(maxpooled['yhat'])
    maxpooled['y'] = T.stack(maxpooled['y'])
    # --> finally, get auc roc for each test set study on each task
    out = {}
    for idx, task_name in enumerate(labels):
        y_true = maxpooled['y'][:, idx].clone()
        y_true[y_true == 3] = 0  # set the missing labels to negative
        mask = y_true != 2  # ignore uncertain labels
        out[task_name] = sklearn.metrics.roc_auc_score(
            y_true=y_true[mask], y_score=maxpooled['yhat'][:, idx][mask])
    # --> write results
    out_fp = f'{cfg.base_dir}/eval_{eval_dset_name}_maxpool.csv'
    ser = pd.Series(out).sort_index()
    ser.to_csv(out_fp, mode='a', header=False)
    print('roc auc, maxpooled across views')
    print(ser.to_string())


def main(args: Dict):
    assert not args['model_mode'].startswith('pbt_'), "run pbt_train.py for a pbt model"

    cfg = train_config(args)

    print(cfg.experiment_id, args)

    if ( ('visualize' in args['do_this'] or 'eval' in args['do_this'])
            and 'train' not in args['do_this']
            and not args['restore_checkpoint']):
        print("\n\nWARNING: to visualize a TRAINED model, you need to add --restore-checkpoint \n\n")

    for instruction in args['do_this']:
        match(instruction,
              'train', lambda _: train(cfg),
              'eval', lambda _: eval(cfg, eval_dset_name=args['eval_dset'], labels=get_chexpert_labels(args['model'])),
              'visualize', lambda _: visualize_masks(cfg),
              'shell', lambda _: start_ipython(cfg),
              'numel', lambda _: numel(cfg),
              'save_checkpoint', lambda _: cfg.save_checkpoint(save_fp=f'{cfg.base_dir}/checkpoints/epoch_{cfg.start_epoch-1}.pth', cfg=cfg, cur_epoch=cfg.start_epoch-1)
              )

    if os.path.exists(f'{cfg.base_dir}'):
        with open(f'{cfg.base_dir}/argv.txt', 'a') as fout:
            fout.write(f'{datetime.datetime.utcnow()} : ' + ' '.join(sys.argv) + '\n')


def arg_parser():
    class F(ap.ArgumentDefaultsHelpFormatter, ap.RawTextHelpFormatter): pass
    par = ap.ArgumentParser(formatter_class=F)
    par.add_argument('--experiment-id', default=None, help=(
        'optional, identifies where results are stored,'
        ' typically ./results/experiment_id'))
    par.add_argument('--model', default='unetD_small', help=dedent('''\
        Options are:
            efficientnet-b[0-7]:[NUM_CLASSES]:[pretrained|fromscratch]
            resnet[18,50,...]:[NUM_CLASSES]:[pretrained|fromscratch]
            densenet[121,...]:[NUM_CLASSES]:[pretrained|fromscratch]
        For example:
            efficientnet-b0:5:pretrained
            '''))
    par.add_argument(
        '--model-mode', default='unmodified_baseline',
        choices=M.get_model_modes(None, None)[::2], help=dedent('''\
            How to modify the model     (default: %(default)s)
              - unmodified_baseline:  do nothing to the model
              - others convert the spatial convolution layers to fixed filters.
              '''))
    par.add_argument(
        '--dset', default='BBBC038v1_split', choices=D._DATASETS.keys(), help=dedent('''\
            Specify a dataset to use     (default: %(default)s)
              '''))
    par.add_argument(
        '--restore-checkpoint', help=dedent('''\
            A filepath to a checkpoint to restore from. If filename
            does not exist on the relative path, will search in
            results/experiment_id/checkpoints/FILE_NAME
            '''))
    par.add_argument(
        '--do-this', nargs='+', choices=['train', 'visualize', 'shell', 'numel', 'eval', 'save_checkpoint'],
        default=['train', 'visualize']
    )
    par.add_argument('--epochs', default=40, type=int)
    par.add_argument(
        '--loss', choices=losses._LOSSES.keys(), default='BCEWithLogitsLoss', help="The loss function to apply")
    par.add_argument(
        '--loss2', choices=losses._PRIORS.keys(), help=(
            "Add a prior to the loss (loss + prior*weight) during training by"
            " analyzing the model.  usage: --loss2 prior_name:weight"))
    par.add_argument(
        '--dset-augmentations-train', default='none',
        help='The kind of preprocessing to apply.  Depends on choice of dataset.')
    par.add_argument('--device', default='cuda')
    par.add_argument(
        '--eval-dset', choices=D._DATASETS_TEST.keys(),
        help="a test dataset, required if '--do-this eval' is specified")

    return par


def load_cfg_from_checkpoint(base_dir, checkpoint_filename,
                             load_random_state:bool=False,
                             device=None
                             ) -> TL.TrainConfig:

    """Assume a model was trained using ./bin/train  """
    with open(f'{base_dir}/argv.txt', 'r') as fin:
        cmdline_args = [x for x in fin.readlines() if ' train ' in x and '--restore-checkpoint' not in x][-1]
    cmdline_args = cmdline_args.split(' : ', 1)[1].strip()
    cmdline_args = cmdline_args.split(' ', 1)[1]
    cmdline_args = cmdline_args.split(' ')
    # don't load the initialization method since we expect that checkpoint loads it.
    cmdline_args.extend(['--model-mode', 'unmodified_baseline'])
    if device:
        cmdline_args.extend(['--device', device])
    cfg = train_config(arg_parser().parse_args(cmdline_args).__dict__)
    if checkpoint_filename:
        assert checkpoint_filename.endswith('.pth')
        TL.load_checkpoint(
            f'{base_dir}/checkpoints/{checkpoint_filename}',
            cfg, load_random_state=load_random_state)
    return cfg

"""
A complete setup to train pytorch models in one file.
Makes use of simplepytorch.api for evaluation metrics and logging.
"""
from os.path import dirname
from os import makedirs, symlink
from termcolor import colored
from tqdm import tqdm
from typing import Callable, Union, Dict, Tuple, Optional
import abc
import contextlib2
import dataclasses as dc
import numpy as np
import random
import time
import torch as T

from simplepytorch import api


class Result(abc.ABC):
    """
    Accumulate and analyze results, for instance over the course of one epoch.

    Any values defined in metrics should have a corresponding property (or attribute).

    This is an abstract base class.
    """
    @property
    def metrics(self) -> Tuple[str]:
        raise NotImplementedError('a list of metrics available in the result')

    def update(self, yhat, y, loss) -> None:
        raise NotImplementedError()

    # don't need to modify below here

    def __str__(self):
        return ", ".join(
            f'{colored(k, "cyan", None, ["bold"])}: {v:.5f}'
            for k, v in self.asdict().items()
            if not isinstance(v, np.ndarray)
        )

    def asdict(self, prefix='') -> Dict[str, any]:
        """Fetch the metrics, and output a (flattened) dictionary"""
        tmp = {f'{prefix}{k}': getattr(self, k) for k in self.metrics}
        rv = dict(tmp)
        # flatten any results of type dict, and prepend the prefix as needed.
        for tmp_key in tmp:
            if isinstance(tmp[tmp_key], dict):
                subdct = {f'{prefix}{k}': v for k,v in rv.pop(tmp_key).items()}
                assert not set(subdct.keys()).intersection(rv), 'code error: Result has nested dictionaries with overlapping keys'
                rv.update(subdct)
        return rv


def train(cfg: 'TrainConfig') -> None:
    data_logger = cfg.logger_factory(cfg)
    for cur_epoch in range(cfg.start_epoch, cfg.epochs + 1):
        with timer() as seconds:
            train_result = cfg.train_one_epoch(cfg)

        log_data = {'epoch': cur_epoch, 'seconds_training_epoch': seconds()}
        log_data.update(train_result.asdict('train_'))
        if cfg.val_loader is not None:
            val_result = cfg.evaluate_perf(cfg)
            log_data.update(val_result.asdict('val_'))

        data_logger.writerow(log_data)
        data_logger.flush()
        # --> also log to console
        console_msg = ''.join([
            (f'{colored("epoch", "green", None, ["bold"])}: {cur_epoch: 4d}'
             f', {colored("seconds_training_epoch", "green", None, ["bold"])}:'
             f' {log_data["seconds_training_epoch"]:1g}'),
            f'\n\t{colored("TRAIN RESULTS: ", "green", None, ["bold"])}', ', '.join([
                f'{colored(k, "cyan", None, ["bold"])}: {v: .5f}' for k, v in log_data.items()
                if k.startswith('train_') and not isinstance(v, np.ndarray)]),
            f'\n\t{colored("VAL RESULTS: ", "green", None, ["bold"])}', ', '.join([
                f'  {colored(k, "yellow", None, ["bold"])}: {v: .5f}' for k, v in log_data.items()
                if k.startswith('val_') and not isinstance(v, np.ndarray)]),])
        print(console_msg)

        if fp := cfg.checkpoint_if(cfg, log_data):
            cfg.save_checkpoint(save_fp=fp, cfg=cfg, cur_epoch=cur_epoch)

        #  if early_stopping(?):
        #      log.info("Early Stopping condition activated")
        #      break
    data_logger.close()


def train_one_epoch(cfg: 'TrainConfig') -> Result:
    cfg.model.train()
    result = cfg.result_factory()
    for minibatch in tqdm(cfg.train_loader, mininterval=1):
        X = minibatch[0].to(cfg.device, non_blocking=True)
        y = minibatch[1].to(cfg.device, non_blocking=True)
        cfg.optimizer.zero_grad()
        yhat = cfg.model(X)
        loss = cfg.loss_fn(yhat, y, *minibatch[2:])
        loss.backward()
        cfg.optimizer.step()
        with T.no_grad():
            result.update(yhat=yhat, y=y, loss=loss)
    return result


def evaluate_perf(cfg: 'TrainConfig', loader=None, result_factory=None) -> Result:
    if loader is None:
        loader = cfg.val_loader
    cfg.model.eval()
    with T.no_grad():
        result = cfg.result_factory() if result_factory is None else result_factory()
        for minibatch in tqdm(loader, mininterval=1):
            X = minibatch[0].to(cfg.device, non_blocking=True)
            y = minibatch[1].to(cfg.device, non_blocking=True)
            yhat = cfg.model(X)
            loss = cfg.loss_fn(yhat, y, *minibatch[2:])
            result.update(yhat=yhat, y=y, loss=loss)
    return result


def save_checkpoint(save_fp: str, cfg: 'TrainConfig', cur_epoch: int, save_model_architecture=False) -> None:
    state = {
        'random.getstate': random.getstate(),
        'np.random.get_state': np.random.get_state(),
        'torch.get_rng_state': T.get_rng_state(),
        'torch.cuda.get_rng_state': T.cuda.get_rng_state(cfg.device),

        'cur_epoch': cur_epoch,
    }
    if save_model_architecture:
        state.update({
            'model': cfg.model,
            'optimizer': cfg.optimizer
        })
    else:
        state.update({
            'model.state_dict': cfg.model.state_dict(),
            'optimizer.state_dict': cfg.optimizer.state_dict(),
        })

    makedirs(dirname(save_fp), exist_ok=True)
    T.save(state, save_fp)
    print("Checkpoint", save_fp)


@dc.dataclass
class TrainConfig:
    model: T.nn.Module
    optimizer: T.optim.Optimizer
    train_dset: T.utils.data.Dataset
    val_dset: T.utils.data.Dataset
    train_loader: T.utils.data.DataLoader
    val_loader: Union[None,T.utils.data.DataLoader]
    loss_fn: Callable[[T.Tensor, T.Tensor], float]
    result_factory: Callable[[], Result]
    # example configurations:
    #  result_factory = lambda: SegmentationResult(
    #      classes=['tumor', 'infection', 'artifact'],
    #      model_final_layer=lambda x: (x > 0).long(),
    #      metrics=('mcc', 'loss', 'confusion_matrix'), )
    #  result_factory = lambda: ClassifierResult(
    #      model_final_layer=lambda x: (x > 0).long(),
    #      metrics=('mcc', 'loss', 'confusion_matrix'), )
    device: Union[T.device, str]
    epochs: int
    experiment_id: str = 'debugging'
    start_epoch: int = 1
    checkpoint_if: Callable[['TrainConfig'], Optional[str]] = lambda cfg, log_data: f'{cfg.base_dir}/checkpoints/epoch_{cfg.epochs}.pth' if log_data['epoch'] == cfg.epochs else None
    # example checkpoint configurations:
    #  checkpoint_if = lambda cfg, log_data: None  # always disabled
    #  checkpoint_if = CheckpointBestOrLast(metric='val_acc', mode='max')  # checkpoint the best model historically and/or also final model
    logger_factory: Callable[['TrainConfig'], api.DataLogger] = lambda cfg: api.DoNothingLogger()
    # example logger configurations.  Note: choice of what to log depends on metrics available in result_factory.
    #  logger_factory = \
    #      lambda cfg: api.LogRotate(api.CsvLogger)(
    #          f'{cfg.base_dir}/perf.csv',
    #          ['epoch', 'seconds_training_epoch'] + ['{prefix}{k}' for k in cfg.result_factory.metrics for prefix in ['train_', 'val_']])
    #  logger_factory = lambda cfg: api.MultiplexedLogger(
    #      api.LogRotate(api.CsvLogger)(f'{cfg.base_dir}/perf.csv', ['epoch', 'seconds_training_epoch', 'train_loss', 'val_loss', ...]),
    #      api.LogRotate(api.HDFLogger)(f'{cfg.base_dir}/perf_tensors.h5', ['train_confusion_matrix', 'val_confusion_matrix']))

    train_one_epoch: Callable[['TrainConfig'], Result] = train_one_epoch
    evaluate_perf: Callable[['TrainConfig'], Result] = evaluate_perf
    save_checkpoint: Callable[[str, 'TrainConfig', int], None] = save_checkpoint
    train: Callable[['TrainConfig'], None] = train

    # stuff you probably don't want to configure

    @property
    def base_dir(self):
        return f'./results/{self.experiment_id}'


@contextlib2.contextmanager
def timer():
    """Example:
        >>> with timer() as seconds:
            do_something(...)
        >>> print('elapsed time', seconds())
    """
    _seconds = []

    class seconds():
        def __new__(self):
            return _seconds[0]
    _tic = time.perf_counter()
    yield seconds
    _toc = time.perf_counter()
    _seconds.append(_toc - _tic)


class IsNextValueMonotonic:
    def __init__(self, mode='max'):
        self.mode = mode
        if mode == 'max':
            self._best_score = float('-inf')
            self._is_better = lambda x: x > self._best_score
        elif mode == 'min':
            self._best_score = float('inf')
            self._is_better = lambda x: x < self._best_score
        else:
            raise ValueError("Input mode is either 'min' or 'max'")

    def __call__(self, metric_value) -> bool:
        if self._is_better(metric_value):
            self._best_score = metric_value
            return True
        return False

class CheckpointBestOrLast:
    """
    Return a filepath to save checkpoints if:
        a) the model is best performing so far
        b) the current epoch equals the last epoch that should be trained

    This should be called at the end of an epoch during training.

        >>> fn = CheckpointBestOrLast('val_loss', mode='min',
                best_filename='best.pth',  # if not None, save checkpoint every time we get best loss.
                last_filename='epoch_{epoch}.pth',  # if not None, save checkpoint if the current epoch equals configured last epoch.
                )
        >>> fn(cfg, {'val_loss': 14.2})  # returns a filepath if model should be checkpointed
    """
    def __init__(self, metric: str, mode='max',
                 best_filename: Optional[str] = 'best.pth',
                 last_filename: Optional[str] = 'epoch_{epoch:03g}.pth'):
        """
        `metric` the name of a metric that will be available in log_data
        `mode` either 'max' or 'min' to maximize or minimize the metric value
        `best_filename` the filename (not filepath) where to save the
            checkpoint for the best performing model so far
        `last_filename` the filename (not filepath) where to save the
            checkpoint for the final state of model at end of training
            (not compatible with early stopping)

        The filenames, respectively, can be assigned None if you don't want to
        save a checkpoint for the best or last epoch.
        """
        self.metric = metric
        self.last_filename = last_filename
        self.best_filename = best_filename
        if best_filename is None:
            self._is_best_score_yet = lambda _: False
        else:
            self._is_best_score_yet = IsNextValueMonotonic(mode)

    def __call__(self, cfg: TrainConfig, log_data: Dict) -> str:
        """Return a filepath if a checkpoint should be saved"""
        is_best = self.best_filename is not None and self._is_best_score_yet(log_data[self.metric])
        is_last = self.last_filename is not None and log_data['epoch'] == cfg.epochs
        best_fp = (f'{{cfg.base_dir}}/checkpoints/{self.best_filename}').format(cfg=cfg, **log_data)
        last_fp = (f'{{cfg.base_dir}}/checkpoints/{self.last_filename}').format(cfg=cfg, **log_data)
        if is_last and is_best:
            makedirs(dirname(last_fp), exist_ok=True)
            symlink(best_fp, last_fp)
            return best_fp
        elif is_best:
            return best_fp
        elif is_last:
            return last_fp
        else:
            return


def load_checkpoint(fp: str, cfg: TrainConfig, load_random_state=True):
    """Update the model and optimizer in the given cfg.
    It's a mutable operation.  To make this point clear, don't return anything.
    """
    print('restoring from checkpoint', fp)
    S = T.load(fp, map_location=cfg.device)
    # random state and seeds
    if load_random_state:
        random.setstate(S['random.getstate'])
        np.random.set_state(S['np.random.get_state'])
        T.cuda.set_rng_state(S['torch.cuda.get_rng_state'].cpu(), cfg.device)
        T.random.set_rng_state(S['torch.get_rng_state'].cpu())
    # model + optimizer
    if 'model' in S:
        cfg.model = S['model']
    else:
        cfg.model.load_state_dict(S['model.state_dict'])
    cfg.model.to(cfg.device, non_blocking=True)
    if 'optimizer' in S:
        cfg.optimizer = S['optimizer']
    else:
        cfg.optimizer.load_state_dict(S['optimizer.state_dict'])
    cfg.start_epoch = S['cur_epoch']+1


@dc.dataclass
class SegmentationResult(Result):
    """
    Assume labels are a stack of one or more binary segmentation masks and report results independently per mask.

    Aggregate results, for instance over the course of an epoch. Aggregates
    per-pixel errors into a binary confusion matrix for each class (evaluating
    how well foreground and background are separated).

    It also records:
     - a confusion matrix for each task.
     - the total loss (sum)
     - the number of pixels processed
     - the number of images processed

    From each confusion matrix, the dice, accuracy and matthew's correlation
    coefficient are extracted.

        >>> res = SegmentationResult(
          classes=['tumor', 'infection', 'artifact'],
          model_final_layer=None  # function to modify the given model predictions before computing confusion matrix.
          metrics=('mcc', 'acc', 'dice', 'loss', 'num_images', 'confusion_matrix', )  # ... or 'combined_confusion_matrix')
        )
        >>> res.update(yhat, y, loss)
        >>> res.asdict()  # a flattened dict of results.  For each dict key defined in `metrics`, get a value.  If value is itself a dict, merge (flatten) it into existing results.

        yhat and y should have shape (B,C,H,W) where C has num channels.  loss is a scalar tensor.
    """
    # adjustable parameters
    classes: Tuple[str] = ('',)  # e.g. ['tumor', 'infection', 'artifact']
    model_final_layer: Union[T.nn.Module, Callable[[T.Tensor], T.Tensor]] = None
    metrics: Tuple[str] = ('mcc', 'acc', 'dice', 'loss', 'num_images', 'confusion_matrix')  #'combined_confusion_matrix')

    # parameters you probably shouldn't adjust
    _cms: Dict[str, T.Tensor] = dc.field(init=False)
    loss: float = 0
    num_images: int = 0

    def __post_init__(self):
        self._cms = {k: T.zeros((2,2), dtype=T.float)
                     for k in self.classes}

    def update(self,
               yhat: T.Tensor,
               y: T.Tensor,
               loss: T.Tensor,
               ):
        assert len(self.classes) == yhat.shape[1] == y.shape[1]

        self.loss += loss.item()
        self.num_images += yhat.shape[0]
        # change yhat to predict class indices
        if self.model_final_layer is not None:
            yhat = self.model_final_layer(yhat)

        # update confusion matrix
        yhat = yhat.permute(0,2,3,1)
        y = y.permute(0,2,3,1)
        device = y.device
        assert len(self.classes) == y.shape[-1], 'sanity check'
        assert len(self.classes) == yhat.shape[-1], 'sanity check'
        for i,kls in enumerate(self.classes):
            self._cms[kls] = self._cms[kls].to(device, non_blocking=True) + api.metrics.confusion_matrix(
                yhat=yhat[...,i].reshape(-1),
                y=y[...,i].reshape(-1),
                num_classes=2)

    @property
    def dice(self) -> Dict[str,float]:
        ret = {}
        for kls in self.classes:
            cm = self._cms[kls]
            (_, fp), (fn, tp) = (cm[0,0], cm[0,1]), (cm[1,0], cm[1,1])
            ret[f'dice_{kls}'] = (2*tp / (tp+fp + tp+fn)).item()
        return ret

    @property
    def mcc(self) -> Dict[str,float]:
        return {
            f'mcc_{k}': api.metrics.matthews_correlation_coeff(
                self._cms[k]).cpu().item() for k in self.classes}

    @property
    def acc(self) -> Dict[str,float]:
        return {
            f'acc_{k}': api.metrics.accuracy(cm).item()
            for k,cm in self._cms.items()}

    @property
    def confusion_matrix(self) -> Dict[str, np.ndarray]:
        return {f'cm_{k}': self._cms[k].cpu().numpy() for k in self.classes}

    @property
    def combined_confusion_matrix(self) -> np.ndarray:
        """ Convert self.confusion_matrices into a flat matrix.  Useful for logging and simpler data storage"""
        tmp = T.cat([self._cms[k] for k in self.classes]).cpu().numpy()
        # add the class names for each confusion matrix as a column
        tmp = np.column_stack(np.repeat(self.classes, tmp.shape[0]), tmp)
        return tmp


@dc.dataclass
class ClassifierResult(Result):
    """
    Aggregate results, for instance over the course of an epoch and maintain a
    confusion matrix.

    It computes a confusion matrix of the epoch, and also records
     - the total loss (sum)
     - the number of samples processed.

    From the confusion matrix, the accuracy and matthew's correlation
    coefficient are extracted.

        >>> res = ClassifierResult(
          num_classes=2,  # for binary classification
          model_final_layer=None,  # function to modify the given model predictions before confusion matrix.
          metrics=('mcc', 'acc', 'loss', 'num_samples', 'confusion_matrix')  # define what values are output by asdict()
        )
        >>> res.update(yhat, y, loss, minibatch_size)
        >>> res.asdict()

        yhat and y each have shape (B,C) for B samples and C classes.
    """
    # params you can adjust
    num_classes: int
    model_final_layer: Union[T.nn.Module, Callable[[T.Tensor], T.Tensor]] = None
    metrics = ('mcc', 'acc', 'loss', 'num_samples', 'confusion_matrix')  # define what values are output by asdict()

    # params you probably shouldn't adjust
    _confusion_matrix: T.Tensor = dc.field(init=False)  # it is updated to tensor.
    loss: float = 0
    num_samples: int = 0

    def __post_init__(self):
        self._confusion_matrix = T.zeros(self.num_classes, self.num_classes)

    def update(self, yhat: T.Tensor, y: T.Tensor, loss: T.Tensor):
        minibatch_size = y.shape[0]
        # update loss
        self.loss += loss.item()
        self.num_samples += minibatch_size
        # change yhat (like apply softmax if necessary)
        if self.model_final_layer is not None:
            yhat = self.model_final_layer(yhat)
        # update confusion matrix
        self._confusion_matrix = self._confusion_matrix.to(y.device, non_blocking=True)
        self._confusion_matrix += api.metrics.confusion_matrix(yhat=yhat, y=y, num_classes=self.num_classes)
        assert np.allclose(self._confusion_matrix.sum().item(), self.num_samples), 'sanity check'

    @property
    def mcc(self) -> float:
        return api.metrics.matthews_correlation_coeff(self._confusion_matrix).item()

    @property
    def acc(self) -> float:
        return api.metrics.accuracy(self._confusion_matrix).item()
        #  return (self._confusion_matrix.trace() / self._confusion_matrix.sum()).item()

    @property
    def confusion_matrix(self) -> np.ndarray:
        return self._confusion_matrix.cpu().numpy()

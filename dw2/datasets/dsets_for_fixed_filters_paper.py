import numpy as np
import torch as T
from pampy import match
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from functools import partial
import os

from simplepytorch.datasets import CheXpert, CheXpert_Small, BBBC038v1
from dw2.datasets import Preprocess, RandomSubset

#  from .triangles import TrianglesSegmentation


NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 5))


def compose(*funcs):
    def _compose(*inpt):
        for fn in funcs:
            inpt = fn(*inpt)
        return inpt
    return _compose


def random_flip(img, mask):
    # random flipping. one of: "dont flip", horizontal, vertical, horizontal+vertical
    dims = [[], [-1], [-2], [-2,-1]][np.random.randint(0, 4)]
    img, mask = T.flip(img, dims), T.flip(mask, dims)
    return img, mask


def cutout_img(img, mask):
    for x in range(np.random.randint(1, 30)):
        pct_side = np.random.uniform(.01, .1)
        ch,h,w = img.shape
        dy,dx = int(pct_side*h), int(pct_side*w)
        y = np.random.randint(0, h-dy)
        x = np.random.randint(0, w-dx)
        img[:, y:min(h,y+dy), x:min(w,x+dx)] = img.mean((1,2), keepdims=True)
    return img, mask


def add_gaussian_noise(img, label, to_img_prob=.8, to_label_prob=.8):
    ch,h,w = img.shape
    if np.random.uniform() < to_img_prob:
        img = (img + T.randn_like(img)/np.random.uniform(5,10)).clamp(0,1)
    if to_label_prob > 0:
        label = label.float()
    if np.random.uniform() < to_label_prob:
        label = (label + T.randn_like(label)/np.random.uniform(1, 5)).clamp(0,1)
    return img, label


def random_intensity_shift_img(img, mask, prob=.8):
    if np.random.uniform() < prob:
        ch = np.s_[:]
        # random color shift
        shift = np.random.uniform(-.4, .4)
        img[ch] = (img[ch]+shift).clamp(0,1)
    return img, mask


def random_inversion(img, mask, prob=.5):
    if np.random.uniform() < prob:
        img = 1-img
    return img, mask


def random_crop(output_shape, img, mask=None):
    H, W = output_shape
    h, w = img.shape[-2:]
    dh, dw = np.random.randint(0, [h-H+1,w-W+1])
    slicer = np.s_[..., dh:dh+H, dw:dw+W]
    cropped_img = img[slicer]
    if mask is None:
        return cropped_img
    else:
        cropped_mask = mask[slicer]
        return (cropped_img, cropped_mask)


def augment_train_set_BBBC038v1(xy, dset_augmentations_train:str):
    """Augmentations for BBBC038v1"""
    img, mask = xy[0], xy[1]
    img, mask = random_crop((256,256), img, mask)

    aug_fn = match(
        dset_augmentations_train,
        'none', lambda _: (lambda *x: x),
        'v1', lambda _: random_flip,
        'v2', lambda _: compose(random_flip, add_gaussian_noise),
        'v3', lambda _: compose(random_flip, random_intensity_shift_img),
        'v4', lambda _: compose(random_flip, random_inversion),
        'v5', lambda _: compose(random_flip, random_inversion, add_gaussian_noise, random_intensity_shift_img),
        'v6', lambda _: compose(random_flip, random_intensity_shift_img, add_gaussian_noise)
        )
    img, mask = aug_fn(img, mask)
    #  if dset_augmentations_train.startswith('albument'):
        #  # use a preprocessing pipeline searched for by AutoAlbument from
        #  # albumentations library.
        #  #  transform = A.load("albumenting/outputs/2020-12-28/20-04-44/policy/epoch_5.json")
        #  #  transform = A.load("albumenting/outputs/2020-12-28/20-04-44/policy/epoch_9.json")
        #  transform = A.load("albumenting/outputs/2020-12-28/20-04-44/policy/epoch_16.json")
        #  dct = transform(image=img.permute(1,2,0).numpy(), mask=mask.permute(1,2,0).numpy())
        #  img, mask = dct['image'], dct['mask']
        #  img = (img-img.min()) / (img.max()-img.min())
        #  return img, mask

    return img, mask


def _upsample_pad_minibatch_imgs_to_same_size(batch, target_is_segmentation_mask=False):
    """a collate function for a dataloader.  """
    shapes = [item[0].shape for item in batch]
    H = max(h for c,h,w in shapes)
    W = max(w for c,h,w in shapes)
    X, Y = [], []
    for item in batch:
        h,w = item[0].shape[1:]
        dh, dw = (H-h), (W-w)
        padding = (dw//2, dw-dw//2, dh//2, dh-dh//2, )
        X.append(T.nn.functional.pad(item[0], padding))
        if target_is_segmentation_mask:
            Y.append(T.nn.functional.pad(item[1], padding))
        else:
            Y.append(item[1])
    return T.stack(X), T.stack(Y)


def _dset_BBBC038v1(split_train:bool=True, epoch_size=None,
                    dset_augmentations_train:str=None,
                    train_batch_size:int=10, val_batch_size:int=1, stage2=False):
    """Get the train and validation sets.  If split_train is true, get the validation set as 30% of training set.
    """
    # below, xy is a tuple (input_img_tensor, masks_tensor)
    train_dset = BBBC038v1('stage1_train', convert_to=T.Tensor)
    assert not (split_train and stage2)
    if split_train:
        N = int(.7*len(train_dset))
        train_dset, val_dset = T.utils.data.random_split(train_dset, [N, len(train_dset)-N])
    elif stage2:
        val_dset = BBBC038v1(
            'stage2_test_final', convert_to=T.Tensor,
            stage2_only_annotated_imgs=True)
    else:
        train_dset = train_dset
        val_dset = BBBC038v1('stage1_test', convert_to=T.Tensor)

    train_dset = Preprocess(train_dset, tvt.Compose([
        lambda xy: (xy[0].cuda(), xy[1].cuda()),
        lambda xy: (xy[0]/ 255., xy[1].sum(0).unsqueeze_(0)),
        # center to .5 mean
        lambda xy: (xy[0] - xy[0].mean() + .5, xy[1]),
        lambda xy: augment_train_set_BBBC038v1(xy, dset_augmentations_train),
    ]))
    val_dset = Preprocess(val_dset, tvt.Compose([
        lambda xy: (xy[0]/ 255., xy[1].sum(0).unsqueeze_(0)),
        # center to .5 mean
        lambda xy: (xy[0] - xy[0].mean() + .5, xy[1]),
    ]))

    if epoch_size:
        train_dset = RandomSubset(train_dset, n=epoch_size)
    return dict(
        train_dset = train_dset,
        val_dset = val_dset,
        train_loader = DataLoader(
            train_dset, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=False),
        val_loader = DataLoader(
            val_dset, batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=False,
            #  collate_fn=partial(_upsample_pad_minibatch_imgs_to_same_size, target_is_segmentation_mask=True),
        ),
    )


def _dset_CheXpert(dset_augmentations_train:str='none', num_imgs_per_epoch=None, num_val_imgs_per_eval=None, subset_train_size=None, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD):
    assert dset_augmentations_train == 'none', 'augmentations not implemented'
    train_data = CheXpert_Small(
        getitem_transform=lambda x: (x['image'], CheXpert.format_labels(
            x, labels=labels, explode=False)))
    # always the same training and validation set split
    N = int(.7*len(train_data))
    train_dset, val_dset = T.utils.data.random_split(train_data, [N, len(train_data)-N], T.Generator().manual_seed(138))
    # --> expose correct labels_csv for the train set split
    train_dset.labels_csv = train_data.labels_csv.iloc[train_dset.indices]

    if subset_train_size is not None:
        train_dset = T.utils.data.Subset(train_dset, T.randperm(len(train_dset))[:int(subset_train_size)])
        # ... expose correct labels_csv
        train_dset.labels_csv = train_dset.dataset.labels_csv.iloc[train_dset.indices]

    _labels_csv = train_dset.labels_csv
    train_dset = Preprocess( train_dset, tvt.Compose([
        lambda xy: (random_crop((320,320), xy[0]), xy[1]),
    ]))
    train_dset.labels_csv = _labels_csv  # preserve labels_csv

    # get train and val loaders
    if num_imgs_per_epoch:
        train_sampler = T.utils.data.RandomSampler(
            train_dset, replacement=True, num_samples=num_imgs_per_epoch)
    else:
        train_sampler=T.utils.data.RandomSampler(train_dset, replacement=False)
    # --> val loader
    if num_val_imgs_per_eval:
        val_sampler = T.utils.data.RandomSampler(
            val_dset, replacement=True, num_samples=num_val_imgs_per_eval)
    else:
        val_sampler = T.utils.data.RandomSampler(val_dset, replacement=False)
    loader_kws = dict(batch_size=4, num_workers=NUM_WORKERS, pin_memory=False)
    train_loader = DataLoader(train_dset, sampler=train_sampler, **loader_kws)
    val_loader = DataLoader(
        val_dset, sampler=val_sampler, collate_fn=_upsample_pad_minibatch_imgs_to_same_size, **loader_kws)
    return dict(
        train_dset=train_dset, val_dset=val_dset,
        train_loader=train_loader, val_loader=val_loader
    )


def _dset_CheXpert_for_testing(dset_augmentations_train, labels=tuple(CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD)):
    assert dset_augmentations_train == 'none', 'test time augmentations not implemented'
    # get test set
    test_dset = CheXpert_Small(
        use_train_set=False,
        getitem_transform=lambda x: (x['image'], CheXpert.format_labels(
            x, labels=labels, explode=False)))
    loader_kws = dict(batch_size=4, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(
        test_dset, collate_fn=_upsample_pad_minibatch_imgs_to_same_size, **loader_kws)
    return {'test_dset': test_dset, 'test_loader': test_loader}


#  def _get_dataloader(dset: T.utils.data.Dataset, batch_size:int, **kws):
#      defaults = dict(batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True, )
#      defaults.update(kws)
#      return DataLoader(dset, **defaults)


#  def _dset_TS1(dset_augmentations_train:str):
#      if dset_augmentations_train != 'none':
#          raise NotImplementedError()
#      # Dataset TS1
#      # goal: just make transposed image of input triangle (input is binary mask)
#      # fully translation and rotation co-variant dataset for Triangle Segmentation
#      train_dset = Preprocess(TrianglesSegmentation(  # TODO
#          num_imgs=100, translate=True, rotate=(0, 359), scale=(1,10),
#          label_func=lambda **kw: kw['img'].T), lambda xy: (
#              T.tensor(xy[0], dtype=T.float).unsqueeze_(0),
#              T.tensor(xy[1], dtype=T.float).unsqueeze_(0)))
#      # --> make the data have 3 channels
#      train_dset = Preprocess(train_dset, lambda xy: (xy[0].repeat(3,1,1), xy[1]))
#      val_dset = train_dset  # data is generated randomly. it's an iid validation set
#      return dict(
#          train_dset = train_dset,
#          val_dset = val_dset,
#          train_loader = _get_dataloader(train_dset, batch_size=5),
#          val_loader = _get_dataloader(val_dset, batch_size=5),
#      )


_DATASETS = {
    #  'TS1': _dset_TS1,
    'BBBC038v1': partial(_dset_BBBC038v1, split_train=False),
    'BBBC038v1_stage2': partial(_dset_BBBC038v1, split_train=False, stage2=True),
    'BBBC038v1_split': partial(_dset_BBBC038v1, split_train=True),
    'BBBC038v1_debug': partial(_dset_BBBC038v1, epoch_size=100),  # report progress every 10th of an epoch
    # chexpert dataset, all 14 diagnostic labels
    'CheXpert_Small_D': partial(_dset_CheXpert, labels=CheXpert.LABELS_DIAGNOSTIC),
    'CheXpert_Small_D_debug': partial(_dset_CheXpert, num_imgs_per_epoch=50, num_val_imgs_per_eval=100, labels=CheXpert.LABELS_DIAGNOSTIC),
    'CheXpert_Small_D_5k_per_epoch': partial(_dset_CheXpert, num_imgs_per_epoch=5000, num_val_imgs_per_eval=5000, labels=CheXpert.LABELS_DIAGNOSTIC),
    'CheXpert_Small_D_15k_per_epoch': partial(_dset_CheXpert, num_imgs_per_epoch=15000, num_val_imgs_per_eval=15000, labels=CheXpert.LABELS_DIAGNOSTIC),
    # smaller chexpert datasets
    'CheXpert_Small_D_150k': partial(_dset_CheXpert, subset_train_size=150e3, labels=CheXpert.LABELS_DIAGNOSTIC),
    'CheXpert_Small_D_10k': partial(_dset_CheXpert, subset_train_size=10e3, labels=CheXpert.LABELS_DIAGNOSTIC),
    'CheXpert_Small_D_50k': partial(_dset_CheXpert, subset_train_size=50e3, labels=CheXpert.LABELS_DIAGNOSTIC),

    'CheXpert_Small_L': partial(_dset_CheXpert, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD),
    'CheXpert_Small_L_debug': partial(_dset_CheXpert, num_imgs_per_epoch=50, num_val_imgs_per_eval=100, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD),
    'CheXpert_Small_L_500_per_epoch': partial(_dset_CheXpert, num_imgs_per_epoch=500, num_val_imgs_per_eval=500, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD),
    'CheXpert_Small_L_5k_per_epoch': partial(_dset_CheXpert, num_imgs_per_epoch=5000, num_val_imgs_per_eval=5000, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD),
    'CheXpert_Small_L_15k_per_epoch': partial(_dset_CheXpert, num_imgs_per_epoch=15000, num_val_imgs_per_eval=15000, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD),
    # smaller chexpert datasets
    'CheXpert_Small_L_150k': partial(_dset_CheXpert, subset_train_size=150e3, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD),
    'CheXpert_Small_L_10k': partial(_dset_CheXpert, subset_train_size=10e3, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD),
    'CheXpert_Small_L_50k': partial(_dset_CheXpert, subset_train_size=50e3, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD),
}

_DATASETS_TEST = {
    'CheXpert_Small_L_valid': partial(_dset_CheXpert_for_testing, labels=CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD),
    'CheXpert_Small_D_valid': partial(_dset_CheXpert_for_testing, labels=CheXpert.LABELS_DIAGNOSTIC),
}


def get_datasets_and_loaders(dset_name:str, dset_augmentations_train:str) -> dict:
    """

    Return a dict like:
        dict(train_loader=..., val_loader=..., train_dset=..., val_dset=...)
    """
    try:
        dset_fn = _DATASETS[dset_name]
    except KeyError:
        dset_fn = _DATASETS_TEST[dset_name]
    return dset_fn(dset_augmentations_train=dset_augmentations_train)
    #  return match(dset_name, *_DATASETS)

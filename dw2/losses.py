import torch as T
from dw2.models.fixed_filters import iter_conv2d
from dw2.kernel import dct_basis_2d
import pandas as pd
from typing import Any, Callable


def prior_dct_l1(model: T.nn.Module):
    l1_prior = 0
    for conv in iter_conv2d(model):
        o, i, h, w = conv.weight.shape
        F = conv.weight.reshape(o*i, h*w)
        B = dct_basis_2d(h, w)
        assert B.shape == (h*w, h, w), 'sanity check'
        B = T.tensor(B.reshape(h*w, h*w), dtype=T.float, device=F.device)
        l1_prior += (F@B.T).abs().mean()
    return l1_prior


def loss_smooth_dice(yhat, y):
    """ Compute soft dice for each image in minibatch, for each channel.
    Reduce by mean over images and channels.
    """
    yhat = T.sigmoid(yhat)
    bs,ch,h,w = yhat.shape
    eps = 1e-7
    pos = (y * yhat).sum((-1, -2))
    neg = ((1-y) * (1-yhat)).sum((-1, -2))
    numerator = T.stack([pos, neg], dim=1).reshape(bs, ch*2)
    denominator = T.stack([
        y.sum((-1,-2))+yhat.sum((-1,-2)),
        (1-y).sum((-1,-2))+(1-yhat).sum((-1,-2)), ], dim=1).reshape(bs, ch*2)
    smooth_dice_per_img = 1- ((eps + numerator) / (eps + denominator)).sum(1)
    return smooth_dice_per_img.mean(0)


def loss_chexpert(yhat, y):
    """
    dumb simple implementation that ignores the uncertain values and sets missing values to negative.

    note: others show missing and uncertain values are very important in this dataset.
    """
    # set missing values to negative, as done in original chexpert paper.
    y = y.clone()
    y[y == 3] = 0
    # ignore uncertain ground truth, equivalent to U-Ignore in original chexpert paper.
    ignore_uncertain_ground_truth = (y != 2).float()
    return T.nn.functional.binary_cross_entropy_with_logits(
        yhat, y.float(), weight=ignore_uncertain_ground_truth)


class loss_chexpert_focal_bce(T.nn.Module):
    """
    Focal BCE loss, where alpha_(pos|neg) is class balancing weight of
    positive(negative) class and where gamma controls how much to de-emphasize
    "easy" predictions where model was confident and correct.

    Focus on examples model is less certain about.  If we trust ground truth a
    lot, we should probably increase gamma because we can emphasize hard
    examples more.

    BCE is:    -[  y log p    +   (1-y) log (1-p) ]    ( note only one term is active if y in {0,1} )
    focal BCE: -[ (a)(1-p)^gamma y log p +   (1-a)(p)^gamma y log (1-p)
        where `a` is basically a class balancing weight.
              `gamma` places emphasis on "hard" samples that the model has low confidence for

    If gamma=0, its just weighted binary cross entropy, weighted by intra and inter- class balancing

    It's specific to chexpert because the alpha_pos|neg is a class balancing weight
    over the positive and negative labels for each Diagnostic task.

    :chexpert_task_names: a list of tasks, either
        D.CheXpert.LABELS_DIAGNOSTIC or D.CheXpert.LABELS_DIAGNOSTIC_LEADERBOARD

    """
    def __init__(self, chexpert_task_names, chexpert_labels_csv: pd.DataFrame,
                 device: str, gamma:float):
        super().__init__()
        self.chexpert_task_names = chexpert_task_names
        self.gamma = gamma
        self.alpha = self.get_class_balancing_weights_chexpert(chexpert_labels_csv, device)

    def forward(self, yhat, y):
        # set missing values to negative, as done in original chexpert paper.
        y = y.clone()
        y[y == 3] = 0
        # ignore uncertain ground truth, equivalent to U-Ignore in original chexpert paper.
        ignore_uncertain_ground_truth = (y != 2).float()

        # compute focal loss for binary cross entropy.
        # --> note that wneg + wpos != 1.  This means if y not in {0,1}, then we
        # can't use binary_cross_entropy_with_logits because the weights will not
        # get applied correctly.  I just use manual method for clarity and forego
        # the numerical stability benefit of bcewithlogits.
        # --> note: ignore the uncertainty values!
        sigm = T.sigmoid(yhat) * .99999 + .000005
        # --> focal loss where alpha is the class balancing weight
        w_pos = self.alpha['pos'] * (1-sigm)**self.gamma
        w_neg = self.alpha['neg'] * (sigm)**self.gamma
        bce_loss = -1.*ignore_uncertain_ground_truth*self.alpha['interclass']*(
            w_pos * y*T.log(sigm) + w_neg * (1-y)*T.log(1-sigm)
            #  y*T.log(sigm) + (1-y)*T.log(1-sigm)
        )
        bce_loss = bce_loss.sum(1).mean(0)
        if T.isnan(bce_loss).any():
            raise Exception("loss cannot be nan")
        return bce_loss  # reduce sum over classes and minibatch samples

    def get_class_balancing_weights_chexpert(self, chexpert_labels_csv: pd.DataFrame, device: str):
        """Intra- and Inter- Class balancing weights for chexpert focal loss.
        Balances the positive and negative values.  Balances across tasks.
        In both cases, based on the following:

            max(w) / w
                where w is:
                    - in the intra-class balancing setting, the counts of
                    positive and of negative labels for any given diagnostic
                    task in the training dataset  (w is a vector of 2 values,
                    and there is one per task)
                    - in the inter-class balancing setting, the count of
                    positive+negative labels across all tasks (w is a vector of
                    length equal to num tasks)

        """
        df = chexpert_labels_csv[self.chexpert_task_names]
        df = df.replace({3: 0})

        # intra-class balancing:
        # balance the pos and negative class balancing weights (the alpha term
        # in focal loss paper, but a different value for each Diagnostic Label)
        # --> ensure that the counts of pos and neg labels are from the training set.

        # --> compute counts, then get class_balancing_weight = (max / counts) as a probability
        counts = df.apply(pd.Series.value_counts)  # (includes unused uncertainty labels here)
        cbw = counts.loc[[0,1]]  # consider only (pos, neg) labels so that with below implementation, all tasks get equal representation in weighting
        cbw = cbw.max() / cbw  # given a diagnostic task, assign each label (pos, neg, uncertain) equal representation
        cbw = cbw/cbw.sum()  # normalize this assignment into a probability.
        cbw_neg = T.tensor(cbw.loc[0].values, device=device, dtype=T.float).reshape(1,-1)
        cbw_pos = T.tensor(cbw.loc[1].values, device=device, dtype=T.float).reshape(1,-1)

        # when the dataset is too small, some tasks have no data and cannot be learned.
        # in this case, weights are zero.
        ignore_mask = (cbw_neg.isnan() | cbw_pos.isnan() | (cbw_neg == 0) | (cbw_pos == 0) )
        if ignore_mask.sum() > 0:
            print("WARNING: train dataset too small.  some diagnostic tasks have 0 training samples: \n",
                  pd.DataFrame({'neg_weights': cbw_neg.cpu().numpy().reshape(-1), 'pos_weights': cbw_pos.cpu().numpy().reshape(-1)}, index=df.columns).T.to_string())
        cbw_pos[ignore_mask] = 0
        cbw_neg[ignore_mask] = 0

        # inter-class balancing: balance the number of samples per class
        # Discard (set to zero) the tasks with either no positive or no negative labels., then normalize.
        inter = counts.loc[[0,1]].sum(0)
        inter = T.tensor(inter.values, device=device, dtype=T.float).reshape(1,-1)
        inter = inter.max()/inter  # class balancing
        inter[ignore_mask] = 0  # discard tasks with either no positive or no negative samples
        inter = inter/inter.sum()

        assert T.allclose(T.tensor(1., device=device), cbw_pos[~ignore_mask] + cbw_neg[~ignore_mask]), 'sanity check'
        assert T.allclose(inter.sum(), T.tensor(1., device=device)), 'sanity check'

        return {'pos': cbw_pos, 'neg': cbw_neg, 'interclass': inter}


def loss_generalized_dice(yhat, y):
    """Compute generalized dice score
    as done in https://arxiv.org/pdf/1707.03237.pdf
    for each image. Then, average the score for each img in the minibatch.

    Use the default weight from the paper to balance volume of foreground and background pixels.
    This implementation assigns weight to each image independently.
    """
    yhat = T.sigmoid(yhat)
    bs,ch,h,w = yhat.shape
    with T.no_grad():
        wpos = (y.sum((1,2,3))+1e-8)**(-2)
        wneg = ((1-y).sum((1,2,3))+1e-8)**(-2)
    numerator = wpos*(y*yhat).sum((1,2,3)) + wneg*((1-y)*(1-yhat)).sum((1,2,3))
    denominator = wpos*(y+yhat).sum((1,2,3)) + wneg*((1-y)+(1-yhat)).sum((1,2,3))
    gdl = 1 - 2*(numerator)/(denominator)
    return gdl.mean(0)


def loss_bcewithlogits(input, target):
    return T.nn.functional.binary_cross_entropy_with_logits(input, target.float())


_LOSSES = {
    'BCEWithLogitsLoss': loss_bcewithlogits,
    'smooth_dice': loss_smooth_dice,
    'generalized_dice': loss_generalized_dice,
    'chexpert': loss_chexpert,
    'chexpert_focal:1': ...,  # updated in train_configloss_chexpert_focal_bce,
    'chexpert_focal:0': ...,  # updated in train_configloss_chexpert_focal_bce,
}

_PRIORS = {
    'dct_l1:1': prior_dct_l1,
    'dct_l1:.1': prior_dct_l1,
    'dct_l1:.01': prior_dct_l1,
    'dct_l1:10': prior_dct_l1,
}


def loss_with_prior(loss_fn,
                    prior_fn: Callable['Any', T.Tensor],
                    prior_attention, *prior_args, **prior_kwargs):
    """Compute loss + prior, where loss is standard pytorch loss, and the
    prior_fn(...) is a function that that analyzes the given input
    and outputs a value.  Both loss and prior are assumed part of backprop.
    Assumes the `prior_args` and `prior_kwargs` keep a reference of the mutable
    state, so as model updates during training, this state is updated too.
    """
    def wrapped_fn(*args, **kwargs):
        loss = loss_fn(*args, **kwargs)
        prior = prior_fn(*prior_args, **prior_kwargs) * prior_attention
        return loss + prior
    return wrapped_fn

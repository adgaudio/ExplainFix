"""
"""
import torch as T
import torch.nn.functional as F
import pytorch_wavelets as pyw
import torchvision.transforms as tvt
from collections import OrderedDict
from dw import lung_opacity, core


def _get_stats(mat: T.Tensor):
    """
    Compute summarizing statistics on given detail coefficients from any level
    of the discrete wavelet transform.
    Each detail matrix is independent of others (which is probably not ideal).
    """
    dim = (-1,-2)
    sh = mat.shape
    stats = [
        T.tensor(sh[-2]*sh[-1], dtype=mat.dtype,
                 device=mat.device).repeat(sh[:-2]),  # count elems
        mat.sum(dim),  # sum
        (mat**2).sum(dim),  # sum sq
        mat.logsumexp(dim),  # max, smooth, convex, differentiable, tight approx
        -1*((-1*mat).logsumexp(dim)),  # min, approximate
        mat.norm(p=0, dim=dim),  # num zeros
        mat.norm(p=1, dim=dim),  # sqrt sum abs val
        mat.norm(p=2, dim=dim),  # sqrt sum sq
        mat.norm(p=3, dim=dim),  # sqrt sum cube
    ]
    stats = T.cat(stats, -1)
    return stats


def summarizing_stats(detail):
    """
    Summarize the detail coefficients from each level of the multi-level
    wavelet transform.

    Each level has three detail matrices describing horizontal edges, vertical
    edges and diagonal edges, respectively. Summarize each matrix (and channel)
    INDEPENDENTLY.  Future work could to consider them dependently,
    or perhaps the wavelet could be a 3d wavelet.

    :mat: (3,h,w) tensor: detail coefficients for a level of the wavelet transform
    :detail: list of shape [(n,c,3,h,w), ...].  Detail coefficients from
        multi-level wavelet transform output by pytorch_wavelets.DWT, where n
        is num images in minibatch, c is num image channels, 3 is the (fixed)
        num of detail matrices, h and w are data height and width.  Length of
        list corresponds to num of levels.
    :returns: Tensor of shape (n,c,27,9), where (n,c) come from input,
      27 corresponds to the number of features extracted from each detail
      matrix, 9 represents the 3 detail matrices, where for each matrix we
      compute three sets of 27 values (corresponding to positive edges,
      negative edges and whole image).

    Statistics:
        - gaussian:
          - mean, var
          - sum sq statistics gives us variance!  var[x] = e[x^2] - x[x]^2
        - uniform:
          - max, min
        - poisson and bernoulli and exponential:
          - sum
        - gamma:
          - sum, product (or i suppose log sum)

        - num zeros (l0 norm)
        - sum abs values (l1 norm)
        - mean pow(3)  (to preserve negative sign and be non-linear)

    We compute these three times for each detail matrix.  One corresponding to
    all edges, once for just positive values (positive edges), once for just
    negative values (negative edges).

    """
    # on detail matrices at each level
    dev = detail[0].device
    # --> for the whole image
    lst1 = (_get_stats(level) for level in detail)
    # --> for only positive wavelet coefficients ("inside" edges when image background is black)
    lst2 = (_get_stats(level.where(level >0, T.tensor(0.).to(dev))) for level in detail)
    # --> for only negative wavelet coefficients ("outside" edges when image background is black)
    lst3 = (_get_stats(level.where(level <0, T.tensor(0.).to(dev))) for level in detail)
    rv = T.stack([x for y in [lst1, lst2, lst3] for x in y], -1)
    n,c = detail[0].shape[:2]
    assert rv.shape == (n,c,
                        9*3, # 9 stats, computed three ways: whole, positive and negative
                        3*len(detail),  # num detail matrices (assume 3 per level)
                        )
    return rv


def wavelet_coefficients_as_tensorimage(approx, detail, normalize=False):
    norm11 = lambda x: (x / max(x.min()*-1, x.max()))  # into [-1,+1] preserving sign
    fixed_dims = detail[0][0].shape[:-3] # num images in minibatch, num channels, etc
    output_shape = fixed_dims + (
        detail[0][0].shape[-2]*2,  # input img height
        detail[0][0].shape[-1]*2)  # input img width
    im = T.zeros(output_shape)
    #  if normalize:
        #  approx = norm11(approx)
    im[..., :detail[-1].shape[-2], :detail[-1].shape[-1]] = approx if approx is not None else 0
    for level in detail:
        lh, hl, hh = level.unbind(-3)
        h,w = lh.shape[-2:]
        if normalize:
            lh, hl, hh = [norm11(x) for x in [lh, hl, hh]]
        #  im[:h, :w] = approx
        im[..., 0:h, w:w+w] = lh  # horizontal
        im[..., h:h+h, :w] = hl  # vertical
        im[..., h:h+h, w:w+w] = hh  # diagonal
    return im


class CheXpertWaveletDataset_Mixin(core.CheXpertMixin):
    wavelet_num_levels = 9  # num levels in wavelet transform
    _dwt = None  # lazy initialized

    def imgtensor_to_wavelet(self, im):
        im = im.unsqueeze(0)
        assert len(im.shape) == 4
        if self._dwt is None:
            self._dwt = pyw.DWT(
                J=self.wavelet_num_levels, wave='haar', mode='zero'
            ).to(self.device)
        approx, detail = self._dwt(im)
        return approx, detail

    def _as_image(self, approx_detail):
        return wavelet_coefficients_as_tensorimage(*approx_detail)

    def get_img_transform(self):
        return tvt.Compose([
            super().get_img_transform(),
            lambda x: x.to(self.device),
            self.imgtensor_to_wavelet,
            self._as_image
        ])



class CheXpertWaveletStatsDataset_Mixin(CheXpertWaveletDataset_Mixin):

    def wavelet_to_summary_stats(self, approx_detail):
        approx, detail = approx_detail
        ret = summarizing_stats(detail)
        return ret

    def get_img_transform(self):
        return tvt.Compose([
            core.CheXpertMixin.get_img_transform(self),
            lambda x: x.to(self.device),
            self.imgtensor_to_wavelet,
            self.wavelet_to_summary_stats,
            lambda x: x.squeeze(0),  # just applied to single img.
        ])


class LinearBlock(T.nn.Sequential):
    def __init__(self, num_input_units, num_output_units):
        super().__init__(
            T.nn.Linear(num_input_units, num_output_units, bias=True),
            T.nn.ReLU(),
            T.nn.BatchNorm1d(num_output_units)
        )


class chexpert_wavelet_stats_tiled_for_efficientnet_mixin:
    def get_img_transform(self):
        """HACK to double the number of stats so
        efficientnet convolutions will work and can be tested.
        This is just dumb, but enables efficientnet to be tested.  """
        return tvt.Compose([ 
            super().get_img_transform(),
            lambda x: x.repeat(1,2,2),
        ])


class LinearLayer_Mixin:
    """Pulls together most components needed
    to train a model on CheXpert

    For extra features and documentation, consult the
    api.FeedForwardModelConfig class, and also explore the library
    available under simplepytorch.api.*
    """
    lr = 0.0001

    def preprocess_hook(self, X, y):
        """
        little hack to flatten the input into a vector per image.
        """
        batch_size = X.shape[0]
        assert X.shape == (batch_size, 1,
                           self._num_input_units()/3/self.wavelet_num_levels,
                           3*self.wavelet_num_levels)
        X = X.reshape(X.shape[0], -1)
        return X, y.float()

    def _num_input_units(self):
        return (
            9 *  # summary stats per edge type
            3 *  # three edge types (whole image, pos edges, neg edges)
            3 *  # three detail matrices per level
            self.wavelet_num_levels)

    def get_model(self):
        in_ = self._num_input_units()
        out_ = 1  # just lung opacity
        layers = [in_, in_, in_, in_, in_, in_, out_]
        self.model = T.nn.Sequential(OrderedDict([
            (f'linearblock{i}', LinearBlock(x, x2))
            for i, (x,x2) in enumerate(zip(layers, layers[1:]))]))

    @staticmethod
    def bce_lung_opacity_loss(input, target):
        # TODO: class balancing weights
        return F.binary_cross_entropy_with_logits(input, target.float())

    def get_lossfn(self):
        self.lossfn = T.nn.BCEWithLogitsLoss()

    def get_optimizer(self):
        self.optimizer = T.optim.Adam(
            self.model.parameters(), lr=self.lr
        )


class LungOpacity_Wavelet_EfficientNet(
    # model, loss, optimizer
    lung_opacity.LungOpacity_EfficientNet_Mixin,
    # evaluation
    lung_opacity.LoggingAndValPerfLungOpacity,
    # dataset
    lung_opacity.CheXpert_only_positive_lung_opacity_labels,
    CheXpertWaveletDataset_Mixin,
    # boilerplate config
    core.BaseFeedForwardConfig): pass


class LungOpacity_WaveletStats_Linear(
    # model, loss, optimizer
    LinearLayer_Mixin,
    # evaluation
    lung_opacity.LoggingAndValPerfLungOpacity,
    # dataset
    lung_opacity.CheXpert_only_positive_lung_opacity_labels,
    CheXpertWaveletStatsDataset_Mixin,
    # boilerplate config
    core.BaseFeedForwardConfig): pass


class LungOpacity_WaveletStats_EfficientNet(
    # model, loss, optimizer
    lung_opacity.LungOpacity_EfficientNet_Mixin,
    # evaluation
    lung_opacity.LoggingAndValPerfLungOpacity,
    # dataset
    chexpert_wavelet_stats_tiled_for_efficientnet_mixin,
    lung_opacity.CheXpert_only_positive_lung_opacity_labels,
    CheXpertWaveletStatsDataset_Mixin,
    # boilerplate config
    core.BaseFeedForwardConfig): pass


if __name__ == "__main__":
    """ debugging """
    import simplepytorch.api as api
    #  C = api.load_model_config(LungOpacity_Wavelet_EfficientNet, '--use-debugging-dataset')
    C2 = api.load_model_config(LungOpacity_WaveletStats_EfficientNet, '--use-debugging-dataset --wavelet-num-levels 12')
    for x,y in C2.data_loaders.train:
        break

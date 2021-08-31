from efficientnet_pytorch import EfficientNet
import scipy as sp
import scipy.stats
import os
import torch as T
import torchvision.models as tvm
from dw2.models import extract_spatial_filters, extract_all_spatial_filters, iter_conv2d
from dw2 import kernel as K


def get_svdsteering_basis_and_parameters(F, center=False, percent_variability_explained=1+1e-7) -> dict:
    """
    Basic idea is that F = USV^T  implies FV = US.  V is a set of basis
    vectors. Consider FV.  It does two things:  1) it has the special property
    that cov(FV) is diagonal; 2) it gives a set of attention weights over the
    basis vectors.  For each i^{th} basis vector, we have a distribution of
    attention weights, and we can describe the distribution with the mean
    (`mean_i`) and standard deviation (`std_i`).  These two facts conveniently
    help each other.

    Next, we can generate a new matrix, G, containing simulated attention
    weights (and diagonalized covariance).  Sample values G ~ N(0,1) * std_i +
    mean_i, where std_i and mean_i are defined from FV for each i^{th} column
    of G (as described above).  Finally, un-doing the diagonalization with GV^T
    will give a sampled approximation of the covariance of F.  The accuracy of
    the approximation increases as the sample size (i.e. number of rows of G)
    increases.

    This function returns just the V^T, mean and std so that you can generate the
    matrix later.

    Also, it's possible to make V and undercomplete basis by using only most
    relevant eigenvectors (e.g. this is PCA).  Use the
    `percent_variability_explained` to pick the first N eigenvectors in V that
    describe that much percent of the cumulative variance.  Ranges [0,1].
    Numbers less than 1 will result in steering an undercomplete basis.

        >>> N = 10000
        >>> random_steering_weights = T.randn((N, num_components))*std.cpu()+mean.cpu()
        >>> F2 = (random_steering_weights @ Bu.cpu())#.reshape(N, 3,3)
    """
    # to center or not to center?  that is a question
    #  svd = T.linalg.svd(F)   # impossible because some networks have >8Million filters
    Fmean = F.mean(0, keepdims=True)
    if center:
        F = F - Fmean
        #  svd = T.linalg.svd((F-F.mean(0, keepdims=True)).T@(F-F.mean(0, keepdims=True))/(F.shape[0]-1))
    #  else:
    svd = T.linalg.svd(F.T@F)  # no centering.
    Vt = svd.V
    S2 = svd.S**2
    pct_variance = (S2/S2.sum()).cumsum(0)
    num_components = (pct_variance <= percent_variability_explained+1e-7).sum()
    #  print(pct_variance, num_components)
    weights = S2[:num_components]
    # saved results
    num_components = num_components
    Bu = undercomplete_basis = Vt[:num_components]
    mean = (F@Bu.T).mean(0)[:num_components]
    std = (F@Bu.T).std(0)[:num_components]
    kde = sp.stats.gaussian_kde(Bu@F.T)
    dct = dict(Bu=Bu.cpu(), mean=mean.cpu()[:num_components], std=std.cpu()[:num_components],
               Fmean=Fmean*(1 if center else 0), pct_variance=pct_variance,
               kde=kde,
               )
    return dct


def plot_random_weights(Bu, mean, std, Fmean, pct_variance, kde, H, W, **ignore):
    num_components = Bu.shape[0]
    #
    #  example to generate N filters
    N = 10000
    #  random_steering_weights = T.randn((N, num_components))*std.cpu()+mean.cpu()
    #  F2 = random_steering_weights @ Bu.cpu() + Fmean  #.reshape(N, 3,3)
    # ... or use the kde (gaussian kernel density estimate)
    F2 = T.from_numpy(kde.resample(N).T).float() @ Bu.cpu()
    #
    #  sanity check the covariance has desired structure
    B = K.dct_basis_2d(H, W, 'DCT-II')
    assert B.shape == (H*W,H,W)
    B = T.tensor(B.reshape(H*W,H*W), dtype=T.float, device=F2.device)
    from matplotlib import pyplot as plt
    plt.figure()
    plt.title(f'{H}x{W}')
    plt.imshow((B@F2.T@F2@B.T / (F2.shape[0]-1)).cpu().numpy())
    #  plt.imshow((F2.T@F2/F2.shape[0] ).cpu().numpy())
    plt.colorbar()


# The default Efficientnet-b7 weights
#  m = EfficientNet.from_pretrained('efficientnet-b7', advprop=True)
#  dcts = {}
#  for H, W in [(3,3), (5,5)]:  #, (7,7), (11,11)]:
#      F = extract_spatial_filters(m, H, W)
#      dcts[(H,W)] = get_svdsteering_basis_and_parameters(F)
#      plot_random_weights(**dcts[(H,W)], H=H, W=W)
#  T.save(dcts, 'svdsteering_efficientnet-b7.pth')


# Aggregate weights from many pretrained models.
from dw2.configs import fixed_filters
def load_model_chexpert(model_name:str):
    base_dir = f'results/{model_name}-unmodified_baseline-CheXpert_Small_L_15k_per_epoch-chexpert_focal:1'
    checkpoint_filename = 'epoch_40.pth'
    cfg = fixed_filters.load_cfg_from_checkpoint(base_dir, checkpoint_filename)
    model = cfg.model
    return model_name, model


models = [
        lambda k=k: (f'{k} pretrained', getattr(tvm, k)(pretrained=True))
        for k in [
            #  'wide_resnet101_2', ]
            'alexnet', 'googlenet', 'vgg19', 'vgg19_bn', 'wide_resnet101_2',
            'mnasnet1_0', 'mobilenet_v2', 'shufflenet_v2_x1_0', 'squeezenet1_1',
                  'resnext101_32x8d', 'resnet152', 'resnet50', 'resnet18',
                  'densenet201', 'densenet121', 'inception_v3'
        ]
    #  ] + [
        #  lambda k=k: (f'{k} pretrained', EfficientNet.from_pretrained(k))
        #  for k in ['efficientnet-b7', 'efficientnet-b0']
    ] + [
        lambda k=k: (f'{k} adv-pretrained', EfficientNet.from_pretrained(k, advprop=True))
        #  for k in ['efficientnet-b7', 'efficientnet-b0']
        for k in ['efficientnet-b0',
                  'efficientnet-b1',
                  'efficientnet-b2',
                  'efficientnet-b3',
                  'efficientnet-b4',
                  'efficientnet-b5',
                  'efficientnet-b6',
                  'efficientnet-b7',
                  ]
        #  for k in ['efficientnet-b7']
    #  ] + [
        #  lambda: load_model_chexpert('C8-efficientnet-b0:5:fromscratch'),
        #  lambda: load_model_chexpert('C8-efficientnet-b0:5:pretrained'),
        #  lambda: load_model_chexpert('C8-densenet121:5:fromscratch'),
        #  lambda: load_model_chexpert('C8-densenet121:5:pretrained'),
        #  lambda: load_model_chexpert('C8-resnet50:5:fromscratch'),
        #  lambda: load_model_chexpert('C8-resnet50:5:pretrained'),
    ]

# for each model, get all spatial (NxM) filters grouped by their shape
all_F = {}
all_models = {}
for get_model in models:
    with T.no_grad():
        model_name, model = get_model()
        filts = list(extract_all_spatial_filters(model))
    del model
    # ... group by kernel_shape
    grouped = {}  # {(3,3): [(0, F_0), ..., (8, F_8)]}
    for (kernel_shape, layer_idx, layer_F) in filts:
        grouped.setdefault(kernel_shape, []).append((layer_idx, layer_F))
        all_models[model_name] = grouped
    del filts

#  dcts = {}
#  for k, v in all_F.items():
    #  H, W = k
    #  if (H!=W): continue  # HACK
    # local averaging of svds method
    #  tmp = {'Bu': 0, 'mean': 0, 'std': 0}
    #  for F in v:
    #      dct = get_svdsteering_basis_and_parameters(F=F)
    #      for k, vv in dct.items():
    #          tmp[k] = vv + tmp[k]/len(v)
    #  if len(v):
    #      plot_random_weights(**tmp, H=H, W=W)
    # global svd method
    #  dcts[k] = get_svdsteering_basis_and_parameters(F=T.cat(
        #  v, 0))
        #  [2*vv-vv.mean(0, keepdims=True)-vv.mean(1,keepdims=True) for vv in v], 0))
    #  plot_random_weights(**dcts[k], H=H, W=W)
    #
    # plot to compare svd steered weights to the model original weights
    #  F = T.cat(v, 0)
    #  from matplotlib import pyplot as plt
    #  B = T.tensor(K.dct_basis_2d(H, W, 'DCT-II').reshape(H*W,H*W), dtype=F.dtype, device=F.device)
    #  plt.figure() ; plt.imshow((B@F.T@F@B.T / (F.shape[0]-1)).cpu().numpy())

#  T.save(dcts, 'svdsteering_avg.pth')
#  T.save(dcts, 'svdsteering/b7wrn.pth')
#  T.save(dcts, 'svdsteering_b0.pth')

for centered in [True, False]:
    save_dir = f'svdsteering_{"centered" if centered else "noncentered"}'
    os.makedirs(save_dir, exist_ok=True)
    for model_name, filts in all_models.items():
        dct = dict()
        dct['layer_params'] = layer_params = {}
        for (H,W), lst in filts.items():
            # --> get the basis and singular values (as percent variance) using SVD
            # ... and also get mean and std globally across model
            # F concatenates spatial filters of all layers of given size (H,W)
            F = T.cat([x[1] for x in lst])
            dct[(H,W)] = get_svdsteering_basis_and_parameters(F=F, center=centered)
            # --> get the mean and std of the principle vectors for each layer
            Bu = dct[(H,W)]['Bu']
            dct[(H,W)]['mean_per_layer'] = {
                layer_idx: (F@Bu.T).mean(0, keepdims=True)
                for layer_idx, F in lst}
            dct[(H,W)]['std_per_layer'] = {
                layer_idx: (F@Bu.T).std(0, keepdims=True)
                for layer_idx, F in lst}
            dct[(H,W)]['kde_per_layer'] = {
                layer_idx: sp.stats.gaussian_kde(((F@Bu.T).T).cpu().numpy())
                for layer_idx, F in lst}
            for layer_idx, _F in lst:
                layer_params[layer_idx] = get_svdsteering_basis_and_parameters(_F)
            del Bu

        T.save(dct, f'{save_dir}/{model_name.split(" ")[0]}.pth')

# example of what happens as you remove bases
#  plot_random_weights(**dct[(3,3)], H=3, W=3)
#  for _ in range(9):
#      dct[(3,3)]['Bu'] = dct[(3,3)]['Bu'][:-1]
#      dct[(3,3)]['Bu'].shape
#      dct[(3,3)]['mean'] = dct[(3,3)]['mean'][:-1]
#      dct[(3,3)]['std'] = dct[(3,3)]['std'][:-1]
#      plot_random_weights(**dct[(3,3)], H=3, W=3)


from matplotlib import pyplot as plt
plt.show()

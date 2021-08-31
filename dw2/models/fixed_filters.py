import numpy as np
import torch as T
import math
from dw2.models import unet
import dw2.kernel as kernel
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from .util import iter_conv2d


def convert_all_spatial_conv2d(model: T.nn.Module, requires_grad: bool, method: str, fix_1x1convs:bool=False):
    """Convert spatial convolutions to fixed filters.
    Optionally, make the filters fixed (not learned).
    Optionally, make the 1x1 convolutions also not learned, but don't modify their values.
    """
    # spatial conv2d: assign a fixed filter
    for layer_idx, conv2d in enumerate(iter_conv2d(model, include_spatial=True, include_1x1=False)):
        convert_conv2d_to_fixed(
            conv2d, requires_grad=requires_grad, method=method,
            layer_idx=layer_idx)
    # 1x1 conv2d: just fix them
    if fix_1x1convs:
        for conv2d in iter_conv2d(model, include_spatial=False, include_1x1=True):
            conv2d.requires_grad_(False)
    return model


def convert_some_randomly(model: T.nn.Module, pct: float, requires_grad: bool, method: str):
    """With probability `pct`, assign the weights of a Conv2D to fixed, non-learned ones"""
    for layer_idx, module in enumerate(iter_conv2d(model, include_spatial=True, include_1x1=False)):
        if T.rand(1).item() > pct:
            continue
        else:
            convert_conv2d_to_fixed(module, requires_grad=requires_grad,
                                    method=method, layer_idx=layer_idx)
    return model


def convert_nth_conv2d(model: T.nn.Module, idxs, requires_grad: bool, method: str):
    """Convert the nth spatial conv2d layer in the model to a fixed filter."""
    if isinstance(idxs, int):
        idxs = [idxs]
    convs = list(iter_conv2d(model, include_spatial=True, include_1x1=False))
    for i in idxs:
        convert_conv2d_to_fixed(
            convs[i], requires_grad=requires_grad, method=method, layer_idx=i)
    return model


def norm01(x):
    return (x-x.min()) / (x.max()-x.min())


def sumto1(x):
    return x/x.sum()


def convert_conv2d_to_fixed(conv2d: T.nn.Conv2d, requires_grad: bool,
                            method: str, layer_idx: int):
    assert isinstance(conv2d, T.nn.Conv2d)
    assert conv2d.kernel_size[0] == conv2d.kernel_size[1]
    ch = np.prod(conv2d.weight.shape[:2])
    kernel_size = conv2d.kernel_size[0]
    device = conv2d.weight.device

    if method == "GHaar4.s1":
        # (random DCT2 steering of the 4 basis filters with lowest frequency)
        N, M = kernel_size, kernel_size
        filters = [
            T.from_numpy(kernel.dct_steered_2d(
                N=N, M=M,
                w=np.random.randn(4),
                basis_idxs=np.array([0,1,M,M+1]))).float()
            for _ in range(ch)]
    elif method == "GHaar4.s2":
        # (random DCT2 steering of the 4 basis filters with 2nd lowest freq)
        N, M = kernel_size, kernel_size
        filters = [
            T.from_numpy(kernel.dct_steered_2d(
                N=N, M=M,
                w=np.random.randn(4),
                basis_idxs=np.array([0,1+1,M+M,M+1+M]))).float()
            for _ in range(ch)]
    elif method == "GHaarR.ms":
        # random DCT2 steering of a random subset of DCT2 basis vectors)
      R = int(np.round(2**(np.random.uniform(0, np.log2(kernel_size*kernel_size), 1))))
      filters = [
          T.from_numpy(kernel.dct_steered_2d(
              N=kernel_size, M=kernel_size,
              w=np.random.randn(R),
              basis_idxs=np.random.choice(kernel_size*kernel_size, R, replace=False))).float()
          for _ in range(ch)]
    elif method == "GHaar2.ms":
        R = 2
        filters = [
          T.from_numpy(kernel.dct_steered_2d(
              N=kernel_size, M=kernel_size,
              w=np.random.randn(R),
              basis_idxs=np.random.choice(kernel_size*kernel_size, R, replace=False))).float()
          for _ in range(ch)]
    elif method == "GHaar4.ms":
        R = 4
        filters = [
          T.from_numpy(kernel.dct_steered_2d(
              N=kernel_size, M=kernel_size,
              w=np.random.randn(R),
              basis_idxs=np.random.choice(kernel_size*kernel_size, R, replace=False))).float()
          for _ in range(ch)]
    elif method == "PsineR.ms":
        # Psine.ms (multi-scale random polynomial steering of a random subset of DCT2 basis vectors)
        R = int(np.round(2**(np.random.uniform(0, np.log2(kernel_size*kernel_size), 1))))
        filters = [
            T.from_numpy(kernel.dct_steered_2d(
                N=kernel_size, M=kernel_size,
                w=np.random.randn(R),
                basis_idxs=np.random.choice(kernel_size*kernel_size, R, replace=False),
                p=np.random.randint(1, max(1+1, 1+(R-1)/2), R))).float()  # min p == 1
            for _ in range(ch)]
    elif method == 'ghaarA':
        # no approximation, all frequency scales, no normalization
        weights = np.random.rand(ch, 3)*2-1
        filters = [
            kernel.haar2d(kernel_size, *hvd, f=np.random.uniform(0, 2*(kernel_size-1)), norm=False)
            for hvd in weights]
    elif method == 'haar':
        # randomly initialized haar weights
        weights = np.random.rand(ch, 3)*2-1
        filters = [
            kernel.haar2d(kernel_size, *hvd, f=np.random.uniform(0, f+1))
            for f, hvd in enumerate(weights)]
        # view some stats on the kernels (after importing a bunch of code)
        #  filters = T.stack(filters)
        #  z = unet.compose_reduce2_fns([
        #      lambda x,y: (T.conv2d(x.reshape(1,1,*x.shape[-2:]).float(),
        #                                  y.reshape(1,1,*y.shape[-2:]).float(),
        #                                  padding=kernel_size-1).squeeze())
        #                   ]*len(filters),
        #      filters,
        #      filters[np.random.randint(0,100)])
        #  plot_img_grid([x.detach().cpu().numpy() for x in z], 'composed fitlers')
        #  print(filters.max(), filters.min(), filters.sum((1,2)).mean())
        #  print(T.stack([T.tensor([x.min(), x.max(), x.sum()]) for x in z]).mean(0))
        #  #  plot_img_grid(filters, 'filters')
    elif method == 'polynomial_ND':
        # least restrictive
        N, M = 2, kernel_size
        _L = (2**(6*T.rand((ch,)))).long()
        filters = [kernel.polynomial_ND(
            T.rand(L, N, M, device=device)*2-.5,
            p=T.rand(L, device=device)*((L-1)/2-1)+1,
            w=sumto1(T.rand(L, device=device)))
            for L in _L]
        filters = [norm01(x)*2-1 for x in filters]
        filters = [ x/(((x**2).sum())**.5) for x in filters]
        #  ...
        #  filters = T.stack(filters)
        #  z = unet.compose_reduce2_fns([
        #      lambda x,y: (T.conv2d(
        #          x.reshape(1,1,*x.shape[-2:]).float(),
        #          y.reshape(1,1,*y.shape[-2:]).float(),
        #          padding=kernel_size-1).squeeze())   ]*len(filters),
        #      filters, filters[np.random.randint(0,100)])
        #  print(filters.max(), filters.min(), filters.sum((1,2)).mean())
        #  print(T.stack([T.tensor([x.min(), x.max(), x.sum()]) for x in z]).mean(0))
        #  plot_img_grid(filters, 'filters')
        #  plot_img_grid([x.detach().cpu().numpy() for x in z], 'composed fitlers')

    elif method == 'polynomial_sin_ND':
        # between least and most restrictive
        _, N, M = ..., 2, kernel_size
        make_params = lambda L, fmax: dict(
            f=T.rand((L, N), device=device)*fmax,
            t=T.rand((L, N), device=device) * math.pi*2,
            p=T.rand((L,), device=device)*((L-1)/2-1)+1,
            w=sumto1(T.rand((L, ), device=device))*2-1,
            M=M,
        )
        _L = (2**(6*T.rand((ch,))))  # just sample in log space from 0 to 64, roughly.
        filters = [
            kernel.polynomial_sin_ND(**make_params(L=int(3+L), fmax=np.random.uniform(1, 5)))
            for L in _L]  # gradually increase complexity as channels increase
        filters = [norm01(x)*2-1 for x in filters]
        filters = [ x/(((x**2).sum())**.5) for x in filters]
    elif method == 'ones':
        filters = T.ones_like(conv2d.weight)
    elif method == 'kaiming_uniform':
        # only the spatial filter.  not the bias.
        filters = T.nn.init.kaiming_uniform_(conv2d.weight, a=math.sqrt(5))
    elif method == 'fill_zero':
        # replace zeroed spatial weights with non-zero values using two methods.
        # first, for kernels that have some values zero, just sample the mean
        # and variance of that kernel and replace those missing values with gaussian noise.
        # second, for the kernels that are entirely zero, reinitialize weights with kaiming uniform values.
        tmp = conv2d.weight.detach().clone()
        # 1. fill the zeroed weights of kernels that are not entirely zero with random values
        tmp2 = conv2d.weight.detach().clone()
        tmp2 = T.nn.init.normal_(tmp2)*tmp.std((-1,-2), keepdim=True) + tmp.mean((-1,-2), keepdim=True)
        mask = T.isclose(tmp, T.tensor(0., dtype=tmp.dtype, device=tmp.device))
        tmp[mask] = tmp2[mask]
        #  2. fill the kernels that are entirely zero with kaiming uniform values
        mask = T.isclose(tmp, T.tensor(0., dtype=tmp.dtype, device=tmp.device))
        if mask.sum():
            tmp2 = T.nn.init.kaiming_uniform_(tmp2, a=math.sqrt(5))
            tmp[mask] = tmp2[mask]
        filters = tmp
        del tmp2, mask, tmp
    elif method == 'unchanged':
        filters = conv2d.weight  # do nothing.  preserve default initialization
    elif method == 'identity':
        h,w = conv2d.weight.shape[-2:]
        r = T.zeros_like(conv2d.weight)
        r[..., h//2, w//2] = 1
        filters = r
    elif method == 'DCT2':
        O,I, H,W = conv2d.weight.shape
        # initialize filters to basis vectors.
        dct_basis = T.tensor(kernel.dct_basis_2d(H, W), device=device, dtype=T.float)
        if I == 1:  # more efficiently handle grouped convolution by computing all at once
            rand_numbers = T.randperm(O*I).reshape(O, I) % (H*W)
            filters = dct_basis[rand_numbers]
            del rand_numbers
        else:  # for each output, give a set of basis vectors
            O,I, H,W = conv2d.weight.shape
            idxs = T.vstack([T.randperm(I)%(H*W) for o in range(O)])
            filters = dct_basis[idxs]
            del idxs
        del dct_basis
    elif method.startswith('DCT2steering:'):
        filename = method.split(':')[1]
        dct = T.load(f'dct2steering/{filename}.pth', map_location=device)
        O, I, H, W = conv2d.weight.shape
        F = conv2d.weight.detach().reshape(O*I, H*W)
        R = T.randn((O*I, H*W), dtype=T.float, device=device)
        mu = dct[layer_idx]['mu']
        std = dct[layer_idx]['std']
        B = T.tensor(kernel.dct_basis_2d(H, W), device=device, dtype=T.float)
        assert B.shape == (H*W, H, W)
        B = B.reshape(H*W,H*W)
        # assume R = XB.T ==> solve for X.
        filters = ((R * std.reshape(1, -1) + mu.reshape(1, -1)) @ B).reshape(O,I,H,W)
    elif method == 'DCT2fill0':
        # for every input channel, set the minimum possible number of output filters (H*W) to the basis
        # filters, and set all other filters to 0.
        O, I, H, W = conv2d.weight.shape
        filters = T.zeros_like(conv2d.weight)
        basis = T.tensor(kernel.dct_basis_2d(H, W), device=device, dtype=T.float)
        for i in range(I):
            random_slice = np.s_[T.randperm(O)[:H*W],i]
            filters[random_slice] = basis
    elif method.startswith('GuidedSteer'):
        filename = method.split(':')[1]
        # SVDsteering, guided per-layer.  SVDsteering by default uses basis derived from all spatial filters, but the steering values are layer specific.  This is truly layerwise
        if method.startswith('GuidedSteerC:'):
            dct = T.load(f'svdsteering_centered/{filename}.pth', map_location=device)
        else:
            dct = T.load(f'svdsteering_noncentered/{filename}.pth', map_location=device)
        dct = dct['layer_params'][layer_idx]
        O, I, H, W = conv2d.weight.shape
        if method.startswith('GuidedSteerU1'):
            a,b = 0,-1
        elif method.startswith('GuidedSteerU2'):
            a,b = 0,-2
        elif method.startswith('GuidedSteerU3'):
            a,b = 0,-3
        elif method.startswith('GuidedSteerV1'):
            a,b = 1,dct['Bu'].shape[0]  # remove approx image
        elif method.startswith('GuidedSteerP:'):
            # select first N principle components, based on pct variance explained by first N principle components of SVD
            num_components = (float(method.split(':')[-1])/100>= dct['pct_variance']).sum()
            a,b = 0,max(1, num_components)
            #  print(b, float(method.split(':')[-1])/100)
            assert b > a, 'sanity check'
            del num_components
        else:
            a,b = 0,dct['Bu'].shape[0]

        filters = (T.from_numpy(dct['kde'].resample(O*I).T).float().to(device)[:,a:b] @ dct['Bu'][a:b]
                   + dct['Fmean']).reshape(O,I,H,W)
        del a,b
    elif method.startswith('SVDsteering'):
        if method.startswith('SVDsteeringC:'):
            filename = method.split(':')[1]
            #  print('LOADING', f'svdsteering/{filename}.pth')
            dct = T.load(f'svdsteering_centered/{filename}.pth', map_location=device)
        elif method.startswith('SVDsteeringNC:'):
            filename = method.split(':')[1]
            #  print('LOADING', f'svdsteering/{filename}.pth')
            dct = T.load(f'svdsteering_noncentered/{filename}.pth', map_location=device)
        elif method == 'SVDsteering_b7':
            # blindly assume the filter is either 3x3 or 5x5 and fail otherwise
            dct = T.load('svdsteering_noncentered/efficientnet-b7.pth', map_location=device)
        elif method == 'SVDsteering_avg':
            dct = T.load('svdsteering_avg.pth', map_location=device)
        else:
            raise NotImplementedError(f'unrecognized SVDsteering method: {method}')
        O, I, H, W = conv2d.weight.shape
        F = conv2d.weight.detach().reshape(O*I, H*W)
        Bu = dct[(H,W)]['Bu']
        if method == 'SVDsteering_b7':
            _mean = lambda x: sum(x) / len(x)
            mean = _mean(dct[(H,W)]['mean_per_layer'].values())
            std = _mean(dct[(H,W)]['std_per_layer'].values())
        else:
            mean = dct[(H,W)]['mean_per_layer'][layer_idx]
            std = dct[(H,W)]['std_per_layer'][layer_idx]
            kde = dct[(H,W)]['kde_per_layer'][layer_idx]
        if method.endswith(':kde'):
            filters = (T.from_numpy(kde.resample(O*I).T).float().to(device) @ Bu
                       + dct[(H,W)]['Fmean']).reshape(O,I,H,W)
        else:
            Fmean = dct[(H,W)]['Fmean']
            num_components = Bu.shape[0]
            random_steering_weights = mean+std*T.randn(
                (O*I, num_components), dtype=T.float, device=device)
            filters = ((random_steering_weights @ Bu) + Fmean).reshape(O, I, H, W)
            del Fmean, num_components, random_steering_weights
        del F, dct, Bu, mean, std, kde
    else:
        raise NotImplementedError(f'unrecognized fixed filter method: {method}')

    if isinstance(filters, list):
        filters = T.stack(filters).unsqueeze_(1).reshape(conv2d.weight.shape).to(device)

    filters = T.nn.Parameter(
        filters, requires_grad=False)
    assert filters.shape == conv2d.weight.shape, 'code bug: unexpected shape mismatch'
    conv2d.weight = filters

    conv2d.requires_grad_(requires_grad)
    return conv2d


def spatial_conv2d_drop_in_replacement(model: T.nn.Module, replace_with: T.nn.Module, **extra_kws):
    """Replace Conv2d with another module that serves as a drop in replacement

    :model: pytorch model with conv2d layers
    :replace_with: a drop-in replacement for T.nn.Conv2d
    :spatial_only: if True, only consider conv2d of kernel size (a,b) where a,b>1
    :**extra_kws: any extra keyword arguments are used to initialize the `replace_with` class.
    """
    recurse_on_these = []
    for name, conv2d in model.named_children():
        if not isinstance(conv2d, T.nn.Conv2d):
            recurse_on_these.append(conv2d)
            continue
        if conv2d.kernel_size[0] <= 1 or conv2d.kernel_size[1] <= 1:
            continue  # applies only to spatial convolutions
        defaults = dict(
            in_channels=conv2d.in_channels, out_channels=conv2d.out_channels,
            kernel_size=conv2d.kernel_size, stride=conv2d.stride,
            padding=conv2d.padding, dilation=conv2d.dilation,
            groups=conv2d.groups, bias=conv2d.bias,
            padding_mode=conv2d.padding_mode,
        )
        defaults.update(extra_kws)
        new_conv2d = replace_with(**defaults).to(conv2d.weight.device)
        if isinstance(conv2d, Conv2dStaticSamePadding):
            # workaround for efficientnet
            new_conv2d = T.nn.Sequential(
                conv2d.static_padding,
                new_conv2d)
        elif issubclass(conv2d.__class__, T.nn.Conv2d) and conv2d.__class__ != T.nn.Conv2d:
            print(
                f"WARNING: converted an instance of {conv2d.__class__} that"
                f" inherits from conv2d to a {replace_with}.  This might cause"
                " bugs.")
        setattr(model, name, new_conv2d)
    # --> recursive through child modules.
    for child_module in recurse_on_these:
        spatial_conv2d_drop_in_replacement(
            child_module, replace_with, **extra_kws)
    return model


def convert_conv2d_to_gHaarConv2d(model: T.nn.Module):
    """Replace Conv2d with kernel.GHaarConv2d
    adapted from https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736/8

    This is the earlier version of `conv2d_drop_in_replacement`
    and is still here because kernel.GHaarConv2d doesn't support padding_mode.
    """
    recurse_on_these = []
    for attr_name, conv2d in model.named_children():
        if not isinstance(conv2d, T.nn.Conv2d):
            recurse_on_these.append(conv2d)
            continue
        if conv2d.kernel_size[0] <= 1 or conv2d.kernel_size[1] <= 1:
            continue
        new_conv2d = kernel.GHaarConv2d(
            conv2d.in_channels, conv2d.out_channels, conv2d.kernel_size,
            stride=conv2d.stride, padding=conv2d.padding,
            dilation=conv2d.dilation, groups=conv2d.groups, bias=conv2d.bias
        ).to(conv2d.weight.device)
        if isinstance(conv2d, Conv2dStaticSamePadding):
            # workaround for efficientnet
            new_conv2d = T.nn.Sequential(
                conv2d.static_padding,
                new_conv2d)
        elif issubclass(conv2d.__class__, T.nn.Conv2d) and conv2d.__class__ != T.nn.Conv2d:
            print(
                f"WARNING: converted an instance of {conv2d.__class__}that inherits from conv2d to"
                " a GHaarConv2d.  This might cause bugs.")
        setattr(model, attr_name, new_conv2d)
    # --> recursive through child modules.
    for child_module in recurse_on_these:
        convert_conv2d_to_gHaarConv2d(child_module)
    return model

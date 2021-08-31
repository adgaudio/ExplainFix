from efficientnet_pytorch import EfficientNet
from pampy import match
from third_party.DeepLabV3Plus_Pytorch.network import deeplabv3plus_resnet50, deeplabv3plus_mobilenet
import torch as T
import torchvision.models as tvm
from dw2 import kernel
from dw2 import models as M


def get_model_modes(model:T.nn.Module, model_name:str):
    return (
        'unmodified_baseline', lambda _: model,

        'spatial_100%_psine', lambda _: M.convert_all_spatial_conv2d(model, False, 'polynomial_sin_ND'),
        'spatial_100%_haar', lambda _: M.convert_all_spatial_conv2d(model, False, 'haar'),
        'spatial_100%_ghaarA', lambda _: M.convert_all_spatial_conv2d(model, False, 'ghaarA'),
        'spatial_100%_ones', lambda _: M.convert_all_spatial_conv2d(model, False, 'ones'),
        'spatial_100%_unchanged', lambda _: M.convert_all_spatial_conv2d(model, False, 'unchanged'),
        'spatial_100%_kuniform', lambda _: M.convert_all_spatial_conv2d(model, False, 'kaiming_uniform'),
        'spatial_100%_DCT2', lambda _: M.convert_all_spatial_conv2d(model, False, 'DCT2'),
        'spatial_100%_GuidedSteer', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteer:{model_name}'),
        'spatial_100%_FillZero', lambda _: M.convert_all_spatial_conv2d(model, False, 'fill_zero'),
        'spatial_100%_identity', lambda _: M.convert_all_spatial_conv2d(model, False, 'identity'),

        #  'spatial_100%_poly', lambda _: M.convert_all_spatial_conv2d(model, requires_grad=False, method='polynomial_ND'),
        #  'spatial_100%_ghaar4', lambda _: M.convert_all_spatial_conv2d(model, False, 'ghaar4'),
        #  'spatial_100%_ghaar', lambda _: M.convert_all_spatial_conv2d(model, False, 'ghaar'),
        #  'spatial_100%_ghaarN', lambda _: M.convert_all_spatial_conv2d(model, False, 'ghaarN'),
        #  'spatial_100%_GHaar4.s1', lambda _: M.convert_all_spatial_conv2d(model, False, 'GHaar4.s1'),
        #  'spatial_100%_GHaar4.s2', lambda _: M.convert_all_spatial_conv2d(model, False, 'GHaar4.s2'),
        #  'spatial_100%_GHaar2.ms', lambda _: M.convert_all_spatial_conv2d(model, False, 'GHaar2.ms'),
        #  'spatial_100%_GHaar4.ms', lambda _: M.convert_all_spatial_conv2d(model, False, 'GHaar4.ms'),
        #  'spatial_100%_GHaarR.ms', lambda _: M.convert_all_spatial_conv2d(model, False, 'GHaarR.ms'),
        #  'spatial_100%_PsineR.ms', lambda _: M.convert_all_spatial_conv2d(model, False, 'PsineR.ms'),

        #  'spatial_100%_DCT2steering', lambda _: M.convert_all_spatial_conv2d(model, False, f'DCT2steering:{model_name}'),
        #  'spatial_100%_DCT2fill0', lambda _: M.convert_all_spatial_conv2d(model, False, 'DCT2fill0'),
        #  'spatial_100%_DCT2min', lambda _: M.convert_all_spatial_conv2d(model, False, 'DCT2min'),

        #  'spatial_100%_SVDsteering', lambda _: M.convert_all_spatial_conv2d(model, False, f'SVDsteering:{model_name}'),
        #  'spatial_100%_SVDsteering_avg', lambda _: M.convert_all_spatial_conv2d(model, False, 'SVDsteering_avg'),
        #  'spatial_100%_SVDsteering_b7', lambda _: M.convert_all_spatial_conv2d(model, False, 'SVDsteering_b7'),
        #  'spatial_100%_SVDsteeringC', lambda _: M.convert_all_spatial_conv2d(model, False, f'SVDsteeringC:{model_name}'),
        #  'spatial_100%_SVDsteeringNC', lambda _: M.convert_all_spatial_conv2d(model, False, f'SVDsteeringNC:{model_name}'),
        #  'spatial_100%_SVDsteeringNC_kde', lambda _: M.convert_all_spatial_conv2d(model, False, f'SVDsteeringNC:{model_name}:kde'),
        #  'spatial_100%_GuidedSteerSome', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteer:{model_name}'),
        #  'spatial_100%_GuidedSteerC', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerC:{model_name}'),
        #  'spatial_100%_GuidedSteerU1', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerU1:{model_name}'),
        #  'spatial_100%_GuidedSteerU2', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerU2:{model_name}'),
        #  'spatial_100%_GuidedSteerU3', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerU3:{model_name}'),
        #  'spatial_100%_GuidedSteerV1', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerV1:{model_name}'),
        #  'spatial_100%_GuidedSteerP99', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerP:{model_name}:99'),
        #  'spatial_100%_GuidedSteerP999', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerP:{model_name}:99.9'),
        #  'spatial_100%_GuidedSteerP9999', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerP:{model_name}:99.99'),
        #  'spatial_100%_GuidedSteerP95', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerP:{model_name}:95'),
        #  'spatial_100%_GuidedSteerP90', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerP:{model_name}:90'),
        #  'spatial_100%_GuidedSteerP80', lambda _: M.convert_all_spatial_conv2d(model, False, f'GuidedSteerP:{model_name}:80'),

        # just using haar filters as an initialization, but learning them normally.
        #  'unmodified_fixed1x1', lambda _: M.convert_all_spatial_conv2d(model, True, 'unchanged', fix_1x1convs=True),
        #  'learned_haar_fixed1x1', lambda _: M.convert_all_spatial_conv2d(model, True, 'haar', fix_1x1convs=True),
        #  'learned_haar_learned1x1', lambda _: M.convert_all_spatial_conv2d(model, True, 'haar', fix_1x1convs=False),

        #  'spatial_75%_psine', lambda _: M.convert_some_randomly(model, .5, False, 'polynomial_sin_ND'),
        #  'spatial_50%_psine', lambda _: M.convert_some_randomly(model, .5, False, 'polynomial_sin_ND'),
        #  'spatial_10%_psine', lambda _: M.convert_some_randomly(model, .1, False, 'polynomial_sin_ND'),
        #  'spatial_1%_psine', lambda _: M.convert_some_randomly(model, .01, False, 'polynomial_sin_ND'),

        #  'unet_v1', lambda _: M.convert_all_spatial_conv2d(
            #  model, requires_grad=False, method='polynomial_ND'),
        #  'unet_v2', lambda _: M.convert_all_spatial_conv2d(
            #  model, requires_grad=True, method='polynomial_ND'),
        #  'unet_v3', lambda _: M.convert_all_spatial_conv2d(
            #  model, requires_grad=False, method='polynomial_sin_ND'),
        #  'unet_v4', lambda _: M.convert_all_spatial_conv2d(
            #  model, requires_grad=True, method='polynomial_sin_ND'),
        #  'unet_v5', lambda _: M.convert_all_spatial_conv2d(  # like unmodified, but fixed filters
            #  model, requires_grad=False, method='unchanged'),
        #  'unet_v6', lambda _: M.convert_all_spatial_conv2d(  # unmodified model
            #  model, requires_grad=True, method='unchanged'),
        #  'unet_v7', lambda _: M.convert_all_spatial_conv2d(
            #  model, requires_grad=False, method='haar'),
        #  'unet_v8', lambda _: M.convert_all_spatial_conv2d(
            #  model, requires_grad=True, method='haar'),
    )


def _get_efficientnet(name):
    name, num_classes, pretrained = name.rsplit(':', 2)
    num_classes = int(num_classes)
    pretrained = match(pretrained, 'pretrained', True, 'fromscratch', False)
    kws = dict(in_channels=1, num_classes=num_classes, )
    if pretrained:
        return EfficientNet.from_pretrained(name, **kws, advprop=True)
    else:
        return EfficientNet.from_name(name, **kws)


def _get_densenet(name):
    name, num_classes, pretrained = name.rsplit(':', 2)
    num_classes = int(num_classes)
    pretrained = match(pretrained, 'pretrained', True, 'fromscratch', False)
    mdl = getattr(tvm, name)(pretrained=pretrained, )
    #  mdl = tvm.densenetXXX(pretrained=True, )
    mdl.classifier = T.nn.Linear(mdl.classifier.in_features, num_classes)
    _copy = lambda conv, keys: {key: getattr(conv, key) for key in keys}
    mdl.features[0] = T.nn.Conv2d(
        in_channels=1, **_copy(mdl.features[0], [
            'out_channels', 'kernel_size', 'stride', 'padding', 'bias']))
    return mdl


def _get_resnet(name):
    name, num_classes, pretrained = name.rsplit(':', 2)
    num_classes = int(num_classes)
    pretrained = match(pretrained, 'pretrained', True, 'fromscratch', False)
    mdl = getattr(tvm, name)(pretrained=pretrained, )
    mdl.fc = T.nn.Linear(mdl.fc.in_features, num_classes)
    _copy = lambda conv, keys: {key: getattr(conv, key) for key in keys}
    mdl.conv1 = T.nn.Conv2d(
        in_channels=1, **_copy(mdl.conv1, [
            'out_channels', 'kernel_size', 'stride', 'padding', 'bias']))
    return mdl


_other = {
    # not deep classifier models
    'haarlogistic:14': lambda _: M.WaveletLinearModel('haar', out_ch=14),
    'dualtreelogistic:14': lambda _: M.DualTreeWaveletLinearModel(out_ch=14),
    'scatteringlogistic:14': lambda _: M.ScatteringLinearModel(out_ch=14),
    'haarlogistic:5': lambda _: M.WaveletLinearModel('haar', out_ch=5),
    'dualtreelogistic:5': lambda _: M.DualTreeWaveletLinearModel(out_ch=5),
    'scatteringlogistic:5': lambda _: M.ScatteringLinearModel(out_ch=5),
}
_other.update({
    # segmentation models
    'unet_pytorch_hub': lambda _: M.UnetPytorchHub(),
    #  'unet': lambda _: M.Unet((3,8,16,32,64, 128, 256, 512, 1024), tail=T.nn.Conv2d(3,1,1, bias=False), depthwise_channel_multiplier=6),
    'unetD_small': lambda _: M.Unet((3,8,16,32,64), tail=T.nn.Conv2d(3,1,1, bias=False), depthwise_channel_multiplier=6),
    #  'unetD_small_ghaar_learnable': lambda _: M.UnetHybrid((3,8,16,32,64), tail=T.nn.Conv2d(3,1,1, bias=False), depthwise_channel_multiplier=6, spatial_conv=kernel.GHaarConv2d),
    #  'unet_small', lambda _: M.Unet((3,8,16,32,64, 128), tail=T.nn.Conv2d(3,1,1, bias=False), depthwise_channel_multiplier=6),
    'deeplabv3plus_resnet50': lambda _: deeplabv3plus_resnet50(num_classes=1, pretrained_backbone=False),
    'deeplabv3plus_mobilenet': lambda _: deeplabv3plus_mobilenet(num_classes=1, pretrained_backbone=False),
})


def get_model(name, mode, device):
    if name.startswith('efficientnet'):
        model = _get_efficientnet(name)
    elif name.startswith('densenet'):
        model = _get_densenet(name)
    elif name.startswith('resnet'):
        model = _get_resnet(name)
    else:
        assert name in _other, f"Unrecognized model name: {name}"
        model = _other[name](name)
    model.to(device)
    # modify model according to the chosen mode.
    model = match(mode, *get_model_modes(model, name))
    return model

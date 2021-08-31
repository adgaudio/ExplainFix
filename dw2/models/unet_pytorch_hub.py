import torch as T


class UnetPytorchHub(T.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        defaults = dict(
            in_channels=3, out_channels=1, init_features=32, pretrained=False)
        defaults.update(kwargs)
        self.unet = T.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet', **defaults)

    def forward(self, x):
        # ensure input and output image size is correct
        h,w = x.shape[-2:]
        if (h,w) != (256,256):
            x = T.nn.functional.interpolate(x, (256,256), mode='bilinear', align_corners=False)
        # run unet
        x = self.unet(x)
        # restore original filesize
        if (h,w) != (256,256):
            x = T.nn.functional.interpolate(x, (h,w), mode='bilinear', align_corners=False)
        x = x[...,:h,:w]
        return x

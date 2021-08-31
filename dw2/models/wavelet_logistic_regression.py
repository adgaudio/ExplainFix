import pywt
import pytorch_wavelets as tw
import torch as T


def center_crop(arr, out_y, out_x):
    """assume input array is LARGER or equal in shape to output array"""
    in_y, in_x = arr.shape[-2:]
    dy = in_y - out_y if in_y > out_y else 0
    dx = in_x - out_x if in_x > out_x else 0
    arr = arr[...,
        dy//2: in_y-dy//2-dy%2,
        dx//2: in_x-dx//2-dx%2]
    assert in_y >= out_y
    assert in_x >= out_x
    assert arr.shape[-2:] == (out_y, out_x), arr.shape
    return arr


class DualTreeWaveletLinearModel(T.nn.Module):
    def __init__(self, encoder=tw.DTCWT(J=8), out_ch=14, input_size=(320,320)):
        super().__init__()
        self.encoder = tw.DTCWT(J=8)
        self.classifier = T.nn.Linear(409672, out_ch)
        self.input_size = input_size
    def forward(self, x):
        if x.shape[-2:] != self.input_size:
            x = center_crop(x, *self.input_size)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    def flatten(self, x):
        assert isinstance(x, tuple)
        assert len(x) == 2
        assert isinstance(x[0], T.Tensor)
        assert isinstance(x[1], list)
        assert all(isinstance(y, T.Tensor) for y in x[1])
        batch_size = x[0].shape[0]
        flat = [x[0].reshape(batch_size, -1)]
        flat.extend([y.reshape(batch_size, -1) for y in x[1]])
        flat = T.cat(flat, 1)
        return flat


class ScatteringLinearModel(T.nn.Module):
    def __init__(self, input_size=(320, 320), out_ch=14):
        super().__init__()
        self.encoder = tw.ScatLayerj2()
        self.input_size = input_size
        self.linear = T.nn.Linear(313600, out_ch)
    def forward(self, x):
        if x.shape[-2:] != self.input_size:
            x = center_crop(x, *self.input_size)
        x = self.encoder(x)
        batch_size = x.shape[0]
        x = self.linear(x.reshape(batch_size, -1))
        return x


class WaveletLinearModel(T.nn.Module):
    """A basic model that center crops input to 320x320 images"""
    def __init__(self, wavelet:pywt.Wavelet='haar', J=8, input_size=(320,320), sigmoid_at_end=False,
                 out_ch=14):
        super().__init__()
        self.input_size = input_size
        self.dwt = tw.DWT(
            J=J,  # pywt.dwt_max_level(img_side_length, wavelet),
            wave=wavelet, mode='zero')
        self.linear = T.nn.Linear(102418, out_ch)
        self.sigmoid_at_end = sigmoid_at_end

    def forward(self, x):
        if x.shape[-2:] != self.input_size:
            x = center_crop(x, *self.input_size)
        assert x.shape[-2:] == self.input_size, "sanity check"  # sanity check assumes input always larger than given input size...
        batch_size = x.shape[0]
        # discrete wavelet transform
        coeffs = self.dwt(x)  # get wavelet coefficients
        # --> flatten wavelet coefficients
        x = [coeffs[0].reshape(batch_size, -1)]
        x.extend(x.reshape(batch_size, -1) for x in coeffs[1])
        x = T.cat(x, 1)
        assert x.shape == (batch_size, 102418), f'sanity check {x.shape}'
        # linear regression
        x = self.linear(x)
        if self.sigmoid_at_end:
            x = T.sigmoid(x)  # just remember to do it yourself during loss or eval steps, or use a threshold of 0.
        return x

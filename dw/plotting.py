from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import PIL.ImageDraw
import scipy.ndimage as ndi
import scipy.stats as stats
from typing import Iterable, Tuple
import joblib
import torch as T

from dw.wavelet import wavelet_coefficients_as_tensorimage


def tolist(val, N):
    if not isinstance(val, (list, tuple)):
        val = [val] * N
    return val


def plot_img_grid(imgs: Iterable, suptitle:str = '', rows_cols: Tuple = None,
                  norm=None, vmin=None, vmax=None, cmap=None,
                  convert_tensor:bool=True, num:int=None,
                  ax_titles:Tuple[str]=None):
    """Plot a grid of n images

    :imgs: a numpy array of shape (n,h,w) or a list of plottable images
    :suptitle: figure title
    :convert_tensor: if True (default), try to convert pytorch tensor
        to numpy.  (Don't try to convert channels-first to channels-last).
    :vmin: and :vmax: and :norm: are passed to ax.imshow(...).  if vmin or vmax
        equal 'min' or 'max, respectively, find the min or max value across all
        elements in the input `imgs`.  vmin, vmax or norm can also each be a
        list, with one value per image in the figure.
    :cmap: a matplotlib colormap
    :num: the matplotlib figure number to use
    """
    if rows_cols is None:
        _n = np.sqrt(len(imgs))
        rows_cols = [int(np.floor(_n)), int(np.ceil(_n))]
        if np.prod(rows_cols) < len(imgs):
            rows_cols[0] = rows_cols[1]
    elif rows_cols[0] == -1:
        rows_cols = list(rows_cols)
        rows_cols[0] = int(np.ceil(len(imgs) / rows_cols[1]))
    elif rows_cols[1] == -1:
        rows_cols = list(rows_cols)
        rows_cols[1] = int(np.ceil(len(imgs) / rows_cols[0]))
    assert np.prod(rows_cols) >= len(imgs), (rows_cols, len(imgs))
    if vmin == 'min':
        vmin = min([x.min() for x in imgs])
    if vmax == 'max':
        vmax = max([x.max() for x in imgs])
    if convert_tensor:
        imgs = (x.cpu().numpy() if isinstance(x, T.Tensor) else x for x in imgs)
    fig, axs = plt.subplots(
        *rows_cols, squeeze=False, figsize=np.multiply((rows_cols[1],rows_cols[0]),2),
        tight_layout=dict(w_pad=0.1, h_pad=0.1), num=num, clear=True)
    #  fig.set_constrained_layout_pads(w_pad=0, h_pad=0, hspace=0., wspace=0.)
    fig.suptitle(suptitle)
    [ax.axis('off') for ax in axs.ravel()]
    if ax_titles is None:
        ax_titles = [None] * len(fig.axes)
    norm = tolist(norm, len(fig.axes))
    vmin = tolist(vmin, len(fig.axes))
    vmax = tolist(vmax, len(fig.axes))
    for zimg, ax, ax_title, norm, vmin, vmax in zip(imgs, axs.flatten(), ax_titles, norm, vmin, vmax):
        ax.imshow(zimg, norm=norm, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(ax_title)
    return fig


def plot_dwt2(approx, detail, ax=None, normalize=False, lognorm=False):
    """Plot all levels of a wavelet decomposition as a single image.
    Assume input is the result of pytorch_wavelets.DWT.

    :approx: (hxw) tensor.  Approximation coefficients for the Nth level of wavelet transform
        If approx=None, then ignore it.
    :detail: list of pytorch tensors. Detail coefficients for all levels of the wavelet
      transform. Each tensor has shape (3,h,w) corresponding to horizontal,
      vertical, diagonal detail.
      The lowest frequency (finest detail, largest matrices) coefficients first, and largest frequency coefficients last.
    :normalize: bool.  If True, 0-1 normalize each matrix independently
    :returns: TODO
    """
    if ax is None:
        fig, ax = plt.subplots(1,1)
    im = wavelet_coefficients_as_tensorimage(approx, detail, normalize)
    im = im.squeeze(0).cpu().numpy()
    # --> normalize for plotting into -1,1
    #  im = (im - im.min()) / im.ptp() * 2 - 1
    # --> color negative values red, positive values green
    r,g,b = np.abs(im), np.abs(im), np.abs(im)
    r[im >= 0] = 0
    g[im <= 0] = 0
    b[:] = 0
    rgb = np.dstack([r,g,b]).clip(0,1)
    if lognorm:
        rgb = np.log(rgb*255 + 1) / np.log(256)
    # don't mess up colors or apply lognorm on the approx image.
    rgb[:approx.shape[-2], :approx.shape[-1], :] = \
            approx.cpu().numpy().squeeze(0).transpose(-2, -1, -3)
    ax.imshow(rgb)
    return ax


def imgrid(lst_of_imgs, fig=None):
    N = len(lst_of_imgs)
    rows, cols = np.floor(np.sqrt(N)).astype('int'), np.ceil(np.sqrt(N)).astype('int')
    if fig is None:
        fig = plt.figure()
    grid = ImageGrid(fig, 111, (rows, cols))
    for g, im in zip(grid, lst_of_imgs):
        g.imshow(im)
    return fig

def make_circle_img(shape=(100, 100), center_yx=(0,0),
                    y_gradient=False, x_gradient=False, circle_fracs=[.8]):
    N = np.prod(shape)
    im = np.zeros(shape, dtype='float')
    if y_gradient:
        im += np.arange(N).reshape(*shape) / N
    if x_gradient:
        im += np.arange(N).reshape(*shape).T / N
        if x_gradient and y_gradient:
            im /= 2
    y, x = shape
    for n, frac in enumerate(circle_fracs):
        # using sine waves
        #  cy,cx = int(y*frac), int(x*frac)
        #  oy, ox = int((y - cy)//2), int((x - cx)//2)
        #  xdat = np.linspace(0, np.pi, cy)
        #  ydat = np.linspace(0, np.pi, cx)
        #  circle_img = 1.0*(
                #  np.outer(np.sin(ydat), np.cos(xdat-np.pi/2)) > .4)
        #  slice = np.s_[oy:oy+cy, ox:ox+cx]
        #  if n%2 == 0:
            #  im[slice] += circle_img
        #  else:
            #  im[slice] -= circle_img
        # using normal distribution (circles are less square)
        y = stats.norm.pdf(np.linspace(-1, 1, shape[0]), center_yx[0], .5)
        x = stats.norm.pdf(np.linspace(-1, 1, shape[1]), center_yx[1], .5)
        circle_img = np.outer(y, x)
        circle_img = 1-((circle_img - circle_img.min()) / circle_img.ptp())
        circle_img = circle_img > frac
        insert_img_op = np.add if n%2==0 else np.subtract
        im = insert_img_op(im, (circle_img > frac)*1.0)
    return im


def make_polygon_img(shape=(100, 100), points='randomtriangle'):
    """
    :points: array of shape (3, 2).  Each column is the y,x coord.  Each row is
    a corner of the polygon.
    """
    Y, X = shape
    if isinstance(points, str) and points == 'randomtriangle':
        points = np.dstack([
            np.random.randint(0, Y, 3), np.random.randint(0, X, 3)])[0]
    im = PIL.Image.new('L', (X, Y), 'white')
    PIL.ImageDraw.Draw(im).polygon(list(np.ravel(points)) + list(np.ravel(points)[:2]), fill='black')
    return np.array(im) / 255


def plot_scatter_coeffs_polar(*scatter_coeffs, normalize='per_array',
                              cmaps=None, fignum=None):
    """Generate a grid of subplots, where each subplot shows the coefficients
    of a 2D wavelet scattering transform.  Intended to visualize what the
    "identity" of a transform looks like.

    The "Invariant Scattering Convolution Networks" paper first created plots
    like these.

    Each subplot is as a nested pie chart (ie. a bar plot in polar coords), where
    the transform scale and rotation correspond to levels and angle in the pie
    chart, respectively.

    :scatter_coeffs: a tuple of numpy arrays of shape (J_i, R_i, S_i, H, W).
    Each i'th array corresponds to a set of scatter coefficients with J_i
    scales, R_i rotations and S_i aggregating statistics.  The HxW matrix
    contains the coefficients, and is fixed for all i arrays. Each element in
    the (HxW) matrix of any given array represents a new subplot, and that
    subplot contains (J_i*R_i*S_i) datapoints.  If you pass in three numpy
    arrays of shape (J,R,S,2,2), you have 2*2*3 subplots.

    :normalize: one of ['global', 'per_array', 'per_subplot'].  This is about
    how to colorize the brightest (max) and darkest (min) coefficients.  Choose
    'global' to compare differences in coefficient values between arrays.
    Choose 'per_array' to compare subplots within an array.  Choose
    'per_subplot' or 'per_stat' to compare values inside each subplot
    independently, where 'per_stat' also normalizes each stat independently
    (useful if the stats have different number ranges).
    Default is 'per_stat'.
    :cmaps:  list of matplotlib color maps to use when showing the coefficients.
    Each group of stats gets a different colormap if there is more than one map.

    Note, the circumference of circle gets uniformly partitioned so each
    rotation has the same arc length.  If two rotations used in the transform
    were actually close together or all angles spanned less than 360 degrees,
    this function ignores that fact.  The reason is each plot should be used to
    visually explain the discriminative features.
    """
    if cmaps is None:
        cmaps = [plt.cm.autumn, plt.cm.spring, plt.cm.summer, plt.cm.winter]
    H, W = scatter_coeffs[0].shape[-2:]
    N = len(scatter_coeffs)
    # set up the figure
    fig, axs = plt.subplots(
        H, W*N, subplot_kw=dict(projection='polar'), squeeze=False, num=fignum)
    fig.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1)
    [x.axis('off') for x in axs.ravel()]
    # show line separators to distinguish between the arrays in scatter_coeffs
    _ax = fig.add_subplot(1,1,1, xlim=(0,1), ylim=(0,1))
    _ax.vlines(np.arange(1, N)/N, 0, 1, colors='gray')
    _ax.axis('off')

    # for visualization, renormalize the scatter coefficients to be in 0-1,
    # using global min and max.
    if normalize == 'global':
        min_ = min(x.min() for x in scatter_coeffs)
        max_ = max(x.max() for x in scatter_coeffs)
        scatter_coeffs = [(x-min_)/(max_-min_) for x in scatter_coeffs]

    for n, sc_arr in enumerate(scatter_coeffs):
        if normalize == 'per_array':
            sc_arr = (sc_arr-sc_arr.min()) / sc_arr.ptp()
        J, R, S, _H, _W = sc_arr.shape
        assert _H == H
        assert _W == W
        for h in range(H):
            for w in range(W):
                ax = axs[h,W*n+w]
                subplot_scatter_coeffs(
                    coeffs=sc_arr[:,:,:,h,w], ax=ax, normalize=normalize,
                    cmaps=cmaps
                )
    return fig


def subplot_scatter_coeffs(coeffs, ax=None,
                           cmaps=None,
                           normalize='per_stat'):
    """
    Helper function to plot 2D wavelet scattering coefficients for a fixed set
    of scales and rotations used in the transform.  You may want to use
    `plot_scatter_coeffs_polar` instead.

    The plot is a nested pie chart (ie. a bar plot in polar coords), where
    the scale and rotation correspond to levels and angle in the pie
    chart, respectively.

    Assume the angles are assumed uniformly distributed over 360 degrees.
    Uniformly distribute the scales over the radius of the plotted circle.

    The Invariant Scattering Convolution Networks paper first created these
    pictures.  Code adapted from a Kymatio example.

    :coeffs: numpy array of shape (J, R, S) containing the scatter coefficients.
        J scales and R rotations and S stats
    :normalize: one of ['per_subplot', 'per_stat'].  By default, use per_stat.
    Ensure all `coeffs` are in [0,1].
    :ax: matplotlib PolarAxes object.
    :cmaps: alternate between these colormaps for each of the S stats.
    Generally, S%len(cmaps)!=1 to avoid two different stats sharing the same coloring
    """
    if cmaps is None:
        cmaps = [plt.cm.autumn, plt.cm.winter, plt.cm.cool],
    if ax is None:
        _, ax = plt.subplots()
    J, R, S = coeffs.shape
    # --> ensure that two neighboring stats never have the same colormap
    if S % len(cmaps) == 1:
        cmaps = cmaps[:-1]
    # --> form the position of each data point in polar coordinates.
    angle = np.linspace(0, 2*np.pi, R*S, endpoint=False)\
        .reshape(1, -1).repeat(J, 0).reshape(J,S,R).transpose(0,2,1)
    radius = np.arange(J, 0, -1).reshape(-1, 1).repeat(R*S, 1).reshape(J,R,S)
    arc_length = 2*np.pi / (R*S)  # uniform for all data points, covering half circle

    if normalize == 'per_subplot':
        coeffs = (coeffs - coeffs.min()) / coeffs.ptp()
    for s in range(S):
        co_ = coeffs[:, :, s]
        if normalize == 'per_stat':
            co_ = (co_ - co_.min()) / co_.ptp()
        # --> visualize the colors on log scale, which is better for human to see.
        # and ensure a different colormap for each stat
        z = np.log(co_.ravel()*255+1)
        color = cmaps[s%len(cmaps)]((z-z.min()) / z.ptp())
        # --> plot!
        ax.bar(
            x=angle[:,:,s].ravel(),
            height=radius[:,:,s].ravel(),
            width=arc_length, color=color)
    return ax


def plot_posneg(im, ax=None):
    rgb = np.zeros((k.shape[0], k.shape[1], 3))
    rgb[...,0][k < 0] = k[k<0]*-1
    rgb[...,1][k > 0] = k[k>0]
    if ax is None:
        ax = plt.subplots()
    ax.imshow((rgb-rgb.min())/rgb.ptp())
if __name__ == "__main__":
    import kymatio.numpy as kpyw
    import torch as T
    import pywt
    import pytorch_wavelets as pyw


    def example_plot_dwt2():
        #  im = make_circle_img()
        im = make_circle_img(circle_fracs=[1, 0, .8, .6,.0,.4,0,.2,0])

        wavelet = 'haar'

        nlevels = pywt.dwtn_max_level(im.shape, wavelet)
        #  nlevels = 3
        dwt = pyw.DWT(J=nlevels, wave=wavelet, mode='zero')
        small_approx, detail = dwt(T.tensor(im.reshape(1, 1, *im.shape)).float())

        plt.figure(0) ; plt.imshow(im, cmap='gray')
        plt.figure(1) ; plot_dwt2(None, detail, ax=plt.gca())


    def example_plot_scatter_coeffs_polar(X, start_fignum):
        R=8  # num rotations
        J=int(np.log2(min(X.shape)))  # max num scales
        #  J = 3
        J=int(np.log2(min(X.shape)))-1  # max num scales - 1
        S = kpyw.Scattering2D(J, X.shape, R, max_order=2)
        y = S(X)
        plt.figure(num=start_fignum) ; plt.imshow(X) ; plt.gca().axis('off')
        # show two groups of scattering coefficients.
        # The first group has 1 stat.  The second group has 5 stats.
        # The coloring of stats for this example is a bit canned because
        # wavelet scattering output only has 1 stat (the average)
        plot_scatter_coeffs_polar(
            y[1:J*R+1].reshape(J,R,1,*y.shape[-2:]),
            y[1:R*12+1:].reshape(3,R,4,*y.shape[-2:]),
            fignum=start_fignum+1)

    def _plot_scattering_for_imgs(imgs, suptitle, normalize='per_stat', cmaps=(plt.cm.autumn, )):
        fig = plt.figure()
        fig.suptitle(suptitle)
        N = len(imgs)
        for col_idx, im in enumerate(imgs):
            ax0 = fig.add_subplot(3, len(imgs), col_idx+1)
            ax1 = fig.add_subplot(3, len(imgs), col_idx+1+N, projection='polar')
            ax2 = fig.add_subplot(3, len(imgs), col_idx+1+2*N, projection='polar')
            ax0.imshow(im, interpolation='none')
            # choose scale J so output is only 1 subplot per order (H,W)==(1,1)
            R, J = 3, int(np.log2(min(im.shape)))
            y = kpyw.Scattering2D(J, im.shape, R, max_order=2)(im)
            subplot_scatter_coeffs(y[1:J*R+1].reshape(J,R,1), ax=ax1, normalize=normalize, cmaps=cmaps)
            subplot_scatter_coeffs(y[J*R+1:].reshape(int(J*(J-1)/2),R*R,1), ax=ax2, normalize=normalize, cmaps=cmaps)
        axs = fig.get_axes()
        [ax.axis('off') for ax in axs]
        axs[0].set_ylabel('Input Img')
        axs[1].set_ylabel('Scattering Coeffs (1st order)')
        axs[2].set_ylabel('Scattering Coeffs (2nd order)')
        fig.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1)
        return fig

    def example_translation_test(X):
        # note: ndi.shift is also suitable I think
        imgs = [
            # translation test  - rotation invariant
            X,
            np.roll(X, 12, 1),  # translated along x
            np.roll(X, 12, 0),  # translated along y
        ]
        _plot_scattering_for_imgs(imgs, 'Translation Invariance test')

    def example_rotation_test(X, cmaps):
        imgs = [
            # rotation test  - rotation covariant
            X,
            np.flipud(X),
            np.fliplr(X),
            ndi.rotate(X, 15),
            ndi.rotate(X, 5),
        ]
        _plot_scattering_for_imgs(imgs, 'Rotation Invariance test', cmaps=cmaps)

    def example_input_shift_test(cmaps):
        X = make_polygon_img(points=[20, 80, 80, 80, 20, 20])
        mask = (X != 1)
        assert X.min() == 0
        assert X.max() == 1
        a = X.copy()
        b = X.copy() ; b[0,0] = -.4 ; b[mask] -= .2
        c = X.copy() ; c[0,0] = 0 ; c[mask] += .2
        d = X.copy() ; d[0,0] = -255*2 ; d[mask] -=255
        e = X.copy() ; e[0,0] = 0 ; e[mask] += 255
        imgs = [a,b,c,d,e]
        _plot_scattering_for_imgs(imgs, 'Input Shift Invariance test', normalize='none', cmaps=cmaps)

    def example_inversion_test(X):
        assert X.min() == 0
        assert X.max() == 1
        imgs = [X, 1-X, X*-1]
        _plot_scattering_for_imgs(imgs, 'Input Inversion Invariance test')

    def example_scale_test(X, cmaps):
        M2 = np.linalg.inv(np.array([[1.5,0,-25], [0,1.5,-25], [0,0,1]]))
        M3 = np.linalg.inv(np.array([[1.5,0,-25], [0,1,0], [0,0,1]]))
        M4 = np.linalg.inv(np.array([[1,0,0], [0,1.5,-25], [0,0,1]]))
        M5 = np.linalg.inv(np.array([[.8,0,0], [0,.8,0], [0,0,1]]))
        imgs = [X] + [
            ndi.affine_transform(X, M, output_shape=X.shape, cval=1).clip(0,1)
            for M in [M2, M3, M4, M5]]
        _plot_scattering_for_imgs(imgs, 'Input Scale Invariance test', cmaps=cmaps)

    def example_object_location_test(cmaps):
        obj = 1-make_polygon_img(points=[43, 50, 50, 50, 38, 38])
        obj = (obj + sum(ndi.rotate(obj, 72*i, order=1, reshape=False)
                         for i in range(5))).clip(0,1)
        oedge = ndi.shift(obj, (-20, 3), cval=0)
        ooutside = ndi.shift(obj, (-30, 30), cval=0)

        # move the dot around inside and outside the circle.
        # move add a polygon noise and see if still works.

        # input shift should be passing for to be testable outside the circle.
        bg1 = np.zeros((100, 100))
        bg2 = make_polygon_img(points=[30,30,30,80,65,80,65,30])/2
            #  y_gradient=False, x_gradient=False, circle_fracs=[.6]) / 2
        bg3 = make_polygon_img(points=[20, 80, 80, 80, 20, 20]) / 2
        assert bg1.shape == bg2.shape

        def overlay(fg, bg):
            z = bg.copy()
            z[(fg > 1e-8)] = 1
            return z.clip(0,1)

        imgs = [
            obj.clip(0,1),
            bg2.clip(0,1),
            overlay(obj, bg2),
            overlay(oedge, bg2),
            overlay(ooutside, bg2),
            overlay(obj, (bg2+bg3)/2)
        ]
        _plot_scattering_for_imgs(imgs, 'Input Object Localization test', cmaps=cmaps)

    def example_local_deformation_test(X, cmaps):
        noise = np.random.randn(*X.shape)
        imgs = [
            X,
            X*2/3+1/3*noise,
            X*1/3+2/3*noise,
            noise,
            np.random.randn(*X.shape)
        ]
        _plot_scattering_for_imgs(imgs, 'Input Gaussian Noise Invariance test', cmaps=cmaps)

    def run_invariance_tests():
        X = make_polygon_img(points=[20, 80, 80, 80, 20, 20])
        assert X.max() == 1
        tests = [
            #  (example_translation_test, (X, ), ),
            #  (example_rotation_test, (X, [plt.cm.gist_ncar]), ),
            #  (example_inversion_test, (X, ), ),
            #  (example_scale_test, (X, [plt.cm.winter]), ),
            #  (example_local_deformation_test, (X, [plt.cm.summer])),
            (example_input_shift_test, ([plt.cm.bone], )),
            #  (example_object_location_test, ([plt.cm.spring], ))
            # TODO: example_adversarial_attack_test, (X, mdl)  # when have scattering layers in middle of network
        ]
        joblib.Parallel(1)(joblib.delayed(f)(*args) for f, args in tests)


    run_invariance_tests()
    #  example_plot_dwt2()
    #  example_plot_scatter_coeffs_polar(
        #  X=make_circle_img((100, 100)),
        #  X=make_polygon_img(points=[20, 80, 80, 80, 20, 20]),
        #  start_fignum=2)
    #  example_plot_scatter_coeffs_polar(
        #  X=make_circle_img(circle_fracs=[1, 0, .8, .6,.0,.4,0,.2,0]),
        #  start_fignum=4)
    #  plt.show()
    plt.show()

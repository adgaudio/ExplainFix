import numpy as np
from dw2.kernel import dct_basis_nd
from dw2.kernel.dct import dct_basis_params_1d
from dw.plotting import plot_img_grid
from matplotlib import pyplot as plt
from matplotlib.colors import CenteredNorm
plt.ion()


def get_1d_basis_as_2d_imgs(K, N):
    # reconstruct 1d basis vectors, but with N points not just K.
    f, t, s = dct_basis_params_1d(K, N).T  # f and t and s are vectors of length K
    col = lambda v: v.reshape(-1,1)
    row = lambda v: v.reshape(1,-1)
    n = np.arange(0, N)  # index position into the signal that DCT would transform
    vecs = col(s) * np.cos(row(n)*col(f) + col(t)) 
    assert vecs.shape == (K, N)
    # project the vector into an image
    out = np.zeros((K, N, N))
    out[np.arange(K).repeat(N), (vecs.reshape(-1)*N+N/2).round().astype('int'), np.arange(N*K)%N ] = 1
    return out


def format_for_plot_fn(K):
    B = dct_basis_nd((K,K))
    B_1d = get_1d_basis_as_2d_imgs(K, 100)
    # make list of plots in order so 1d imgs are along top row and left-most column
    tmp = list(B_1d) + list(B)
    tmp[0:0] = [np.zeros((1,1,4))]
    for k,im in enumerate(B_1d):
        idx = (k+1)*(K+1)
        tmp[idx:idx] = [im.T]
    return tmp, B


def dravel(M, edges_first=True):
    """
    Diagonal Traversal of a matrix, in lines parallel to top right to bottom
    left.  Optionally, sample the edges of the diagonal lines before the
    middles in stable order.  In this optional case, the matrix must be square.

    Outputs a vector of values of M in this diagonal scan order.

    :M: matrix
    :edges_first: if True, sample the edges of the traversed diagonal lines
        before middles
    :returns: a flattened vector of M, in diagonal traversal ordering
    """
    if M.shape[0] != M.shape[1]:
        raise Exception("non-square matrix")
    z = np.zeros(M.shape)

    # invent a matrix with highest values (least important) in bottom right and
    # lowest (most important) in top right.  This naturally orders the matrix
    # in a diagonal line scan ordering, with lines traveling in the direction top
    # right to bottom left.
    max_ = 2*sum(x-1 for x in M.shape)
    z = 2*np.mgrid[:M.shape[0], :M.shape[1]].sum(0)
    assert z.max() == max_
    # augment the matrix to make values closer to the diagonal less important
    # but guarantee the difference is small enough that for each diagonal scan line:
    # the max value is less than the min of the line to the right;
    # the min value is greater than the max of the line to the left.
    # --> THIS part requires a square matrix
    z = z * max_/2-np.fliplr(np.abs(z-max_/2))

    # do the diagonal traversal
    return M.ravel()[z.ravel().argsort(kind='stable')]


def prettify(fig, K):
    fig.subplots_adjust(hspace=5, wspace=5)
    for ax in fig.axes[1:]:
        ax.axis('on')
        ax.set_xticks([])
        ax.set_yticks([])
    # label the basis filters
    kws = dict(
        horizontalalignment='center',
        verticalalignment='center',
        fontsize='xx-large')
    for n, ax in enumerate(dravel(np.array(fig.axes).reshape(K+1, K+1)[1:,1:])):
        ax.text(K//2,K//2, f"{n}", **kws)

for K in [3,5]:
    imgs, B = format_for_plot_fn(K)
    fig = plot_img_grid(
        imgs, rows_cols=(K+1,K+1),
        vmin=B.min(), vmax=B.max(), cmap='PRGn')#, norm=CenteredNorm())
    prettify(fig, K)
    fig.savefig(f'basis_dctII_{K}x{K}.png', bbox_inches='tight')
#  fig = plot_img_grid(format_for_plot_fn(10), rows_cols=(11,11), vmin=-.4, vmax=.4, cmap='PRGn', norm=CenteredNorm())
#  prettify(fig)

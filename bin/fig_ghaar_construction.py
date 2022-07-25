import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import dw2.kernel as K
import dw.plotting as P
from matplotlib.colors import CenteredNorm

rcParams.update({
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'lines.linewidth': 3,
    'lines.markersize': 10,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})


def norm01(x):
    return (x-x.min()) / (x.max()-x.min())


# haar basis
fig, axs = plt.subplots(2,3, figsize=(10,4))
axs = axs.ravel()
[ax.axis('off') for ax in axs]
#  a = np.array([1/np.sqrt(2), 1/np.sqrt(2)]).reshape(2,1)
#  b = np.array([1/np.sqrt(2), -1/np.sqrt(2)]).reshape(2,1)

a = np.array([1,1]).reshape(2,1)
b = np.array([1,-1]).reshape(2,1)
_x = np.linspace(0, np.pi, 2)
assert np.allclose(a, np.sin(0*_x+np.pi/2).reshape(2,1))
assert np.allclose(b, np.sin(_x+np.pi/2).reshape(2,1))
vertical = a@b.T
horizontal = b@a.T
diagonal = b@b.T
axs[0].imshow(vertical, norm=plt.cm.colors.CenteredNorm(0))
axs[0].set_title(r'Vertical, $\mathbf{a} \mathbf{b}^T$')
axs[1].set_title(r'Horizontal, $\mathbf{b} \mathbf{a}^T$')
axs[1].imshow(horizontal, norm=plt.cm.colors.CenteredNorm(0))
axs[2].set_title(r'Diagonal, $\mathbf{b} \mathbf{b}^T$')
axs[2].imshow(diagonal, norm=plt.cm.colors.CenteredNorm(0))
_x = np.linspace(0, np.pi, 3)
a = np.array(_x*0+np.pi/2).reshape(3,1)
b = np.sin(_x+np.pi/2).reshape(3,1)
vertical = a@b.T
horizontal = b@a.T
diagonal = b@b.T
axs[3].imshow(vertical, norm=plt.cm.colors.CenteredNorm(0))
axs[4].imshow(horizontal, norm=plt.cm.colors.CenteredNorm(0))
axs[5].imshow(diagonal, norm=plt.cm.colors.CenteredNorm(0))
#  fig.tight_layout()
fig.subplots_adjust(wspace=.05, hspace=.05)
fig.savefig('./haar_basis.png', bbox_inches='tight')


# pres: creating a haar filter
N=50; f=1; t=np.pi/2
x = np.linspace(0, np.pi, N)

        #  x_h = sin(xf+t) ;  y_h = sin(0+t)
        #  x_v = sin(0+t) ;  y_v = sin(xf+t)
        #  x_d = sin(xf+t) ;  y_d = sin(xf+t)

h_x = np.sin(x*f + t)
h_y = np.sin(x*0 + t)
h_outer = np.outer(h_x, h_y)

fig, ax = plt.subplots(1,1, figsize=(4,4))
ax.plot(x, np.arange(len(x)), 'o')
ax.set_title(r"$x = $linspace$(0, \pi, N)$")
ax.set_xlabel('x')
ax.set_ylabel('N')
fig.savefig('./ghaar_construction1.png', bbox_inches='tight')

fig2, (ax1, ax2) = plt.subplots(1,2, figsize=(8, 4))
ax1.plot(x, h_x, 'o')
ax1.set_title('h_x = $\sin(x*f+t)$')
ax2.plot(x, h_y, 'o')
ax2.set_title('h_y = $\sin(x*0+t)$')
ax1.set_xlabel('x')
ax2.set_xlabel('x')
ax1.set_ylabel('$h_x$')
ax2.set_ylabel('$h_y$')
fig2.tight_layout()
fig2.savefig('./ghaar_construction2.png', bbox_inches='tight')

fig3, axs = plt.subplots(1,1, figsize=(4,4))
axs.imshow(h_outer)
axs.set_title('$h_x h_y^T$')
fig3.savefig('./ghaar_construction3.png', bbox_inches='tight')


# paper: polynomial sine filters
import torch as T
M=100
Fs = []
def make_filter(maxp, L, L_smaller):
    f = np.random.uniform(-4, 4, (L, 2))
    t = np.random.uniform(0, np.pi, (L,2))
    #  p = np.random.uniform(0, (L-1)/2, (L, ))
    p = np.random.uniform(0, maxp, (L, ))
    p[0] = maxp
    #  p = np.abs(np.arange(maxp, maxp-L, -1)*1.)
    assert p.max() <= maxp
    #  p[0] = maxp  # just to ensure maxp is actually correct for the visual
    w = np.random.uniform(0, 1, (L, )) ; w = w/w.sum() * 2 - 1

    return (
        -1+2*norm01(K.polynomial_sin_ND(M=T.tensor(M), f=T.tensor(f), t=T.tensor(t), p=T.tensor(p), w=T.tensor(w))),
        -1+2*norm01(K.polynomial_sin_ND(M=T.tensor(M), f=T.tensor(f[:L_smaller]), t=T.tensor(t[:L_smaller]), p=T.tensor(p[:L_smaller]), w=T.tensor(w[:L_smaller]))),
    )
fig4, axs = plt.subplots(2, 5, figsize=(5*3, 2*3))
for nthcol, maxp in enumerate(np.linspace(3, 40, 5)):
    maxp = int(maxp)
    L = 2*maxp+1
    L_sparse = L//6  # int(L/10)  # max(1, int(L/10))
    Fs.extend(make_filter(maxp, L, L_sparse))
    axs[0, nthcol].set_title('$max(\mathbf{p})=%s$\n $\ell=%s$' % (maxp, L))
    axs[0, nthcol].imshow(Fs[-2], norm=plt.cm.colors.CenteredNorm(0), cmap='PuOr')
    axs[1, nthcol].set_title(r'$\ell=%s$' % L_sparse)
    axs[1, nthcol].imshow(Fs[-1], norm=plt.cm.colors.CenteredNorm(0), cmap='PuOr')
[ax.axis('off') for ax in axs.ravel()]
fig4.savefig('./polynomial_sin_L_vs_p.png', bbox_inches='tight')
    #  print(
        #  (Fs[-2].mean(), Fs[-2].min(), Fs[-2].max()),
        #  (Fs[-1].mean(), Fs[-1].min(), Fs[-1].max()),)
fig4a = P.plot_img_grid(Fs[0::2], rows_cols=(1,5), norm=plt.cm.colors.CenteredNorm(0), cmap='PuOr')
fig4b = P.plot_img_grid(Fs[1::2], rows_cols=(1,5), norm=plt.cm.colors.CenteredNorm(0), cmap='PuOr')
                      #  ) + Fs[1::2], rows_cols=(2,5))
fig4a.savefig('./polynomial_sin_correct_ell.png', bbox_inches='tight')
fig4b.savefig('./polynomial_sin_wrong_ell.png', bbox_inches='tight')
print([x.mean() for x in Fs[1::2]])
plt.close('all')

plt.show()

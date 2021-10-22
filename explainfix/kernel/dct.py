from typing import Union, Tuple
import numpy as np
import math
import torch as T
from itertools import product
import warnings


def dct_basis_params_1d(K: int, N=None, basis:Union[str,Tuple[float,float]]='DCT-II'):
    """Return the frequency, phase and amplitude parameters for the first
    K basis vectors of the Discrete Cosine Transform.
    This function implements only DCT Type 2 (DCT-II) or Type 4 (DCT-IV),
    but you can pass in custom `basis` to specify other DCT types.
    Considering the basis vectors as a matrix, then DCT-III and DCT-II are
    inverses of each other.

    The DCT basis vectors have the general form:
        s*cos(2*pi/N*(k+b)*(n+a))
          for any real numbers a and b, N samples, nth sample, kth basis vector
          and scale s.

    Different DCT basis vary in choices (a,b).

        DCT Type-I:     a=0,      b=0
        DCT Type-II:    a=1/2,    b=0
        DCT Type-III:   a=0,      b=1/2
        DCT Type-IV:    a=1/2,    b=1/2

    The basis vectors are already orthogonal.  Choosing `s` correctly can make
    the basis vectors orthonormal.
        DCT-II: orthonormal, and each vector (except first one) has sum=0.
        DCT-IV: orthonormal, but vectors do not all sum to 1.

    For further reference, consult:
        - wikipedia for "Discrete Cosine Transform"
        - Martucci, Stephen A. "Symmetric convolution and the discrete sine and cosine transforms." IEEE Transactions on Signal Processing 42.5 (1994): 1038-1051.

    Pseudo-code to construct the basis vectors is:
        >>> f, t, s = get_bases_1d().T  # f and t and s are vectors of length K
        >>> col = lambda v: v.reshape(-1,1)
        >>> row = lambda v: v.reshape(1,-1)
        >>> n = range(0, N)  # index position into the signal that DCT would transform
        >>> col(s) * cos(row(n)*col(f) + col(t))  #  (K, N) matrix, each row is a basis vector

    :K: the number of basis vectors.
    :N: the number of elements in any given basis vector.
      If N=K, the basis is complete.  If N>K, the basis is undercomplete.
      By default, N=K.
    :basis:  The type of basis.  Currently supports "DCT-II" or "DCT-IV" or pass
        a tuple of floats (a,b).  If passed a string argument, the "a" will
        be a value other than 1 that makes the vectors orthonormal.
    :returns: a array of shape (K, 3) containing the parameters (f, t, a) for
        each of the K basis vectors
    """
    if N is None:
        N = K
    if basis == 'DCT-II':
        a, b = 1/2, 0
    elif basis == 'DCT-IV':
        a, b = 1/2, 1/2
    #  elif basis == 'DCT-III':
    #      a, b = 0, 1/2
    #  elif basis == 'DCT-I':
    #      a, b = 0, 0
    elif isinstance(basis, str):
        raise NotImplementedError(basis)
    else:
        a, b = basis
    #  P = np.array([[ np.pi*(k+1/2)/N, np.pi/N*(k/2+1/4), np.sqrt(2/N)] for k in range(K)])  # type IV
    P = np.array([[ np.pi/N*(k+b), np.pi/N*(k+b)*a, 1] for k in range(K)])  # type IV
    if basis == 'DCT-IV':
        P[:,2] = np.sqrt(2/N)
    elif basis == 'DCT-II':
        P[:,2] = np.sqrt(2/N)
        P[0,2] = 1/np.sqrt(N)
    #  elif basis == 'DCT-III':
    #      pass
    #  if basis == 'DCT-I':
    #      pass
    return P


def dct_basis_1d(K, basis:str="DCT-II"):
    """
    Construct the complete, orthonormal 1D basis vectors spanning vectors of
    length K.  Currently, `basis` determines the type of DCT basis,
    and only DCT-II or DCT-IV are orthonormal in this implementation.

    :returns: KxK matrix, each row is a basis vector.
    """
    warnings.warn("DEPRECATED.  Use dct_basis_nd((K, ), ...) ")
    N = K  # num samples == num basis vectors
    f, t, s = dct_basis_params_1d(K, basis=basis).T
    n = np.arange(N).reshape(1,-1)
    return s.reshape(-1,1)*np.cos(f.reshape(-1,1)*n+t.reshape(-1,1))


def dct_basis_2d(N, M, basis:str="DCT-II"):
    """
    Construct the complete, orthonormal 2D basis (as a set of (N,M) matrices)
    for the Discrete Cosine Transform (Type 2).  Returned basis spans space of
    real numbers in (N,M).

    `N` the num rows of the matrices you wish to represent
    `M` the num cols of the matrices you wish to represent
    `basis` the type of Discrete Cosine Transform basis.  Currently, DCT-II and
        DCT-IV are supported.

    :returns: numpy array containing basis vectors along rows, in form: (NM, N,
    M).  This set of (N,M) sub-matrices forms a complete, orthonormal basis.
    """
    warnings.warn("DEPRECATED.  Use dct_basis_nd((N, M), ...).  ")
    return np.stack([
        np.outer(vy, vx)
        for vy in dct_basis_1d(N, basis=basis)
        for vx in dct_basis_1d(M, basis=basis)])
#  shape = (h,w,...)
#  def dct_basis_nd(kernel_shape, basis='DCT-II')
#  np.eye(shape)*-2+1  # output like this [ [-1,1,1], [1,-1,1], [1,1,-1] ]

def dct_basis_nd(kernel_shape:tuple[int], basis:str='DCT-II'):
    """
    Construct the complete orthonormal N-D basis for the Discrete Cosine Transform (Type 2).

    `kernel_shape` the space of tensors that the basis will represent.
    `basis` the type of Discrete Cosine Transform basis.  Currently, DCT-II and
        DCT-IV are supported.

    :returns: numpy array containing basis vectors along rows, with shape: (M, *kernel_shape) where M = prod(kernel_shape)
    """
    # There are d spatial dimensions.  do a neat trick with itertools.product to
    # have d nested for loops for the d spatial dimensions.
    bases = [
        dct_basis_1d(dim_size, basis=basis).reshape(dim_size, *basis_vector_shape)
        for dim_size, basis_vector_shape in zip(
            kernel_shape, np.eye(len(kernel_shape), dtype='int')*-2+1)]
    basis_nd = [math.prod(vecs) for vecs in product(*bases)]
    return np.array(basis_nd)


def ghaar4_2d(N:int,  M:int,  a:float,  h:float,  v:float,  d:float) -> np.ndarray:
    """Generalize 2D Haar wavelet filters by 
    a) using, for rows and for columns, the first two DCT-2 basis vectors instead of step functions
    b) steering the approximation, horizontal, vertical and diagonal basis filters.
    #
    Note that the basis filters, when flattened into vectors, are orthonormal
    but a set of steered filters from this basis likely aren't orthogonal
    #
    # Note: not the version used in ExplainFix paper, but it's basically the same
    #
    :N:, :M: - the desired output shape (N, M)
    :a,h,v,d: - weights for approximation, horizontal, vertical, diagonal basis filters
    :return: (N,M) matrix.
    """
    # orthonormal 1d basis vectors for rows
    fy, ty, sy = dct_basis_params_1d(2, N).T
    y = np.arange(N).reshape(1,-1)
    y1, y2 = sy.reshape(-1,1)*np.cos(fy.reshape(-1,1)*y+ty.reshape(-1,1))
    # orthonormal 1d basis vectors for columns
    fx, tx, sx = dct_basis_params_1d(2, M).T
    x = np.arange(M).reshape(1,-1)
    x1, x2 = sx.reshape(-1,1)*np.cos(fx.reshape(-1,1)*x+tx.reshape(-1,1))
    # construct orthonormal 2d basis filters
    approximation = np.outer(y1,x1)
    horizontal = np.outer(y2, x1)
    vertical = np.outer(y1, x2)
    diagonal = np.outer(y2,x2)
    #  diagonal = np.sin(fdy*y+t)*np.sin(fdx*x+t)
    A = approximation.reshape(-1)
    H = horizontal.reshape(-1)
    V = vertical.reshape(-1)
    D = diagonal.reshape(-1)
    filter = a*approximation + h*horizontal + v*vertical + d*diagonal
    return filter


def dct_steered_2d(N:int, M:int, w:np.ndarray, basis_idxs: np.ndarray, p:np.ndarray=None) -> np.ndarray:
    """Steer the DCT basis for 2d matrices.

    :N:, :M: size of output matrix
    :basis_idxs: array of L<NM indexes of the relevant basis filters to be steered.
    :w: array of L weights used to steer the filters
    :p: optional.  array of L values that can perform polynomial steering.
    :returns: a (N,M) steered matrix that is the linear combination or polynomial combination
      of the chosen basis.
    """
    B = dct_basis_2d(N, M)[basis_idxs]
    L = len(basis_idxs)
    if p is None:
        p = np.ones((L,1,1))
    else:
        p = p.reshape(L,1,1)
    assert B.shape == (L, N, M)
    return (B*w.reshape(L,1,1)**p).sum(0)


def _ensure_minlength(val, n=2):
    try:
        val[n-1]
    except TypeError:
        val = tuple(val for _ in range(n))
    return val


if __name__ == "__main__":
    z = dct_basis_2d(3, 4, 'DCT-II')
    print(z.shape)
    B = z.reshape(12, 12)  # vecs on rows
    # DCT-II and DCT-IV pass orthonormal tests:
    assert np.allclose(B@B.T, np.eye(12))
    B = dct_basis_2d(3,4, 'DCT-IV').reshape(12, 12)
    assert np.allclose(B@B.T, np.eye(12))
    #  print((B.T@B).round(10))

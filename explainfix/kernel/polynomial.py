from typing import Dict, List, Union, Iterable, Tuple, Optional
import torch as T
import math


def polynomial_sin_ND(M, f, t, p, w) -> T.Tensor:
    """
    Psine initialization.
    Generate and return a large kernel with very few parameters, based on sine waves.

    Return a N-dimensional tensor containing M elements in each dimension.   The
    tensor is a polynomial combination of L separable tensors.  The L separable
    sub-kernels are each computed as outer tensor product of a parameterized sine
    wave function evaluated over domain [-pi, +pi], and they all have the
    same shape as the returned kernel.

    The method is:

    For every separable kernel l, and dimension n, we compute:
    $$ dim_ln = \sin(x * f_{l,n} + t_{l,n})  $$

    Then, to create a kernel l, we combine the dimensions
    $$ kernel_l = outer( outer(dim_l1, dim_l2), ... dim_lN)  $$

    Finally, we combine all kernels with a polynomial function
    $$ kernel = \sum_l w_l * kernel_l**{p_l}  $$


    :M: scalar.  number of values per dimension
    :f: (L, N) tensor.  frequency parameters
    :t: (L, N) tensor.  translation parameters
    :p: (L, ) tensor.  exponent parameters.  Should be positive integers, but
    if float, will round to nearest integer
    :w: (L, ) tensor.  coefficient weight for each term in the polynomial eqtn.

    :return: (M, M, ...) tensor with N dimensions.
    """
    L, N = f.shape
    assert (L, N) == f.shape == t.shape
    assert (L, ) == p.shape == w.shape
    f = f.reshape(L, N, 1)
    t = t.reshape(L, N, 1)

    x_LNM = T.linspace(0, math.pi, M, device=f.device).reshape(1, 1, M).repeat((L, N,1))
    sin_x_LNM = (T.sin(x_LNM * f + t))
    # TODO: add a learned "mother wavelet" style function here.
    # should it be sin_x_LNM * mother or shoud it just be x_LNM without the params?
    return polynomial_ND(sin_x_LNM, p, w)


def polynomial_ND(arr_LNM: T.Tensor, p: T.Tensor, w: T.Tensor) -> T.Tensor:
    """ Generate a large kernel by creating a polynomial combination of L
    separable sub-kernels, each with side length M and dimension N.  `p` is the
    element-wise power and `w` is the weight under the polynomial combination.

    Return a N-dimensional tensor of shape (M,M,M,...).  The tensor is a
polynomial combination of L separable tensors, or sub-kernels.  The L separable
sub-kernels are each computed as outer tensor product of a (M, ) length vector
from the given :arr_LNM:.

    The method is:

    For every separable kernel l, and dimension n, we compute:
    $$ dim_{l,n} = arr_LNM[l,n]

    Then, to create a kernel l, we combine the dimensions
    $$ kernel_l = outer( outer(dim_{l,1}, dim_{l,2}), ..., dim_{l,N})  $$

    Finally, we combine all kernels with a polynomial function, where the
    exponent is element-wise over all values of the kernel.
    $$ kernel = \sum_l w_l * kernel_l**{p_l}  $$

    :arr_LNM: a float tensor of shape (L,N,M) with optionally learnable values.
        L - Number of sub-kernels to combine under a polynomial combination
        N - scalar. kernel num dimensions.  2 is a square, 3 is a cube, ...
        M - scalar.  number of values per dimension
        You can, for instance, design this array so each element is learnable,
        or it can be produced by a generating function of fewer learned parameters.
    :p: - a float tensor of length (L, ) containing the exponent to raise each
        element of the corresponding sub-kernel.
        NOTE: Decimal values of p could create complex numbers.  For example,
        if any value, x_j, in an outer product `kernel_l` is negative, and if
        the power, p_i, is a decimal number, then the resulting number x_j**p_i
        could have multiple possible answers (google search for "what is
        (-1)**(1/3)" - this particular example happens to have three possible
        values).  As a hack and constraint, we can force a real number by first
        rounding p_i to an integer.  This limits the capacity of this method,
        but serves as a workaround to the fact that exponentiation of negative
        numbers to decimal powers is ambiguous.
    :w: - a float tensor of length (L, ) the coefficient weight of each
        sub-kernel matrix

    :returns:  N-dimensional tensor of shape (M,M,M,...).
    """
    L, N, M = arr_LNM.shape
    kern = T.zeros((M, ) * N, dtype=T.float, device=arr_LNM.device)
    p = p.round()
    for i in range(L):
        x_NM = arr_LNM[i]
        # tensor outer product for each L-th kernel
        assert x_NM.shape == (N, M)
        einsum_letters = [chr(97+i) for i in range(N)]
        einsum_notation = '%s->%s' % (','.join(einsum_letters), ''.join(einsum_letters))  # 'a,b->ab'
        kernel_i = T.einsum(einsum_notation, *[x_NM[n] for n in range(N)])
        kern += (kernel_i**p[i] ) * w[i]
    return kern


def _rand_ones_init(shape, device, noise_radius=.5):
    return T.ones(shape, dtype=T.float, device=device) + (T.rand((shape), device=device)-.5)*noise_radius*2


def _init_params_polynomial_sin_ND(L=6, N=2, M=15, device='cpu', **kwargs):
    """
    Generate learnable pytorch parameters for the polynomial_sin_ND with
    reasonable defaults near frequencies of 1 and where the powers are floating
    numbers near 1.

    N - kernel num dimensions.  2 is a square, 3 is a cube, ...
    L - Number of sub-kernels to combine under a polynomial combination
    **kwargs  parameters to the polynomial_sin_ND function
      - M=FloatTensor, f
      - f=FloatTensor of shape (L, N)
      - t=FloatTensor of shape (L, N)
      - p=FloatTensor of shape (L, )
      - w=FloatTensor of shape (L, )
    """
    #  rv = dict(
        #  f=_rand_ones_init(shape=(L, N), device=device),  # frequencies close to 1
        #  t=T.rand((L, N), device=device) * math.pi*2,  # in Uniform(0, 2\pi)
        #  p=T.randn((L, ), device=device) +1,
        #  w=_rand_ones_init(shape=(L, ), device=device),
    #  )
    rv = dict(
        f=_rand_ones_init((L, N), device=device),
        t=_rand_ones_init((L, N), device=device) * math.pi/2,
        #  L = 2p + 1 --> p = (L-1)/2
        p=_rand_ones_init((L, ), device=device) * (L-1)/2,  # TODO: if p is entirely even or entirely odd, then can do *(L-1)
        #  p=T.ones((L,  ), dtype=T.float, device=device),
        w=_rand_ones_init((L, ), device=device),
    )
    for k in ['f', 't', 'p', 'w']:
        rv[k] = T.nn.Parameter(rv[k], requires_grad=True)
    #  rv['p'].requires_grad = False
    #  rv['f'].requires_grad = False
    rv['M'] = T.nn.Parameter(T.tensor(M), requires_grad=False)
    return T.nn.ParameterDict(rv)


polynomial_sin_ND.init = _init_params_polynomial_sin_ND


def _init_polynomial_ND(L, N, M, device='cpu', **kwargs):
    # arbitrary sine wave initialization
    arr_LNM = T.linspace(0, 2*math.pi, M, dtype=T.float, device=device).reshape(1,1,M)
    f = _rand_ones_init((L, N, 1), device=device)
    t = _rand_ones_init((L, N, 1), device=device)
    arr_LNM = T.sin(arr_LNM * f + t)

    assert arr_LNM.shape == (L, N, M)
    p=_rand_ones_init((L, ), device=device) * (L-1)/2
    w=_rand_ones_init((L, ), device=device)
    return T.nn.ParameterDict(dict(
        arr_LNM=T.nn.Parameter(arr_LNM, requires_grad=True),
        p=T.nn.Parameter(p, requires_grad=True),
        w=T.nn.Parameter(w, requires_grad=True),
    ))


polynomial_ND.init = _init_polynomial_ND


class PolyConv2d(T.nn.Module):
    """DEPRECATED"""
    def __init__(self,
                 in_channels: int, out_channels: int,
                 params: List[T.nn.ParameterDict] = None,
                 L: Union[int, Iterable[int]] = None,
                 M: int = None,
                 conv_params: Dict[str,any] = None,
                 kernel_method=polynomial_ND,
                 ):
        """
        :params:
            parameters used to generate the filter kernel.
            These are keyword arguments to polynomial_sin_ND.
            The length of the list should be (out_channels * in_channels)
            and indexed according to [(out_1, in_1), (out_1, in_2), ... ].
        :L: required only if params is None.
            There is one value of L for every (out_channel, in_channel).
            L specifies the number of sub-kernels to combine under a polynomial
            combination.  If an int, use same L for all in and out channels.
            If a Tensor, should be of shape (out_channels, in_channels).
        :M: required only if params is None.  Size of the 2d filter kernel.
        :conv_params: keywords passed to pytorch.nn.Conv2d
        :kernel_method:  a function that generates the kernel.  Currently 2 options:
            kernel_method=polynomial_ND - generates kernel where num parameters linearly proportional to L or M.
            kernel_method=polynomial_sin_ND - num parameters constant w.r.t. kernel size M, linear with L
        """
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        if params is None:
            assert (M is not None) and isinstance(M, int)
            assert L is not None
            if isinstance(L, int):
                L = [L]*(out_channels*in_channels)
            params = [kernel_method.init(N=2, L=L_i, M=M) for L_i in L]
            # register parameters
            for i, param_dct in enumerate(params):
                for k,v in param_dct.items():
                    self.register_parameter(f'{k}-{i}', v)
        self.params = params
        self.conv_params = conv_params or {}
        self.kernel_method = kernel_method

    def forward(self, x):
        filters = T.stack([self.kernel_method(**dct) for dct in self.params])
        M = filters.shape[-1]
        filters = filters.reshape(self.out_channels, self.in_channels, M, M)
        x = T.conv2d(x, filters, **self.conv_params)
        return x


def ghaar2d(M: int, h: float, v: float, d: float, a:float=0, f:float=1., norm=True) -> T.Tensor:
    """Generalize 2D ghaar wavelet filters by steering horizon, vertical, 
    diagonal and optionally approximation components.

    Actually, the computation here looks more complicated than it is!
    It's just a weighted sum of three outer products.

        x = linspace(0, pi, M)
        t = pi/2
        f = 1

        x_h = sin(xf+t) ;  y_h = sin(0+t)
        x_v = sin(0+t) ;  y_v = sin(xf+t)
        x_d = sin(xf+t) ;  y_d = sin(xf+t)

        filter = h*outer(x_h,y_h) + v*outer(x_v, y_v) + d*outer(x_d,y_d)
        return filter / l2_norm(filter.flatten())

    Returned filter sums to 0 and is normalized to have l2 norm of 1.

    :M: kernel size (num elements in any dimension)
    :h,v,d: scalar weights to combine the horizontal, vertical and diagonal filters.  The combination performs filter steering.
      Proper haar wavelets satisfy (h,v,d) in { (1,0,0), (0,1,0), (0,0,1) }.
      For example, you can rotate the horizontal and vertical filters to make
      arbitrary orientations using (a,b,0) for any scalar a and b.

      I think it is possible to define h,v,d as scalar tensors with
      requires_grad, but all three values must have requires_grad.  This option
      is not tested.
    :f: the frequency of the sine waves.  If f is an odd integer, it naturally gives a wavelet filter that sums to 0.  Typically between [0,M+1), since it's cyclical.

      Horizontal:  ghaar2d(M, 1,0,0,0)
      Vertical:    ghaar2d(M, 0,1,0,0)
      Diagonal:    ghaar2d(M, 0,0,1,0)
      Approx:      ghaar2d(M, 0,0,0,1)

    :returns: Tensor of shape (M,M), where
      - the matrix elements sum to 0
      - the sum of squares of matrix elements equals 1
    """
    w = T.tensor([h,v,d,a])
    f = float(f)
    if isinstance(h, T.Tensor) and h.requires_grad:
        assert v.requires_grad and d.requires_grad, "user error"
        w.requires_grad=True
    ret = polynomial_sin_ND(
        M, T.tensor([[f,0.], [0.,f], [f,f], [0.,0.]]), T.tensor([[math.pi/2, math.pi/2]]*4), T.tensor([1.,1.,1.,1.]), w)
    # normalize so it's a proper wavelet (sum == 0) and sum(i**2) = 1
    if norm:
        ret = ret / (ret**2).sum()**.5
    return ret

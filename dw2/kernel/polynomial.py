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
    """Generalize 2D haar wavelet filters by steering horizon, vertical, 
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


class GHaarConv2d(T.nn.Module):
    """
    Learnable Generalized Haar Wavelets Filters.

    Drop-in replacement for Conv2d (supporting the most common keyword
    arguments) that uses the generalized Haar filters.  The filter size can
    have any shape 2x2 or larger, and regardless of size, a spatial filter is
    always created from 4 parameters: h, v, d, f.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = False,
        #  padding_mode: str = 'zeros',  # not implemented
    ):
        super().__init__()
        assert isinstance(kernel_size, (int, tuple))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.hvdf = T.nn.Parameter(T.Tensor(out_channels, in_channels // groups, 4))
        if bias is not None:
            self.bias = T.nn.Parameter(T.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.conv2d_hyperparams = dict(
            stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=self.bias)
        self.reset_parameters()
        self.filters = None  # filters are cached in eval mode.

    def reset_parameters(self, fmin=0, fmax=5) -> None:
        # init h, v, d weights
        T.nn.init.uniform_(self.hvdf[..., :3], -1, 1)
        # init frequency
        T.nn.init.uniform_(self.hvdf[..., 3], fmin, fmax)
        # bias, if enabled
        if self.bias is not None:  # kaiming uniform initialization
            in_ch = self.hvdf.shape[1]
            #  out_ch = self.hvdf.shape[0]
            receptive_field_size = math.prod(self.kernel_size)
            fan_in = in_ch * receptive_field_size
            #  fan_out = out_ch * receptive_field_size
            bound = 1 / math.sqrt(fan_in)
            T.nn.init.uniform_(self.bias, -bound, bound)

    @staticmethod
    def ghaar(kernel_size, hvdf, bigger=0):
        assert hvdf.shape[-1] == 4, hvdf.shape
        y_bigger = math.pi/(kernel_size[0]-1)*(bigger//2)
        x_bigger = math.pi/(kernel_size[1]-1)*(bigger//2)
        y = T.linspace(0-y_bigger, math.pi+y_bigger, kernel_size[0]+bigger,
                       device=hvdf.device).reshape(1,1,kernel_size[0]+bigger,1)
        x = T.linspace(0-x_bigger, math.pi+x_bigger, kernel_size[1]+bigger,
                       device=hvdf.device).reshape(1,1,1,kernel_size[1]+bigger)
        h = hvdf[..., 0].unsqueeze(-1).unsqueeze(-1)
        v = hvdf[..., 1].unsqueeze(-1).unsqueeze(-1)
        d = hvdf[..., 2].unsqueeze(-1).unsqueeze(-1)
        f = hvdf[..., 3].unsqueeze(-1).unsqueeze(-1)
        t = math.pi/2
        x_h = T.sin(x*f+t) ;  y_h = T.sin(y*0+t)
        x_v = T.sin(x*0+t) ;  y_v = T.sin(y*f+t)
        x_d = T.sin(x*f+t) ;  y_d = T.sin(y*f+t)
        filter = h*(x_h*y_h) + v*(x_v*y_v) + d*(x_d*y_d)
        return filter

    def forward(self, x):
        # cache the filters during model.eval(). NO cache during model.train()
        if not self.training:
            if self.filters is None:
                filters = self.filters = self.ghaar(self.kernel_size, self.hvdf)
            else:
                filters = self.filters
        else:
            self.filters = None
            filters = self.ghaar(self.kernel_size, self.hvdf)
        return T.conv2d(x, filters, **self.conv2d_hyperparams)


class FixedHaarConv2d(T.nn.Module):
    """
    DEPRECATED IDEA
    Perform grouped convolutions on image with a set of redundant and square
    Haar filters.  Groups are one per input channel.

    :filter_size: The size, M, of the square Haar filters in either dimension.
        Increase M to gain stability to noise and also ignore high frequency
        information (Note: increasing dilation might be a more efficient way to
        ignore high frequency information).
    :hvd_weights: A tuple of weightings that generates a set of Haar filters.
        Each Haar filter is created by a weighted
        sum of horizontal (h), vertical (v) and diagonal (d) basis filters, where
        (h,v,d) are the weights.  The redundant Haar filters generated by
        cartesian product of `hvd_weights` over the 3 basis filters.
        The number of output filters is `len(hvd_weights)**3`.
        Default value is `(1,0)`.

        Smaller differences in values result in coefficients that are closer
        together (contraction), while larger differences results in values
        farther apart (expansion).  Maybe this is useful in combination with a
        non-linear activation function (like clamp) to enhance or decrease contrast.
        For more (exponentially) more redundancy, you can use larger
        tuples (.1,.2,0,-.1,-.2).  Note that (1,0,-1) gives 27 filters, the
        first 12 and last 12 are inversely equal, while (10,1,0) gives the same
        set of filters twice, but the duplicate filters will have intensity
        values in a different range.
    :dilation:
    :stride:
        dilation and stride are convolution parameters passed to pytorch.conv2d.
        Dilation is used determine padding ("same" convolution), while stride
        is ignored when determining padding.
        Increase stride to downsample the image.
        Increase dilation to perform multi-scale search.
    :padding: passed to pytorch.conv2d if it is an int or tuple.  If auto,
        ensure that dilation does not change output size but let stride downsample the image.
        Output size may be off by 1 row/col depending on choice of stride/dilation.
    :bias: An optional Parameter tensor with one value for each output channel.
        You should have already initialized the tensor with proper values.
        This parameter requires you to define a fixed number of input channels.
    :remove_symmetric: If True, check if the hvd_weights are symmetric around zero.
        This means the first half of filters are inversely equal to the second
        half.  This occurs if the hvd_weights=(1,0,-1) or (1,.5,0,-.5,-1).
        Remove one half of the filters so they are not computed.

    :returns: Output a tensor of shape (B,C,N,H,W) corresponding to Batch size, Channels, len(hvd_weights)**3, Height, Width
    # TODO: return value might be a list of varying h and w...
    """
    def __init__(self,
                 filter_size: int = 2,
                 stride: Union[int,Tuple[int]] = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int], str] = 'auto',
                 bias: Optional[T.nn.Parameter] = None,
                 hvd_weights: Tuple[float] = (1,0,-1),  # TODO: just (1,0)?
                 remove_symmetric: bool = True,
                 include_approximation_img: bool = True,
                 ):
        # boilerplate
        super().__init__()
        self.filter_size = filter_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.bias = bias
        # pre-build the haar filters (kernels) for this image
        hvd_vals = [(x,y,z) for x in hvd_weights for y in hvd_weights for z in hvd_weights]
        _middle = len(hvd_vals)//2
        if remove_symmetric:
            is_symmetric = hvd_vals[_middle] == (0,0,0) \
                    and all(all(hvd_vals[i][j] == -1.0*hvd_vals[-i-1][j] for j in range(3))
                            for i in range(_middle))
            if is_symmetric:
                hvd_vals = hvd_vals[:_middle]
        filters = []
        if include_approximation_img:
            # --> approximation filter (just a box filter)
            filters.append(T.ones((self.filter_size, self.filter_size), dtype=T.float))
        # --> haar filters
        filters.extend([ghaar2d(self.filter_size, h,v,d) for h,v,d in hvd_vals])
        self.filters = T.nn.Parameter(T.stack(filters), requires_grad=False)

    def forward(self, x):
        B, C = x.shape[:2]  # batch size, channels, height, width
        M = self.filter_size
        N = len(self.filters)  # num filters
        convolved_outputs = []
        scales = [1]  # TODO: multi-scale search
        for j in scales:
            # TODO: if current img size, x.shape, depends on scale, change the assert
            assert all(M <= dim for dim in x.shape[-2:]), f"Last two dimensions of input with {x.shape} are too small for the filters of size {M}."

            #  if M > x.shape[-1] or M > x.shape[-2]:
                #  break  # skip scales that are too large for this image.
                #  # TODO: test this if loop when the bug arises because img too small for the filter.

            filters = self.filters.reshape(1,N, M,M).repeat(C, 1,1,1).reshape(C*N, 1,M,M)
            #  assert filters.shape == (self.out_channels, self.in_channels, M, M), 'code bug'

            if j == 1:
                H, W = x.shape[-2:]
                conv_params = dict(
                    padding=(self._get_padding1d(H), self._get_padding1d(W)),
                    stride=self.stride, dilation=self.dilation, bias=self.bias,
                    groups=C, )
                print(conv_params)
                del H, W  # to avoid coding bugs
            else:
                # TODO: depends on scale.  choose how to deal with multi-scale search.
                raise NotImplementedError('cannot do other scales yet')

            conv_img = T.conv2d(x, filters, **conv_params)
            convolved_outputs.append(
                conv_img.reshape(B, C, N, *conv_img.shape[-2:]))
            # TODO: may fail due to different size filters making different H and W
        assert all(x.shape == convolved_outputs[0].shape for x in convolved_outputs)
        ret = T.cat(convolved_outputs, dim=2)
        print(ret.shape)
        assert ret.shape[:3] == (B, C, len(self.filters))
        return ret


    def _get_padding1d(self, i):
        """Compute padding so output is as close to image size as possible,
        except let stride downsample the image

        :i: input size, scalar

        Based on the formula:  o = (i + 2*P - M - (M-1)*(D-1))/S + 1
          where P is padding, M filter size, S stride and D dilation.
          o and i correspond to input and output size (i == o in this function).
        """
        S=1#self.stride
        D=self.dilation
        if self.padding == 'auto':
            M = self.filter_size
            return math.ceil((S*(i-1)-i+D*(M-1)+1)/2)
        else:
            return self.padding



#  def conv_filter(M, N):
#      k = T.tensor((M, ) * N, requires_grad=True)
#      T.nn.init.kaiming_uniform_(k, a=math.sqrt(5))
#          if self.bias is not None:
#              fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#              bound = 1 / math.sqrt(fan_in)
#              init.uniform_(self.bias, -bound, bound)

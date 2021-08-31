from . import polynomial

from .polynomial import (polynomial_ND, polynomial_sin_ND, haar2d, GHaarConv2d)
from .dct import (
    dct_basis_nd,
    # these are deprecated:
    dct_basis_2d, dct_basis_1d,
    # these are not really used
    DCTConv2d, ghaar4_2d, dct_steered_2d
)

"""
Objects imported here will live in the `pycollocation.basis_functions`
namespace.

"""
from . basis_functions import BasisFunctionLike
from . basis_splines import BSplineBasis
from . polynomials import PolynomialBasis

__all__ = ["BasisFunctionLike", "BSplineBasis", "PolynomialBasis"]

"""
Objects imported here will live in the `pycollocation.basis_functions`
namespace.

"""
from . basis_functions import BasisFunctionLike
from . polynomials import PolynomialBasis

__all__ = ["BasisFunctionLike", "PolynomialBasis"]

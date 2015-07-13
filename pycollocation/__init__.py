"""
Objects imported here will live in the `pycollocation` namespace

"""
__all__ = ["TwoPointBVP", "TwoPointBVPLike", "SymbolicTwoPointBVPLike",
           "SymbolicTwoPointBVP", "OrthogonalPolynomialSolver", "Visualizer"]

from . import orthogonal_polynomials
from . import visualizers

from . bvp import *
from . orthogonal_polynomials import OrthogonalPolynomialSolver
from . visualizers import Visualizer

# Add Version Attribute
from pkg_resources import get_distribution

__version__ = get_distribution('pyCollocation').version

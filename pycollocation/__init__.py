"""
Objects imported here will live in the `pycollocation` namespace

"""
__all__ = ["BoundaryValueProblem", "OrthogonalPolynomialSolver", "Solution"]


from . import models
from . import orthogonal_polynomials
from . import solutions

from . models import BoundaryValueProblem
from . orthogonal_polynomials import OrthogonalPolynomialSolver
from . solutions import Solution

# Add Version Attribute
from .version import version as __version__

"""
Objects imported here will live in the `pycollocation` namespace

"""
__all__ = ["SymbolicBoundaryValueProblem", "OrthogonalPolynomialSolver", "Visualizer"]


from . import boundary_value_problems
from . import differential_equations
from . import models
from . import orthogonal_polynomials
from . import visualizers

from . boundary_value_problems import SymbolicBoundaryValueProblem
from . orthogonal_polynomials import OrthogonalPolynomialSolver
from . visualizers import Visualizer

# Add Version Attribute
from pkg_resources import get_distribution

__version__ = get_distribution('pyCollocation').version

"""
Objects imported here will live in the `pycollocation.solvers` namespace

"""
from . over_identified import LeastSquaresSolver
from . solvers import Solver, SolverLike
from . solutions import SolutionLike, Solution

__all__ = ["LeastSquaresSolver", "Solver", "Solution", "SolutionLike",
           "SolverLike"]

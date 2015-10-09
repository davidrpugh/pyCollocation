"""
Objects imported here will live in the `pycollocation.solvers` namespace

"""
from . solvers import LeastSquaresSolver, Solver, SolverLike
from . solutions import SolutionLike, Solution

__all__ = ["LeastSquaresSolver", "Solver", "Solution", "SolutionLike",
           "SolverLike"]

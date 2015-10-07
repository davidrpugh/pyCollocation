"""
Objects imported here will live in the `pycollocation.solvers` namespace

"""
from . solvers import Solver, SolverLike
from . solutions import SolutionLike, Solution

__all__ = ["Solver", "Solution", "SolutionLike", "SolverLike"]

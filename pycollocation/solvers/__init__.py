"""
Objects imported here will live in the `pycollocation.solvers` namespace

"""
from . bases import SolverLike
from . polynomials import PolynomialSolver
from . solutions import SolutionLike, Solution

__all__ = ["PolynomialSolver", "Solution", "SolutionLike", "SolverLike"]

"""
Objects imported here will live in the `pycollocation.solvers` namespace

"""
from . bases import SolutionLike, SolverLike
from . polynomials import PolynomialSolver
from . solutions import Solution

__all__ = ["PolynomialSolver", "Solution", "SolutionLike", "SolverLike"]

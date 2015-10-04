"""
Classes for representing solutions to boundary value problems.

@author : davidrpugh

"""


class SolutionLike(object):

    @property
    def basis_kwargs(self):
        return self._basis_kwargs

    @property
    def functions(self):
        return self._functions

    @property
    def nodes(self):
        return self._nodes

    @property
    def problem(self):
        return self._problem

    @property
    def residual_function(self):
        return self._residual_function

    @property
    def result(self):
        return self._result


class Solution(SolutionLike):
    """Class representing the solution to a Boundary Value Problem (BVP)."""

    def __init__(self, basis_kwargs, functions, nodes, problem, residual_function, result):
        """
        Initialize an instance of the Solution class.

        Parameters
        ----------
        basis_kwargs : dict
        functions : list
        nodes : numpy.ndarray
        problem : TwoPointBVPLike
        residual_function : callable
        result : OptimizeResult

        """
        self._basis_kwargs = basis_kwargs
        self._functions = functions
        self._nodes = nodes
        self._problem = problem
        self._residual_function = residual_function
        self._result = result

    def evaluate_residual(self, points):
        return self.residual_function(points)

    def evaluate_solution(self, points):
        return [f(points) for f in self.functions]

    def normalize_residuals(self, points):
        """Normalize residuals by the level of the variable."""
        residuals = self.evaluate_residual(points)
        solutions = self.evaluate_solution(points)
        return [resid / soln for resid, soln in zip(residuals, solutions)]

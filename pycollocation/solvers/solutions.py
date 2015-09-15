from . import bases


class Solution(bases.SolutionLike):
    """
    Class representing the solution to a Boundary Value Problem (BVP).

    Attributes
    ----------
    basis_kwargs : dict
    functions : list
    nodes : ndarray
    problem : TwoPointBVPLike
    residual_function : callable
    result : OptimizeResult

    """
    def evaluate_residual(self, points):
        return self.residual_function(points)

    def evaluate_solution(self, points):
        return [f(points) for f in self.functions]

    def normalize_residuals(self, points):
        """Normalize residuals by the level of the variable."""
        residuals = self.evaluate_residual(points)
        solutions = self.evaluate_solution(points)
        return [resid / soln for resid, soln in zip(residuals, solutions)]

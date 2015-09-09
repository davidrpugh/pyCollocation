import collections


SolutionBase = collections.namedtuple("SolutionBase",
                                      field_names=['basis_kwargs',
                                                   'domain',
                                                   'functions',
                                                   'nodes',
                                                   'problem',
                                                   'residuals',
                                                   'result',
                                                   ],
                                      )


class Solution(SolutionBase):
    """
    Class representing the solution to a Boundary Value Problem (BVP).

    Attributes
    ----------
    basis_kwargs : dict
    domain : tuple
    functions : list
    nodes : ndarray
    problem : TwoPointBVPLike
    residuals : callable
    result : OptimizeResult

    """

    def normalized_residuals(self, points):
        """Normalize residuals by the level of the variable."""
        residuals = self.residuals(points)
        variables = [soln_func(points) for soln_func in self.functions]
        return [resid / variable for resid, variable in zip(residuals, variables)]

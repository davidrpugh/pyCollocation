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
    Represents the solution to a Two-Point Boundary Value Problem (BVP).

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

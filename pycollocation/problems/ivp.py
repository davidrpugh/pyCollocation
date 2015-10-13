"""
Classes for modeling initial value problems.

@author : David R. Pugh

"""
from . bvp import TwoPointBVP


class IVP(TwoPointBVP):
    r"""
    Class for modeling Initial Value Problems (IVPs).

    Attributes
    ----------
    bcs_lower : function
        Function that calculates the difference between the lower boundary
        conditions and the current values of the model dependent variables.
    number_bcs_lower : int
        The number of lower boundary conditions (BCS).
    number_odes : int
        The number of Ordinary Differential Equations (ODEs) in the system.
    params : dict(str: float)
        A dictionary of model parameters.
    rhs : function
        Function which calculates the value of the right-hand side of a
        system of Ordinary Differential Equations (ODEs).

    """

    def __init__(self, bcs_lower, number_bcs_lower, number_odes, params, rhs):
        super(IVP, self).__init__(bcs_lower, None, number_bcs_lower,
                                  number_odes, params, rhs)

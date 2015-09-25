"""
Classes for representing boundary value problems.

@author : David R. Pugh

"""
import collections


TwoPointBVPBase = collections.namedtuple("TwoPointBVPBase",
                                         field_names=['bcs_lower',
                                                      'bcs_upper',
                                                      'number_bcs_lower',
                                                      'number_odes',
                                                      'params',
                                                      'rhs',
                                                      ],
                                         )


class TwoPointBVP(TwoPointBVPBase):
    r"""
    Class for representing Two-Point Boundary Value Problems (BVP).

    Attributes
    ----------
    bcs_lower : function
        Function that calculates the difference between the lower boundary
        conditions and the current values of the model dependent variables.
    bcs_upper : function
        Function that calculates the difference between the upper boundary
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

    def __new__(cls, bcs_lower, bcs_upper, number_bcs_lower, number_odes,
                params, rhs):
        return super(TwoPointBVP, cls).__new__(cls, bcs_lower, bcs_upper,
                                               number_bcs_lower, number_odes,
                                               params, rhs)


class IVP(TwoPointBVPBase):
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

    def __new__(cls, bcs_lower, number_bcs_lower, number_odes,
                params, rhs):
        return super(IVP, cls).__new__(cls, bcs_lower, None, number_bcs_lower,
                                       number_odes, params, rhs)

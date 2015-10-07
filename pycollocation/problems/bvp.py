"""
Classes for representing boundary value problems.

@author : David R. Pugh

"""


class TwoPointBVPLike(object):

    @property
    def bcs_lower(self):
        """
        Function that calculates the difference between the lower boundary
        conditions and the current values of the model dependent variables.

        :getter: Return the current boundary condition.
        :type: function

        """
        return self._bcs_lower

    @property
    def bcs_upper(self):
        """
        Function that calculates the difference between the upper boundary
        conditions and the current values of the model dependent variables.

        :getter: Return the current boundary condition.
        :type: function

        """
        return self._bcs_upper

    @property
    def number_bcs_lower(self):
        """
        The number of lower boundary conditions (BCS).

        :getter: Return the number of lower BCS.
        :type: int

        """
        return self._number_bcs_lower

    @property
    def number_odes(self):
        """
        The number of Ordinary Differential Equations (ODEs) in the system.

        :getter: Return the number of ODEs.
        :type: int

        """
        return self._number_odes

    @property
    def params(self):
        """
        A dictionary of parameters.

        :getter: Return the current parameters.
        :type: function

        """
        return self._params

    @property
    def rhs(self):
        """
        Function which calculates the value of the right-hand side of a
        system of Ordinary Differential Equations (ODEs).

        :getter: Return the current rhs.
        :type: function

        """
        return self._rhs


class TwoPointBVP(TwoPointBVPLike):
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
        A dictionary of parameters.
    rhs : function
        Function which calculates the value of the right-hand side of a
        system of Ordinary Differential Equations (ODEs).

    """

    def __init__(self, bcs_lower, bcs_upper, number_bcs_lower, number_odes,
                 params, rhs):
        self._bcs_lower = bcs_lower
        self._bcs_upper = bcs_upper
        self._number_bcs_lower = number_bcs_lower
        self._number_odes = number_odes
        self._params = dict(params)  # shallow copy!
        self._rhs = rhs

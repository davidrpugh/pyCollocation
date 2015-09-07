"""
Classes for representing two-point boundary value problems.

@author : David R. Pugh

"""


class TwoPointBVPLike(object):

    @property
    def bcs_lower(self):
        r"""
        Function that calculates the difference between the lower boundary
        conditions and the current values of the model dependent variables.

        :getter: Return the function defining the lower boundary conditions.
        :type: callable

        """
        return self._bcs_lower

    @property
    def bcs_lower_jac(self):
        return self._bcs_lower_jac

    @property
    def bcs_upper(self):
        r"""
        Function that calculates the difference between the upper boundary
        conditions and the current values of the model dependent variables.

        :getter: Return the function defining the upper boundary conditions.
        :type: callable

        """
        return self._bcs_upper

    @property
    def bcs_upper_jac(self):
        return self._bcs_upper_jac

    @property
    def number_odes(self):
        r"""
        Dimensionality of the system of Ordinary Differential Equations (ODEs).

        :getter: Return the current number of ODEs in the system.
        :type: int

        """
        return self._number_odes

    @property
    def number_bcs_lower(self):
        r"""
        Number of left boundary conditions (BCS).

        :getter: Return the current number of left boundary conditions.
        :type: int

        """
        return self._number_bcs_lower

    @property
    def params(self):
        r"""
        Dictionary of model parameters.

        :getter: Return the current dictionary of parameters.
        :type: dict

        """
        return self._params

    @property
    def rhs(self):
        r"""
        A function which calculates the value of the right-hand side of the
        system of Ordinary Differential Equations (ODEs).

        :getter: Return the current right-hand side of the system of ODEs.
        :types: callable.

        """
        return self._rhs

    @property
    def rhs_jac(self):
        return self._rhs_jac


class TwoPointBVP(TwoPointBVPLike):
    """Represents a Two Point Boundary Value Problem (BVP)."""

    def __init__(self, bcs_lower, bcs_upper, number_bcs_lower, number_odes,
                 rhs, params=None, bcs_lower_jac=None, bcs_upper_jac=None,
                 rhs_jac=None):
        """Create an instance of the TwoPointBVP class."""
        self._bcs_lower = bcs_lower
        self._bcs_lower_jac = bcs_lower_jac
        self._bcs_upper = bcs_upper
        self._bcs_upper_jac = bcs_upper_jac
        self._number_bcs_lower = number_bcs_lower
        self._number_odes = number_odes
        self._params = params
        self._rhs = rhs
        self._rhs_jac = rhs_jac


class IVP(TwoPointBVP):
    """Represents an Initial Value Problem (IVP)."""

    def __init__(self, bcs_lower, number_bcs_lower, number_odes, rhs,
                 params=None, bcs_lower_jac=None, rhs_jac=None):
        """Create an instance of the IVP class."""
        super(IVP, self).__init__(bcs_lower, None, number_bcs_lower,
                                  number_odes, rhs, params, bcs_lower_jac,
                                  None, rhs_jac)

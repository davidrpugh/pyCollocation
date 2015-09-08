"""
Classes for representing boundary value problems.

@author : David R. Pugh

"""
import collections


TwoPointBVPBase = collections.namedtuple("TwoPointBVPBase",
                                         field_names=['bcs_lower',
                                                      'bcs_lower_jac',
                                                      'bcs_upper',
                                                      'bcs_upper_jac',
                                                      'number_bcs_lower',
                                                      'number_odes',
                                                      'params',
                                                      'rhs',
                                                      'rhs_jac',
                                                      ],
                                         )


class TwoPointBVP(TwoPointBVPBase):

    def __new__(cls, bcs_lower, bcs_upper, number_bcs_lower, number_odes,
                params, rhs, bcs_lower_jac=None, bcs_upper_jac=None,
                rhs_jac=None):
        r"""
        Return a new instance of the TwoPointBVP class.

        Parameters
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
        bcs_lower_jac : function (optional, default=None)
        bcs_upper_jac : function (optional, default=None)
        rhs_jac : function (optional, default=None)

        """
        return super(TwoPointBVP, cls).__new__(cls, bcs_lower, bcs_lower_jac,
                                               bcs_upper, bcs_upper_jac,
                                               number_bcs_lower, number_odes,
                                               params, rhs, rhs_jac)


class IVP(TwoPointBVPBase):

    def __new__(cls, bcs_lower, number_bcs_lower, number_odes,
                params, rhs, bcs_lower_jac=None, bcs_upper_jac=None,
                rhs_jac=None):
        r"""
        Return a new instance of the IVP class.

        Parameters
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
        bcs_lower_jac : function (optional, default=None)
        rhs_jac : function (optional, default=None)

        """
        return super(IVP, cls).__new__(cls, bcs_lower, bcs_lower_jac, None,
                                       None, number_bcs_lower, number_odes,
                                       params, rhs, rhs_jac)

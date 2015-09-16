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
    bcs_lower_jac : function (optional, default=None)
    bcs_upper_jac : function (optional, default=None)
    rhs_jac : function (optional, default=None)

    """

    def __new__(cls, bcs_lower, bcs_upper, number_bcs_lower, number_odes,
                params, rhs, bcs_lower_jac=None, bcs_upper_jac=None,
                rhs_jac=None):
        return super(TwoPointBVP, cls).__new__(cls, bcs_lower, bcs_lower_jac,
                                               bcs_upper, bcs_upper_jac,
                                               number_bcs_lower, number_odes,
                                               params, rhs, rhs_jac)


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
    bcs_lower_jac : function (optional, default=None)
    rhs_jac : function (optional, default=None)

    """

    def __new__(cls, bcs_lower, number_bcs_lower, number_odes,
                params, rhs, bcs_lower_jac=None, rhs_jac=None):
        return super(IVP, cls).__new__(cls, bcs_lower, bcs_lower_jac, None,
                                       None, number_bcs_lower, number_odes,
                                       params, rhs, rhs_jac)

import sympy as sym


class SymbolicTwoPointBVP(TwoPointBVPBase):
    """How to leverage duck typing?"""

    def __new__(cls, bcs_lower, bcs_upper, independent_var, dependent_vars,
                number_bcs_lower, number_odes, params, rhs):
        funcs = [bcs_lower, bcs_upper, rhs]  # note the order!
        exprs = cls._generate_symbolic_exprs(funcs, independent_var,
                                             dependent_vars, params)
        args = cls._generate_symbolic_args(independent_var, dependent_vars,
                                           params)
        lambified_funcs = cls._lambda_function_factory(args, exprs)

        jac_exprs = cls._generate_symbolic_jacs(exprs, dependent_vars)
        lambdified_jacs = cls._lambda_function_factory(args, jac_exprs)

        return super(TwoPointBVP, cls).__new__(cls, lambdified_bcs_lower, lambdified_bcs_lower_jac,
                                               lambdified_bcs_upper, lambdified_bcs_upper_jac,
                                               number_bcs_lower, number_odes, params
                                               lambdified_rhs, lamdified_rhs_jac)

    @classmethod
    def _combine_symbolic_args(cls, variables, params):
        return variables + params

    @staticmethod
    def _generate_symbolic_args(independent_var, dependent_vars, params):
        symbolic_params = cls._generate_symbolic_params(params)
        symbolic_vars = cls._combine_symbolic_vars(independent_var,
                                                   dependent_vars)
        symbolic_args = cls._combine_symbolic_args(symbolic_vars,
                                                   symbolic_params)
        return symbolic_args

    @staticmethod
    def _generate_symbolic_exprs(funcs, independent_var, dependent_vars, params):
        symbolic_params = {k: sym.symbols(k) for k in params.keys()}  # hash order is irrelevant!
        return [f(independent_var, *dependent_vars, **symbolic_params) for f in funcs]

    @staticmethod
    def _generate_symbolic_jacs(exprs, dependent_vars):
        matrices = [sym.Matrix(expr) for expr in exprs]
        return [matrix.jacobian(dependent_vars) for matrix in matrices]

    @staticmethod
    def _combine_symbolic_vars(independent_var, dependent_vars):
        return [independent_var] + dependent_vars

    @staticmethod
    def _generate_symbolic_params(params):
        return [sym.symbols(p) for p in params.keys()]  # hash order is crucial!

    @staticmethod
    def _lambda_function_factory(args, exprs, modules=None, printer=None,
                                 use_imps=True, dummify=False):
        return [sym.lambdify(args, expr, modules, dummify) for expr in exprs]

    @staticmethod
    def _func_factory(lamdified_funcs):
        def func(independent_var, *dependent_vars, **params):
            return [f(independent_var, *dependent_vars, **params) for f in lamdified_funcs]
        return func

    @staticmethod
    def _jac_factory(lamdified_jac):
        def jac(independent_var, *dependent_vars, **params):
            return lamdified_jac(independent_var, *dependent_vars, **params)
        return jac

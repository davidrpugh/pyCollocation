"""
Classes for constructing systems of ordinary differential equations.

@author : davidrpugh

"""
import sympy as sym

from . import models
from . import symbolics


class DifferentialEquation(models.ModelLike):

    def __init__(self, dependent_vars, independent_var, rhs, params):
        """Create an instance of the DifferentialEquation class."""
        self._dependent_vars = self._validate_variables(dependent_vars)
        self._independent_var = self._validate_variable(independent_var)
        self._rhs = self._validate_rhs(rhs)
        self.params = params

    def _validate_rhs(self, rhs):
        """Validate the rhs attribute."""
        if not isinstance(rhs, dict):
            mesg = "Attribute `rhs` must be of type `dict` not {}"
            raise AttributeError(mesg.format(rhs.__class__))
        elif not (len(rhs) == len(self.dependent_vars)):
            mesg = "Number of equations must equal number of dependent vars."
            raise ValueError(mesg)
        else:
            return rhs

    @staticmethod
    def _validate_variable(symbol):
        """Validate the independent_var attribute."""
        if not isinstance(symbol, str):
            mesg = "Attribute must be of type `string` not {}"
            raise AttributeError(mesg.format(symbol.__class__))
        else:
            return symbol

    @classmethod
    def _validate_variables(cls, symbols):
        """Validate the dependent_vars attribute."""
        return [cls._validate_variable(symbol) for symbol in symbols]


class SymbolicDifferentialEquation(DifferentialEquation,
                                   symbolics.SymbolicModelLike):

    _cached_rhs_functions = {}  # not sure if this is good practice!

    def _rhs_functions(self, var):
        """Cache lamdified rhs functions for numerical evaluation."""
        if self._cached_rhs_functions.get(var) is None:
            eqn = self.rhs[var]
            self._cached_rhs_functions[var] = self._lambdify_factory(eqn)
        return self._cached_rhs_functions[var]

    @staticmethod
    def _validate_expression(expression):
        """Validates a symbolic expression."""
        if not isinstance(expression, sym.Basic):
            mesg = "Attribute must be of type `sympy.Basic` not {}"
            raise AttributeError(mesg.format(expression.__class__))
        else:
            return expression

    def _validate_rhs(self, rhs):
        """Validate a the rhs attribute."""
        super(SymbolicDifferentialEquation, self)._validate_rhs(rhs)
        exprs = {}
        for var, expr in rhs.items():
            exprs[var] = self._validate_expression(expr)
        else:
            return exprs

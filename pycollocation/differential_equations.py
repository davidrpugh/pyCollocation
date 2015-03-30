"""
Classes for constructing systems of ordinary differential equations.

@author : davidrpugh

"""
import sympy as sym

from . import models
from . import symbolics


class SymbolicDifferentialEquation(models.ModelLike,
                                   symbolics.SymbolicModelLike):

    _cached_rhs_functions = {}  # not sure if this is good practice!

    def __init__(self, dependent_vars, independent_var, rhs, params):
        """Create an instance of the DifferentialEquation class."""
        self._dependent_vars = self._validate_symbols(dependent_vars)
        self._independent_var = self._validate_symbol(independent_var)
        self._rhs = self._validate_rhs(rhs)
        self.params = self._validate_params(params)

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

    @staticmethod
    def _validate_symbol(symbol):
        """Validate the independent_var attribute."""
        if not isinstance(symbol, sym.Symbol):
            mesg = "Attribute must be of type `sympy.Symbol` not {}"
            raise AttributeError(mesg.format(symbol.__class__))
        else:
            return symbol

    @classmethod
    def _validate_symbols(cls, symbols):
        """Validate the dependent_vars attribute."""
        return [cls._validate_symbol(symbol) for symbol in symbols]

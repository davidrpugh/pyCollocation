"""
Classes for representing differential and difference equations.

@author : David R. Pugh

"""
import collections

import sympy as sym

from . import symbolics


class ModelLike(object):

    @property
    def dependent_vars(self):
        """
        Model dependent variables.

        :getter: Return the model dependent variables.
        :type: list

        """
        return self._dependent_vars

    @property
    def independent_var(self):
        """
        Model independent variable.

        :getter: Return the model independent variable.
        :type: string

        """
        return self._independent_var

    @property
    def params(self):
        """
        Dictionary of model parameters.

        :getter: Return the model parameters as an ordered dictionary.
        :setter: Set a new parameter dictionary.
        :type: dict

        """
        return self._params

    @params.setter
    def params(self, value):
        """Set a new parameter dictionary."""
        valid_params = self._validate_params(value)
        self._params = self._order_params(valid_params)

    @property
    def rhs(self):
        """
        Right-hand side of the system of differential/difference equations.

        :getter: Return the right-hand side of the system of equations.
        :type: dict

        """
        return self._rhs

    @staticmethod
    def _order_params(params):
        """Cast a dictionary to an order dictionary."""
        return collections.OrderedDict(sorted(params.items()))

    @staticmethod
    def _validate_params(value):
        """Validate the dictionary of parameters."""
        if not isinstance(value, dict):
            mesg = "Attribute 'params' must have type dict, not {}"
            raise AttributeError(mesg.format(value.__class__))
        else:
            return value

    def _validate_rhs(self, rhs):
        """Validate the rhs attribute."""
        if not isinstance(rhs, dict):
            mesg = "Attribute `rhs` must be of type `dict` not {}"
            raise AttributeError(mesg.format(rhs.__class__))
        elif not (len(rhs.keys()) == len(self.dependent_vars)):
            mesg = "Number of equations must equal number of dependent vars."
            raise ValueError(mesg)
        else:
            return rhs

    @staticmethod
    def _validate_variable(variable):
        """Validate a single variable."""
        if not isinstance(variable, str):
            mesg = "Attribute must be of type `string` not {}"
            raise AttributeError(mesg.format(variable.__class__))
        else:
            return variable

    @classmethod
    def _validate_variables(cls, variables):
        """Validate a collection of variables."""
        return [cls._validate_variable(variable) for variable in variables]


class SymbolicModelLike(symbolics.SymbolicLike, ModelLike):

    _cached_rhs_functions = collections.defaultdict(lambda: None)

    @property
    def _symbolic_args(self):
        """List of symbolic arguments used to lambdify expressions."""
        return self._symbolic_vars + list(self._symbolic_params.values())

    @property
    def _symbolic_jacobian(self):
        """Symbolic Jacobian matrix of partial derivatives."""
        args = self.dependent_vars
        return self._symbolic_system.jacobian(args)

    @property
    def _symbolic_params(self):
        """List of symbolic model parameters."""
        return collections.OrderedDict((k, sym.symbols(k)) for (k, v) in self.params.items())

    @property
    def _symbolic_system(self):
        """Represents rhs as a symbolic matrix."""
        return sym.Matrix([self.rhs[var] for var in self.dependent_vars])

    @property
    def _symbolic_vars(self):
        """List of symbolic model variables."""
        return sym.symbols([self.independent_var] + self.dependent_vars)

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
        super(SymbolicModelLike, self)._validate_rhs(rhs)
        exprs = {}
        for var, expr in rhs.items():
            exprs[var] = self._validate_expression(expr)
        return exprs

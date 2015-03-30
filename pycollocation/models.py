import collections

import sympy as sym

from . import symbolics


class Model(object):

    def __init__(self, dependent_vars, independent_var, rhs, params):
        """Create an instance of the Model class."""
        self._dependent_vars = self._validate_symbols(dependent_vars)
        self._independent_var = self._validate_symbol(independent_var)
        self._rhs = self._validate_rhs(rhs)
        self.params = self._validate_params(params)

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
        Symbolic variable representing the independent variable.

        :getter: Return the symbol representing the independent variable.
        :type: sympy.Symbol

        """
        return self._independent_var

    @property
    def params(self):
        """
        Dictionary of model parameters.

        :getter: Return the current parameter dictionary.
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
        Symbolic representation of the right-hand side of a system of
        differential/difference equations.

        :getter: Return the right-hand side of the system of equations.
        :type: dict

        """
        return self._rhs

    @staticmethod
    def _order_params(params):
        """Cast a dictionary to an order dictionary."""
        return collections.OrderedDict(sorted(params.items()))

    @staticmethod
    def _validate_expression(expression):
        """Validates a symbolic expression."""
        if not isinstance(expression, sym.Basic):
            mesg = "Attribute must be of type `sympy.Basic` not {}"
            raise AttributeError(mesg.format(expression.__class__))
        else:
            return expression

    @staticmethod
    def _validate_params(value):
        """Validate the dictionary of parameters."""
        if not isinstance(value, dict):
            mesg = "Attribute 'params' must have type dict, not {}"
            raise AttributeError(mesg.format(value.__class__))
        else:
            return value

    def _validate_rhs(self, rhs):
        """Validate a the rhs attribute."""
        if not isinstance(rhs, dict):
            mesg = "Attribute `rhs` must be of type `dict` not {}"
            raise AttributeError(mesg.format(rhs.__class__))
        elif not (len(rhs) == len(self.dependent_vars)):
            mesg = "Number of equations must equal number of dependent vars."
            raise ValueError(mesg)
        else:
            exprs = {}
            for var, expr in rhs.items():
                exprs[var] = self._validate_expression(expr)
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


class SymbolicModel(symbolics.SymbolicModelLike, Model):
    """Base class for all symbolic models."""

    _cached_rhs_functions = {}  # not sure if this is good practice!

    def _rhs_functions(self, var):
        """Cache lamdified rhs functions for numerical evaluation."""
        if self._cached_rhs_functions.get(var) is None:
            eqn = self.rhs[var]
            self._cached_rhs_functions[var] = self._lambdify_factory(eqn)
        return self._cached_rhs_functions[var]

    def _validate_rhs(self, rhs):
        """Validate a the rhs attribute."""
        if not isinstance(rhs, dict):
            mesg = "Attribute `rhs` must be of type `dict` not {}"
            raise AttributeError(mesg.format(rhs.__class__))
        elif not (len(rhs) == len(self.dependent_vars)):
            mesg = "Number of equations must equal number of dependent vars."
            raise ValueError(mesg)
        else:
            exprs = {}
            for var, expr in rhs.items():
                exprs[var] = self._validate_expression(expr)
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


class DifferentialEquation(SymbolicModel):
    pass


class BoundaryValueProblem(DifferentialEquation):
    """Class for representing two-point boundary value problems."""

    __lower_boundary_condition = None

    __upper_boundary_condition = None

    def __init__(self, boundary_conditions, dependent_vars, independent_var,
                 rhs, params):
        """
        Create an instance of a two-point boundary value problem (BVP).

        """
        super(BoundaryValueProblem, self).__init__(dependent_vars,
                                                   independent_var,
                                                   rhs,
                                                   params)
        self.boundary_conditions = boundary_conditions

    @property
    def _lower_boundary_condition(self):
        """Cache lambdified lower boundary condition for numerical evaluation."""
        condition = self.boundary_conditions['lower']
        if condition is not None:
            if self.__lower_boundary_condition is None:
                self.__lower_boundary_condition = self._lambdify_factory(condition)
            return self.__lower_boundary_condition
        else:
            return None

    @property
    def _upper_boundary_condition(self):
        """Cache lambdified upper boundary condition for numerical evaluation."""
        condition = self.boundary_conditions['upper']
        if condition is not None:
            if self.__upper_boundary_condition is None:
                self.__upper_boundary_condition = self._lambdify_factory(condition)
            return self.__upper_boundary_condition
        else:
            return None

    @property
    def boundary_conditions(self):
        """
        Boundary conditions for the problem.

        :getter: Return the boundary conditions for the problem.
        :setter: Set new boundary conditions for the problem.
        :type: dict

        """
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, conditions):
        """Set new boundary conditions for the model."""
        self._boundary_conditions = self._validate_boundary(conditions)

    def _sufficient_boundary(self, conditions):
        """Check that there are sufficient boundary conditions."""
        number_conditions = 0
        if conditions['lower'] is not None:
            number_conditions += len(conditions['lower'])
        if conditions['upper'] is not None:
            number_conditions += len(conditions['upper'])
        return number_conditions == len(self.dependent_vars)

    def _validate_boundary(self, conditions):
        """Validate a dictionary of lower and upper boundary conditions."""
        if not isinstance(conditions, dict):
            mesg = "Attribute `boundary_conditions` must have type `dict` not {}"
            raise AttributeError(mesg.format(conditions.__class__))
        elif not (set(conditions.keys()) < set(['lower', 'upper', None])):
            mesg = "Keys for `boundary_conditions` dict must be {}, {}, or {}"
            raise AttributeError(mesg.format('lower', 'upper', 'None'))
        elif not self._sufficient_boundary(conditions):
            mesg = "Number of conditions must equal number of dependent vars."
            raise ValueError(mesg)
        else:
            bcs = {'lower': self._validate_boundary_exprs(conditions['lower']),
                   'upper': self._validate_boundary_exprs(conditions['upper'])}
            return bcs

    def _validate_boundary_exprs(self, expressions):
        """Check that lower/upper boundary_conditions are expressions."""
        if expressions is None:
            return None
        else:
            return [self._validate_expression(expr) for expr in expressions]

import collections

import numpy as np
import sympy as sym

import models


class Solver(object):
    """Base class for all Solvers."""

    __lower_boundary_condition = None

    __upper_boundary_condition = None

    _cached_rhs_functions = {}  # not sure if this is good practice!

    _modules = [{'ImmutableMatrix': np.array}, 'numpy']

    def __init__(self, model, params):
        """
        Create an instance of the Solver class.

        Parameters
        ----------
        model : models.Model
            An instance of models.Model to solve.
        params : dict
            A dictionary of model parameters.

        """
        self.model = model
        self.params = params

    @property
    def _lower_boundary_condition(self):
        """Cache lambdified lower boundary condition for numerical evaluation."""
        condition = self.model.boundary_conditions['lower']
        if condition is not None:
            if self.__lower_boundary_condition is None:
                self.__lower_boundary_condition = self._lambdify_factory(condition)
            return self.__lower_boundary_condition
        else:
            return None

    @property
    def _symbolic_args(self):
        """List of symbolic arguments used to lambdify expressions."""
        return self._symbolic_vars + self._symbolic_params

    @property
    def _symbolic_params(self):
        """List of symbolic model parameters."""
        return sym.var(list(self.params.keys()))

    @property
    def _symbolic_vars(self):
        """List of symbolic model variables."""
        return [self.model.independent_var] + self.model.dependent_vars

    @property
    def _upper_boundary_condition(self):
        """Cache lambdified upper boundary condition for numerical evaluation."""
        condition = self.model.boundary_conditions['upper']
        if condition is not None:
            if self.__upper_boundary_condition is None:
                self.__upper_boundary_condition = self._lambdify_factory(condition)
            return self.__upper_boundary_condition
        else:
            return None

    @property
    def model(self):
        """
        Symbolic representation of the model to solve.

        :getter: Return the current model.
        :setter: Set a new model to solve.
        :type: models.Model

        """
        return self._model

    @model.setter
    def model(self, model):
        """Set a new model to solve."""
        self._model = self._validate_model(model)

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

    def _rhs_functions(self, symbol):
        """Cache lamdified rhs functions for numerical evaluation."""
        if self._cached_rhs_functions.get(symbol) is None:
            eqn = self.model.rhs[symbol]
            self._cached_rhs_functions[symbol] = self._lambdify_factory(eqn)
        return self._cached_rhs_functions[symbol]

    def _lambdify_factory(self, expr):
        """Lambdify a symbolic expression."""
        return sym.lambdify(self._symbolic_args, expr, self._modules)

    @staticmethod
    def _order_params(params):
        """Cast a dictionary to an order dictionary."""
        return collections.OrderedDict(sorted(params.items()))

    @staticmethod
    def _validate_model(model):
        """Validate the dictionary of parameters."""
        if not isinstance(model, models.SymbolicModel):
            mesg = "Attribute 'model' must have type models.SymbolicModel, not {}"
            raise AttributeError(mesg.format(model.__class__))
        else:
            return model

    @staticmethod
    def _validate_params(value):
        """Validate the dictionary of parameters."""
        if not isinstance(value, dict):
            mesg = "Attribute 'params' must have type dict, not {}"
            raise AttributeError(mesg.format(value.__class__))
        else:
            return value

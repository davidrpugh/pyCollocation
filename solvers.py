import collections

import numpy as np
import sympy as sym

import models


class Solver(object):
    """Base class for all Solvers."""

    _cached_functions = {}  # not sure if this is good practice!

    _modules = [{'ImmutableMatrix': np.array}, 'numpy']

    def __init__(self, model, params):
        """
        Create an instance of the Solver class.

        Parameters
        ----------
        model: models.Model
            An instance of models.Model to solve.
        params : dict
            A dictionary of model parameters.

        """
        self.model = model
        self.params = params

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

    def _function_factory(self, symbol, expr):
        if self._cached_functions.get(symbol) is None:
            self._cached_functions[symbol] = self._lambdify_factory(expr)
        return self._cached_functions[symbol]

    def _lambdify_factory(self, expr):
        """Lambdify a symbolic expression."""
        return sym.lambdify(self._symbolic_args, expr, self._modules)

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


if __name__ == '__main__':

    # define some variables
    t, k, c = sym.symbols('t, k, c')

    # define some parameters
    alpha, sigma = sym.symbols('alpha, sigma')
    g, n, s, delta = sym.symbols('g, n, s, delta')

    # intensive output has the CES form
    y = (alpha * k**((sigma - 1) / sigma) + (1 - alpha))**(sigma / (sigma - 1))

    # define the Solow model
    k_dot = s * y - (g + n + delta) * k

    solow = models.DifferentialEquation(dependent_vars=[k],
                                        independent_var=t,
                                        rhs=[k_dot])

    solow_params = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.15,
                    'sigma': 2.0, 'delta': 0.04}
    solow_solver = Solver(solow, solow_params)

    # define the Ramsey-Cass-Coopmans model
    rho, theta = sym.symbols('rho, theta')

    mpk = sym.diff(y, k, 1)
    k_dot = y - c - (g + n) * k
    c_dot = ((mpk - rho - theta * g) / theta) * c

    ramsey = models.DifferentialEquation(dependent_vars=[k, c],
                                         independent_var=t,
                                         rhs=[k_dot, c_dot])

    ramsey_params = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.15,
                     'sigma': 2.0, 'delta': 0.04, 'rho': 0.03, 'theta': 2.5}
    ramsey_solver = Solver(ramsey, ramsey_params)

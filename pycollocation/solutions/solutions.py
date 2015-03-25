"""
Base class for all Solution objects.

TODO:

1. Should `domain` be a generic attribute for all solvers?

"""
import numpy as np
import pandas as pd
from scipy import optimize


class Solution(object):
    """Base class for all Solution objects."""

    @property
    def _residuals(self):
        """Return the residuals stored as a dict of NumPy arrays."""
        tmp = {}
        for var, function in self.residual_functions.iteritems():
            tmp[var] = pd.Series(function(self.interpolation_knots),
                                 index=self.interpolation_knots)
        return tmp

    @property
    def _solution(self):
        """Return the solution stored as a dict of NumPy arrays."""
        tmp = {}
        for var, function in self.functions.iteritems():
            tmp[var] = pd.Series(function(self.interpolation_knots),
                                 index=self.interpolation_knots)
        return tmp

    @property
    def coefficients(self):
        """
        Coefficients to use when constructing the basis functions.

        :getter: Return the `coefficients` attribute.
        :type: dict

        """
        raise NotImplementedError

    @property
    def derivatives(self):
        """
        Derivatives of the approximating basis functions.

        :getter: Return the `derivatives` attribute.
        :type: dict

        """
        raise NotImplementedError

    @property
    def functions(self):
        """
        The basis functions used to approximate the solution to the model.

        :getter: Return the `functions` attribute.
        :type: dict

        """
        raise NotImplementedError

    @property
    def interpolation_knots(self):
        """
        Interpolation knots to use when computing the solution.

        :getter: Return the array of interpolation knots.
        :setter: Set a new array of interpolation knots.
        :type: numpy.ndarray

        """
        return self._interpolation_knots

    @interpolation_knots.setter
    def interpolation_knots(self, value):
        """Set new array of interpolation knots."""
        self._interpolation_knots = self._validate_interpolation_knots(value)

    @property
    def result(self):
        """
        An instance of the `optimize.optimize.OptimizeResult` class that stores
        the raw output of a `solvers.Solver` object.

        :getter: Return the `result` attribute.
        :type: `optimize.optimize.OptimizeResult`

        """
        return self._result

    @property
    def residual_functions(self):
        return self._construct_residual_funcs(self.derivatives, self.functions)

    @property
    def residuals(self):
        """
        Solution residuals represented as a Pandas `DataFrame`.

        :getter: Return the `DataFrame` representing the solution residuals.
        :type: `pandas.DataFrame`

        """
        return pd.DataFrame.from_dict(self._residuals)

    @property
    def solution(self):
        """
        Solution to the model represented as a Pandas `DataFrame`.

        :getter: Return the `DataFrame` representing the current solution.
        :type: `pandas.DataFrame`

        """
        return pd.DataFrame.from_dict(self._solution)

    @property
    def success(self):
        """
        True, if a solution was found by the `solvers.Solver` object;
        otherwise, False.

        :getter: Return the `success` attribute.
        :type: `boolean`

        """
        return self.result.success

    @staticmethod
    def _validate_interpolation_knots(number):
        """Validates the `interpolation_knots` attribute."""
        if not isinstance(number, np.ndarray):
            mesg = ("The 'number_knots' attribute must have type " +
                    "'numpy.ndarray', not {}.")
            raise AttributeError(mesg.format(number.__class__))
        else:
            return number

    @staticmethod
    def _validate_result(result):
        """Validates the `result` attribute."""
        if not isinstance(result, optimize.optimize.OptimizeResult):
            mesg = ("The 'result' attribute must have type " +
                    "'optimize.optimize.OptimizeResult', not {}.")
            raise AttributeError(mesg.format(result.__class__))
        else:
            return result

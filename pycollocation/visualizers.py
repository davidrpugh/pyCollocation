"""
Base class for all Visualizer objects.

"""
import numpy as np
import pandas as pd


class Visualizer(object):
    """Base class for all Visualizer objects."""

    def __init__(self, solver):
        """Create an instance of the Visualizer class."""
        self.solver = solver

    @property
    def _residuals(self):
        """Return the residuals stored as a dict of NumPy arrays."""
        tmp = {}
        for var, function in self.solver.residual_functions.items():
            tmp[var] = pd.Series(function(self.interpolation_knots),
                                 index=self.interpolation_knots)
        return tmp

    @property
    def _solution(self):
        """Return the solution stored as a dict of NumPy arrays."""
        tmp = {}
        for var, function in self.solver.functions.items():
            tmp[var] = pd.Series(function(self.interpolation_knots),
                                 index=self.interpolation_knots)
        return tmp

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
    def normalized_residuals(self):
        """
        Absolute values of the solution residuals normalized by the value of
        the solution.

        :getter: Return the normalized solution residuals.
        :type: `pandas.DataFrame`

        """
        return self.residuals.abs() / self.solution

    @property
    def residuals(self):
        """
        Solution residuals.

        :getter: Return the solution residuals.
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

    @staticmethod
    def _validate_interpolation_knots(number):
        """Validates the `interpolation_knots` attribute."""
        if not isinstance(number, np.ndarray):
            mesg = ("The 'number_knots' attribute must have type " +
                    "'numpy.ndarray', not {}.")
            raise AttributeError(mesg.format(number.__class__))
        else:
            return number

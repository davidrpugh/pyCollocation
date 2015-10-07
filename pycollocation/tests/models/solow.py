import functools

from ... import problems


class SolowModel(problems.IVP):
    """
    Class representing a generic Solow growth model.

    Attributes
    ----------
    equilibrium_capital : function
        Equilibrium value for capital (per unit effective labor).
    intensive_output : function
        Output (per unit effective labor supply).
    params : dict(str: float)
        Dictionary of model parameters.

    """

    def __init__(self, f, k_star, params):
        """
        Initialize an instance of the SolowModel class.

        Parameters
        ----------
        f : function
            Output (per unit effective labor supply).
        k_star : function
            Equilibrium value for capital (per unit effective labor).
        params : dict(str: float)
            Dictionary of model parameters.

        """
        rhs = self._rhs_factory(f)
        self._equilbrium_capital = k_star
        self._intensive_output = f
        super(SolowModel, self).__init__(self._initial_condition, 1, 1, params, rhs)

    @property
    def equilibrium_capital(self):
        return self._equilbrium_capital

    @property
    def intensive_output(self):
        return self._intensive_output

    @staticmethod
    def _initial_condition(t, k, k0, **params):
        return [k - k0]

    @classmethod
    def _solow_model(cls, t, k, f, delta, g, n, s, **params):
        return [s * f(k, **params) - (g + n + delta) * k]

    @classmethod
    def _rhs_factory(cls, f):
        return functools.partial(cls._solow_model, f=f)

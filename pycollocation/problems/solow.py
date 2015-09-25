import functools

import numpy as np

from . import ivp


class IVP(ivp.IVP):
    """
    Class representing a generic Solow growth model.

    Attributes
    ----------
    f: function
        Output (per unit effective labor supply).
    mpk : function
        Marginal product of capital (per unit effective labor supply).
    params : dict(str: float)
        Dictionary of model parameters.

    """

    def __init__(self, f, params):
        bcs_lower = lambda t, k, k0, **params: [k - k0]
        rhs = self._rhs_factory(f)
        super(IVP, self).__init__(bcs_lower, 1, 1, params, rhs)

    def __repr__():
        pass

    def __str__():
        pass

    @staticmethod
    def _solow_model(t, k, f, delta, g, n, s, **params):
        return [s * f(k, **params) - (g + n + delta) * k]

    @classmethod
    def _rhs_factory(cls, f):
        return functools.partial(cls._solow_model, f=f)


class InitialPoly(object):

    def __init__(self, equilibrium_capital):
        self._equilibrium_capital = equilibrium_capital

    def create_mesh(self, basis_kwargs, num, problem):
        ts = np.linspace(*basis_kwargs['domain'], num=num)
        kstar = self._equilibrium_capital(**problem.params)
        ks = kstar - (kstar - problem.params['k0']) * np.exp(-ts)
        return ts, ks

    def fit(self, basis_kwargs, num, problem):
        ts, ks = self.create_mesh(basis_kwargs, num, problem)
        basis_poly = getattr(np.polynomial, basis_kwargs['kind'])
        return basis_poly.fit(ts, ks, basis_kwargs['degree'], basis_kwargs['domain'])

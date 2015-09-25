import functools

import numpy as np

from . import bvp


class IVP(bvp.IVP):
    """
    Class representing a generic Solow growth model.

    Attributes
    ----------
    f: function
        Output (per unit effective labor supply).
    mpk : function
        Marginal product of capital (per unit effective labor supply).
    params : dict(str: float)

    """

    def __new__(cls, f, params):
        rhs = cls._rhs_factory(f)
        return super(IVP, cls).__new__(cls, cls._bcs_lower, 1, 1, params, rhs)

    def __repr__():
        pass

    def __str__():
        pass

    @staticmethod
    def _bcs_lower(t, k, k0, **params):
        return [k - k0]

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

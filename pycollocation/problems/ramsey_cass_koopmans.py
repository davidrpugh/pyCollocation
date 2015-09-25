import functools

import numpy as np

from . import bvp


class TwoPointBVP(bvp.TwoPointBVP):
    """
    Class representing a generic Ramsey-Cass-Koopmans optimal savings model.

    Attributes
    ----------
    A : function
        Pratt-Arrow absolute risk aversion function.
    A_prime: function
        First derivative of the Pratt-Arrow absolute risk aversion function.
    c_star : function
        Equilibrium (i.e., steady-state) value for consumption (per unit
        effective labor supply).
    f: function
        Output (per unit effective labor supply).
    k_star : function
        Equilibrium (i.e., steady-state) value for capital stock (per unit
        effective labor supply).
    mpk : function
        Marginal product of capital (per unit effective labor supply).
    mpk_prime : function
        First derivative of mpk (i.e., second derivative of f).
    params : dict(str: float)

    """

    def __new__(cls, A, A_prime, c_star, f, k_star, mpk, mpk_prime, params):
        bcs_upper = cls._bcs_upper_factory(c_star)
        rhs = cls._rhs_factory(A, A_prime, f, mpk, mpk_prime)
        rhs_jac = cls._rhs_jac_factory(A, A_prime, mpk, mpk_prime)
        return super(TwoPointBVP, cls).__new__(cls, cls._bcs_lower, bcs_upper,
                                               cls._bcs_upper_jac, 1, 1, params,
                                               rhs, cls._bcs_lower_jac, rhs_jac)

    def __repr__():
        pass

    def __str__():
        pass

    @staticmethod
    def _actual_investment(k, c, f, **params):
        return f(k, **params) - c

    @staticmethod
    def _break_even_investment(k, delta, g, n, **params):
        return (g + n + delta) * k

    @staticmethod
    def _bcs_lower(t, k, c, k0, **params):
        return [k - k0]

    @staticmethod
    def _bcs_lower_jac(t, k, c, k0, **params):
        return [[1.0, 0.0]]

    @staticmethod
    def _c_star_residual(t, k, c, c_star, **params):
        return [c - c_star(**params)]

    @staticmethod
    def _bcs_upper_factory(c_star):
        return functools.partial(TwoPointBVP._c_star_residual, c_star=c_star)

    @staticmethod
    def _bcs_upper_jac(t, k, c, **params):
        return [[0.0, 1.0]]

    @staticmethod
    def _c_dot(t, k, c, A, mpk, delta, rho, **params):
        return (mpk(k, **params) - delta - rho) / A(c, **params)

    @staticmethod
    def _k_dot(t, k, c, f, delta, g, n, **params):
        """Equation of motion for capital (per unit effective labor supply)."""
        out = (TwoPointBVP._actual_investment(k, c, **params) -
               TwoPointBVP._break_even_investment(k, delta, g, n))
        return out

    @staticmethod
    def _ramsey_model(t, k, c, A, f, mpk, delta, g, n, rho, **params):
        out = [TwoPointBVP._k_dot(t, k, c, f, delta, g, n, **params),
               TwoPointBVP._c_dot(t, k, c, A, mpk, delta, rho, **params)]
        return out

    @staticmethod
    def _ramsey_model_jac(t, k, c, A, A_prime, mpk, mpk_prime, delta, g, n, rho, **params):
        jac = [[mpk(k, **params) - (g + n + delta), -1],
               [mpk_prime(k, **params) / A(c, **params), -(mpk(k, **params) - delta - rho) * (A_prime(c, **params) / A(c, **params)**2)]]
        return jac

    @staticmethod
    def _rhs_factory(A, A_prime, f, mpk, mpk_prime):
        rhs = functools.partial(TwoPointBVP._ramsey_model, A=A, A_prime=A_prime,
                                f=f, mpk=mpk, mpk_prime=mpk_prime)
        return rhs

    @staticmethod
    def _rhs_jac_factory(A, A_prime, mpk, mpk_prime):
        rhs_jac = functools.partial(TwoPointBVP._ramsey_model_jac, A=A,
                                    A_prime=A_prime, mpk=mpk,
                                    mpk_prime=mpk_prime)
        return rhs_jac


class InitialPoly(object):

    @staticmethod
    def create_mesh(self, basis_kwargs, num, problem):
        ts = np.linspace(*basis_kwargs['domain'], num=num)
        kstar = self._equilibrium_capital(**problem.params)
        ks = kstar - (kstar - problem.params['k0']) * np.exp(-ts)
        return ts, ks

    def fit(self, basis_kwargs, num, problem):
        ts, ks = self.create_mesh(basis_kwargs, num, problem)
        basis_poly = getattr(np.polynomial, basis_kwargs['kind'])
        return basis_poly.fit(ts, ks, basis_kwargs['degree'], basis_kwargs['domain'])

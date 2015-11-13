import functools

import numpy as np

from ... import problems


class RamseyCassKoopmansModel(problems.TwoPointBVP):
    """
    Class representing a generic Solow growth model.

    Attributes
    ----------
    equilibrium_capital : function
        Equilibrium value for capital (per unit effective labor).
    equilibrium_consumption : function
        Equilibrium value for consumption (per unit effective labor).
    intensive_output : function
        Output (per unit effective labor supply).
    marginal_product_capital : function
        Marginal product of capital (i.e., first derivative of intensive output).
    params : dict(str: float)
        Dictionary of model parameters.
    pratt_arrow_risk_aversion : function
        Pratt-Arrow relative risk aversion function.

    """

    def __init__(self, ARA, f, k_star, mpk, params):
        """
        Initialize an instance of the RamseyCassKoopmans class.

        Parameters
        ----------
        ARA : function
            Pratt-Arrow absolute risk aversion function.
        f : function
            Output (per unit effective labor supply).
        k_star : function
            Equilibrium (i.e., steady-state) value for capital stock (per unit
            effective labor supply).
        mpk : function
            Marginal product of capital (per unit effective labor supply).
        params : dict(str: float)
            Dictionary of model parameters

        """
        self._equilibrium_capital = k_star
        self._intensive_output = f
        self._marginal_product_capital = mpk
        self._pratt_arrow_risk_aversion = ARA

        # construct the terminal condition
        c_star = self._c_star_factory(k_star)
        terminal_condition = self._terminal_condition_factory(c_star)
        self._equilibrium_consumption = c_star

        # construct the RHS of the system of ODEs
        rhs = self._rhs_factory(ARA, f, mpk)

        super(RamseyCassKoopmansModel, self).__init__(self._initial_condition,
                                                      terminal_condition, 1, 2,
                                                      params, rhs)

    @property
    def equilibrium_capital(self):
        return self._equilibrium_capital

    @property
    def equilibrium_consumption(self):
        return self._equilibrium_consumption

    @property
    def intensive_output(self):
        return self._intensive_output

    @property
    def marginal_product_capital(self):
        return self._marginal_product_capital

    @property
    def pratt_arrow_risk_aversion(self):
        return self._pratt_arrow_risk_aversion

    @staticmethod
    def _actual_investment(k_tilde, c_tilde, f, **params):
        return f(k_tilde, **params) - c_tilde

    @staticmethod
    def _breakeven_investment(k_tilde, delta, g, n, **params):
        return (g + n + delta) * k_tilde

    @classmethod
    def _c_tilde_dot(cls, t, k_tilde, c_tilde, ARA, mpk, A0, delta, g, rho, **params):
        A = cls._technology(t, A0, g)
        return ((mpk(k_tilde, **params) - delta - rho) / (A * ARA(t, A * c_tilde, **params))) - g * c_tilde

    @staticmethod
    def _initial_condition(t, k_tilde, c_tilde, A0, K0, N0, **params):
        return [k_tilde - (K0 / (A0 * N0))]

    @staticmethod
    def _technology(t, A0, g):
        return A0 * np.exp(g * t)

    @classmethod
    def _k_dot(cls, t, k_tilde, c_tilde, f, delta, g, n, **params):
        k_dot = (cls._actual_investment(k_tilde, c_tilde, f, **params) -
                 cls._breakeven_investment(k_tilde, delta, g, n))
        return k_dot

    @classmethod
    def _ramsey_model(cls, t, k_tilde, c_tilde, ARA, f, mpk, A0, delta, g, n, rho, **params):
        out = [cls._k_dot(t, k_tilde, c_tilde, f, delta, g, n, **params),
               cls._c_tilde_dot(t, k_tilde, c_tilde, ARA, mpk, A0, delta, g, rho, **params)]
        return out

    @classmethod
    def _rhs_factory(cls, ARA, f, mpk):
        return functools.partial(cls._ramsey_model, ARA=ARA, f=f, mpk=mpk)

    @staticmethod
    def _terminal_condition(t, k_tilde, c_tilde, c_star, **params):
        return [c_tilde - c_star(**params)]

    @classmethod
    def _terminal_condition_factory(cls, c_star):
        return functools.partial(cls._terminal_condition, c_star=c_star)

    def _c_star(self, k_star, **params):
        k_tilde = k_star(**params)
        c_star = (self.intensive_output(k_tilde, **params) -
                  self._breakeven_investment(k_tilde, **params))
        return c_star

    def _c_star_factory(self, k_star):
        return functools.partial(self._c_star, k_star=k_star)

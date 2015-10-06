import functools


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

    def __init__(self, A, f, k_star, mpk, params):
        """
        Initialize an instance of the RamseyCassKoopmans class.

        Parameters
        ----------
        A : function
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
        self._pratt_arrow_risk_aversion = A

        # construct the terminal condition
        c_star = self._c_star_factory(k_star)
        terminal_condition = self._terminal_condition_factory(c_star)
        self._equilibrium_consumption = c_star

        # construct the RHS of the system of ODEs
        rhs = self._rhs_factory(A, f, mpk)

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
    def _actual_investment(k, c, f, **params):
        return f(k, **params) - c

    @staticmethod
    def _breakeven_investment(k, delta, g, n, **params):
        return (g + n + delta) * k

    @staticmethod
    def _c_dot(t, k, c, A, mpk, delta, g, rho, **params):
        return (((mpk(k, **params) - delta - rho) / A(t, c, **params)) - g) * c

    @staticmethod
    def _initial_condition(t, k, c, k0, **params):
        return [k - k0]

    @classmethod
    def _k_dot(cls, t, k, c, f, delta, g, n, **params):
        k_dot = (cls._actual_investment(k, c, f, **params) -
                 cls._breakeven_investment(k, delta, g, n))
        return k_dot

    @classmethod
    def _ramsey_model(cls, t, k, c, A, f, mpk, delta, g, n, rho, **params):
        out = [cls._k_dot(t, k, c, f, delta, g, n, **params),
               cls._c_dot(t, k, c, A, mpk, delta, g, rho, **params)]
        return out

    @classmethod
    def _rhs_factory(cls, A, f, mpk):
        return functools.partial(cls._ramsey_model, A=A, f=f, mpk=mpk)

    @staticmethod
    def _terminal_condition(t, k, c, c_star, **params):
        return [c - c_star(**params)]

    @classmethod
    def _terminal_condition_factory(cls, c_star):
        return functools.partial(cls._terminal_condition, c_star=c_star)

    def _c_star(self, k_star, **params):
        k = k_star(**params)
        c_star = (self.intensive_output(k, **params) -
                  self._breakeven_investment(k, **params))
        return c_star

    def _c_star_factory(self, k_star):
        return functools.partial(self._c_star, k_star=k_star)

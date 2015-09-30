import unittest

import numpy as np
from scipy import stats

from .. problems import bvp
from .. import solvers


class RamseyModel(unittest.TestCase):
    """
    Test case using the Ramsey-Cass-Koopmans model of optimal savings.

    Model assumes Cobb-Douglas production technology and Constant Relative Risk
    Aversion (CRRA) preferences.  I then generate random parameter values
    consistent with a constant consumption-output ratio (i.e., constant savings
    rate).

    """

    @staticmethod
    def bcs_lower(t, k, c, k0, **params):
        return [k - k0]

    @staticmethod
    def cobb_douglas_output(k, alpha, **params):
        """Cobb-Douglas output (per unit effective labor supply)."""
        return k**alpha

    @staticmethod
    def equilibrium_capital(g, n, alpha, delta, rho, theta, **params):
        """Steady state value for capital stock (per unit effective labor)."""
        return (alpha / (delta + rho + theta * g))**(1 / (1 - alpha))

    @classmethod
    def equilibrium_consumption(cls, g, n, alpha, delta, rho, theta, **params):
        """Steady state value for consumption (per unit effective labor)."""
        kstar = cls.equilibrium_capital(g, n, alpha, delta, rho, theta)
        return cls.cobb_douglas_output(kstar, alpha) - (g + n + delta) * kstar

    @staticmethod
    def pratt_arrow_risk_aversion(c, theta, **params):
        """Pratt-Arrow absolute risk aversion for CRRA utility."""
        return theta / c

    @staticmethod
    def cobb_douglas_mpk(k, alpha, **params):
        """Marginal product of capital with Cobb-Douglas production."""
        return alpha * k**(alpha - 1)

    @classmethod
    def random_params(cls, sigma):
        # random g, n, delta such that sum of these params is positive
        g, n = stats.norm.rvs(0.05, sigma, size=2)
        delta, = stats.lognorm.rvs(sigma, loc=g + n, size=1)
        assert g + n + delta > 0

        # choose alpha consistent with theta > 1
        alpha, = stats.uniform.rvs(scale=(g + delta) / (g + n + delta), size=1)
        assert alpha < (delta + g) / (g + n + delta)

        lower_bound = delta / (alpha * (g + n + delta) - g)
        theta, = stats.lognorm.rvs(sigma, loc=lower_bound, size=1)
        rho = alpha * theta * (g + n + delta) - (delta * theta * g)
        print theta, lower_bound
        assert rho > 0

        # choose k0 so that it is not too far from equilibrium
        kstar = cls.equilibrium_capital(g, n, alpha, delta, rho, theta)
        k0, = stats.uniform.rvs(0.5 * kstar, 1.5 * kstar, size=1)
        assert k0 > 0

        params = {'g': g, 'n': n, 'delta': delta, 'rho': rho, 'alpha': alpha,
                  'theta': theta, 'k0': k0}

        return params

    @classmethod
    def analytic_solution(cls, t, k0, g, n, alpha, delta, theta, **params):
        """Analytic solution for model with Cobb-Douglas production."""
        lmbda = (g + n + delta) * (1 - alpha)
        ks = ((k0**(1 - alpha) * np.exp(-lmbda * t) +
              (1 / (theta * (g + n + delta))) * (1 - np.exp(-lmbda * t)))**(1 / (1 - alpha)))
        ys = cls.cobb_douglas_output(ks, alpha, **params)
        cs = ((theta - 1) / theta) * ys
        return [ks, cs]

    @classmethod
    def bcs_upper(cls, t, k, c, **params):
        return [c - cls.equilibrium_consumption(**params)]

    @classmethod
    def create_mesh(cls, basis_kwargs, num, problem):
        # compute equilibrium values
        cstar = cls.equilibrium_consumption(**problem.params)
        kstar = cls.equilibrium_capital(**problem.params)
        ystar = cls.cobb_douglas_output(kstar, **problem.params)

        # create the mesh for capital
        ts = np.linspace(*basis_kwargs['domain'], num=num)
        ks = kstar - (kstar - problem.params['k0']) * np.exp(-ts)

        # create the mesh for consumption
        s = 1 - (cstar / ystar)
        ys = cls.cobb_douglas_output(ks, **problem.params)
        cs = (1 - s) * ys

        return ts, ks, cs

    @classmethod
    def fit_initial_polys(cls, basis_kwargs, num, problem):
        ts, ks, cs = cls.create_mesh(basis_kwargs, num, problem)
        basis_poly = getattr(np.polynomial, basis_kwargs['kind'])
        capital_poly = basis_poly.fit(ts, ks, basis_kwargs['degree'],
                                      basis_kwargs['domain'])
        consumption_poly = basis_poly.fit(ts, cs, basis_kwargs['degree'],
                                          basis_kwargs['domain'])
        return capital_poly, consumption_poly

    @classmethod
    def _c_dot(cls, t, k, c, g, delta, rho, theta, **params):
        out = ((cls.cobb_douglas_mpk(k, **params) - delta - rho - theta * g) /
               cls.pratt_arrow_risk_aversion(c, theta, **params))
        return out

    @classmethod
    def _k_dot(cls, t, k, c, g, n, delta, **params):
        return cls.cobb_douglas_output(k, **params) - c - (g + n + delta) * k

    @classmethod
    def rhs(cls, t, k, c, delta, g, n, rho, theta, **params):
        """Equation of motion for capital (per unit effective labor supply)."""
        out = [cls._k_dot(t, k, c, g, n, delta, **params),
               cls._c_dot(t, k, c, g, delta, rho, theta, **params)]
        return out

    def setUp(self):
        """Set up a Solow model to solve."""
        self.bvp = bvp.TwoPointBVP(self.bcs_lower, self.bcs_upper, 1, 2,
                                   self.random_params(0.1), self.rhs)

    def _test_polynomial_collocation(self, basis_kwargs):
        """Test collocation solver using Chebyshev polynomials for basis."""
        nodes = solvers.PolynomialSolver.collocation_nodes(**basis_kwargs)
        initial_polys = self.fit_initial_polys(basis_kwargs, 1000, self.bvp)
        capital_poly, consumption_poly = initial_polys
        initial_coefs = np.hstack([capital_poly.coef, consumption_poly.coef])

        solution = solvers.PolynomialSolver.solve(basis_kwargs,
                                                  initial_coefs,
                                                  nodes,
                                                  self.bvp)

        # check that solver terminated successfully
        self.assertTrue(solution.result.success, msg="Solver failed!")

        # compute the residuals
        ts, _, _ = self.create_mesh(basis_kwargs, 1000, self.bvp)
        normed_residuals = solution.normalize_residuals(ts)

        # check that residuals are close to zero on average
        mesg = "Normed residuals:\n{}\n\nDictionary of model params: {}"
        self.assertTrue(np.mean(normed_residuals) < 1e-6,
                        msg=mesg.format(normed_residuals, self.bvp.params))

        # check that the numerical and analytic solutions are close
        numeric_soln = solution.evaluate_solution(ts)
        analytic_soln = self.analytic_solution(ts, **self.bvp.params)
        mesg = "Error:\n{}"
        self.assertTrue(np.mean(numeric_soln[0] - analytic_soln[0]) < 1e-6,
                        msg=mesg.format(numeric_soln[0] - analytic_soln[0]))
        self.assertTrue(np.mean(numeric_soln[1] - analytic_soln[1]) < 1e-6,
                        msg=mesg.format(numeric_soln[1] - analytic_soln[1]))

    def test_chebyshev_collocation(self):
        """Test collocation solver using Chebyshev polynomials for basis."""
        basis_kwargs = {'kind': 'Chebyshev', 'degree': 50, 'domain': (0, 100)}
        self._test_polynomial_collocation(basis_kwargs)

    def test_legendre_collocation(self):
        """Test collocation solver using Legendre polynomials for basis."""
        basis_kwargs = {'kind': 'Legendre', 'degree': 50, 'domain': (0, 100)}
        self._test_polynomial_collocation(basis_kwargs)

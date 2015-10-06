import unittest

import numpy as np
from scipy import stats

from .. import basis_functions
from .. problems import ivp
from .. import solvers


class SolowModel(unittest.TestCase):

    @staticmethod
    def analytic_solution(t, k0, g, n, s, alpha, delta, **params):
        """Analytic solution for model with Cobb-Douglas production."""
        lmbda = (g + n + delta) * (1 - alpha)
        ks = (((s / (g + n + delta)) * (1 - np.exp(-lmbda * t)) +
               k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))
        return ks

    @staticmethod
    def bcs_lower(t, k, k0, **params):
        return [k - k0]

    @staticmethod
    def cobb_douglas_output(k, alpha, **params):
        """Cobb-Douglas output (per unit effective labor supply)."""
        return k**alpha

    @staticmethod
    def equilibrium_capital(g, n, s, alpha, delta, **params):
        """Steady state value for capital (per unit effective labor supply)."""
        return (s / (g + n + delta))**(1 / (1 - alpha))

    @classmethod
    def random_params(cls, sigma):
        # random g, n, delta such that sum of these params is positive
        g, n = stats.norm.rvs(0.05, sigma, size=2)
        delta, = stats.lognorm.rvs(sigma, loc=g + n, size=1)
        assert g + n + delta > 0

        # s and alpha can be arbitrary on (0, 1)
        s, alpha = stats.beta.rvs(a=1, b=3, size=2)

        # choose k0 so that it is not too far from equilibrium
        kstar = cls.equilibrium_capital(g, n, s, alpha, delta)
        k0, = stats.uniform.rvs(0.5 * kstar, 1.5 * kstar, size=1)
        assert k0 > 0

        params = {'g': g, 'n': n, 'delta': delta, 's': s, 'alpha': alpha,
                  'k0': k0}

        return params

    @classmethod
    def create_mesh(cls, basis_kwargs, num, problem):
        ts = np.linspace(*basis_kwargs['domain'], num=num)
        kstar = cls.equilibrium_capital(**problem.params)
        ks = kstar - (kstar - problem.params['k0']) * np.exp(-ts)
        return ts, ks

    @classmethod
    def fit_initial_polys(cls, basis_kwargs, num, problem):
        ts, ks = cls.create_mesh(basis_kwargs, num, problem)
        basis_poly = getattr(np.polynomial, basis_kwargs['kind'])
        return [basis_poly.fit(ts, ks, basis_kwargs['degree'], basis_kwargs['domain'])]

    @classmethod
    def rhs(cls, t, k, delta, g, n, s, **params):
        """Equation of motion for capital (per unit effective labor supply)."""
        return [s * cls.cobb_douglas_output(k, **params) - (g + n + delta) * k]

    def setUp(self):
        """Set up a Solow model to solve."""
        self.ivp = ivp.IVP(self.bcs_lower, 1, 1, self.random_params(0.1), self.rhs)

    def _test_polynomial_collocation(self, basis_kwargs, num=1000):
        """Test collocation solver using Chebyshev polynomials for basis."""
        polynomial_basis = basis_functions.PolynomialBasis()
        roots = polynomial_basis.nodes(**basis_kwargs)
        initial_polys = self.fit_initial_polys(basis_kwargs, num, self.ivp)
        initial_coefs = np.hstack([poly.coef for poly in initial_polys])

        solver = solvers.Solver(polynomial_basis)
        solution = solver.solve(basis_kwargs, initial_coefs, roots, self.ivp)

        # check that solver terminated successfully
        self.assertTrue(solution.result.success, msg="Solver failed!")

        # compute the residuals
        ts, _ = self.create_mesh(basis_kwargs, num, self.ivp)
        normed_residuals = solution.normalize_residuals(ts)

        # check that residuals are close to zero on average
        mesg = "Normed residuals:\n{}\n\nDictionary of model params: {}"
        self.assertTrue(np.mean(normed_residuals) < 1e-6,
                        msg=mesg.format(normed_residuals, self.ivp.params))

        # check that the numerical and analytic solutions are close
        numeric_soln = solution.evaluate_solution(ts)
        analytic_soln = self.analytic_solution(ts, **self.ivp.params)
        self.assertTrue(np.mean(numeric_soln - analytic_soln) < 1e-6)

    def test_chebyshev_collocation(self):
        """Test collocation solver using Chebyshev polynomials for basis."""
        basis_kwargs = {'kind': 'Chebyshev', 'degree': 50, 'domain': (0, 100)}
        self._test_polynomial_collocation(basis_kwargs)

    def test_legendre_collocation(self):
        """Test collocation solver using Legendre polynomials for basis."""
        basis_kwargs = {'kind': 'Legendre', 'degree': 50, 'domain': (0, 100)}
        self._test_polynomial_collocation(basis_kwargs)

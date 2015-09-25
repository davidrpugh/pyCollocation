import unittest

import numpy as np
from scipy import stats

from .. problems import solow
from .. import solvers


class SolowTestCase(unittest.TestCase):

    @staticmethod
    def cobb_douglas_equilibrium_capital(g, n, s, alpha, delta, **params):
        """Steady state value for capital stock (per unit effective labor)."""
        return (s / (g + n + delta))**(1 / (1 - alpha))

    @staticmethod
    def cobb_douglas_output(k, alpha, **params):
        return k**alpha

    @staticmethod
    def cobb_douglas_params():
        g, n = stats.norm.rvs(size=2)
        delta, = stats.lognorm.rvs(1.0, -(g + n), 1.0, size=1)
        s, alpha = stats.uniform.rvs(size=2)
        k0, = stats.lognorm.rvs(1.0, size=1)
        params = {'g': g, 'n': n, 'delta': delta, 's': s, 'alpha': alpha,
                  'k0': k0}
        return params

    @staticmethod
    def cobb_douglas_solution(t, k0, g, n, s, alpha, delta, **params):
        """Analytic solution for model with Cobb-Douglas production."""
        lmbda = (g + n + delta) * (1 - alpha)
        ks = (((s / (g + n + delta)) * (1 - np.exp(-lmbda * t)) +
               k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))
        return ks

    def setUp(self):
        """Set up a Solow model to solve."""
        self.ivp = solow.IVP(self.cobb_douglas_output,
                             self.cobb_douglas_params(),
                             )

    def _test_polynomial_collocation(self, basis_kwargs):
        """Test collocation solver using Chebyshev polynomials for basis."""
        nodes = solvers.PolynomialSolver.collocation_nodes(**basis_kwargs)
        initial_poly = solow.InitialPoly(self.cobb_douglas_equilibrium_capital)
        initial_coefs = initial_poly.fit(basis_kwargs, 1000, self.ivp).coef

        solution = solvers.PolynomialSolver.solve(basis_kwargs,
                                                  initial_coefs,
                                                  nodes,
                                                  self.ivp)

        # check that solver terminated successfully
        self.assertTrue(solution.result.success, msg="Solver failed!")

        # compute the residuals
        ts, _ = initial_poly.create_mesh(basis_kwargs, 1000, self.ivp)
        normed_residuals = solution.normalize_residuals(ts)

        # check that residuals are close to zero on average
        mesg = "Normed residuals:\n{}\n\nDictionary of model params: {}"
        self.assertTrue(np.mean(normed_residuals) < 1e-6,
                        msg=mesg.format(normed_residuals, self.ivp.params))

        # check that the numerical and analytic solutions are close
        numeric_soln = solution.evaluate_solution(ts)
        analytic_soln = self.cobb_douglas_solution(ts, **self.ivp.params)
        self.assertTrue(np.mean(numeric_soln - analytic_soln) < 1e-6)

    def test_chebyshev_collocation(self):
        """Test collocation solver using Chebyshev polynomials for basis."""
        basis_kwargs = {'kind': 'Chebyshev', 'degree': 50, 'domain': (0, 100)}
        self._test_polynomial_collocation(basis_kwargs)

    def test_legendre_collocation(self):
        """Test collocation solver using Legendre polynomials for basis."""
        basis_kwargs = {'kind': 'Legendre', 'degree': 50, 'domain': (0, 100)}
        self._test_polynomial_collocation(basis_kwargs)

    def test_standard_collocation(self):
        """Test collocation solver using Standard polynomials for basis."""
        basis_kwargs = {'kind': 'Polynomial', 'degree': 50, 'domain': (0, 100)}
        self._test_polynomial_collocation(basis_kwargs)

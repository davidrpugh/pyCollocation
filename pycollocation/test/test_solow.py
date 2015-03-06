import unittest

import numpy as np
import sympy as sym

from .. import models
from .. import orthogonal_collocation


class SolowModel(unittest.TestCase):

    @staticmethod
    def steady_state(g, n, s, alpha, delta, sigma):
        rho = (sigma - 1) / sigma
        return ((1 - alpha) / (((g + n + delta) / s)**rho - alpha))**(1 / rho)

    def setUp(self):
        # define some variables
        t, k, c = sym.symbols('t, k, c')

        # define some parameters
        alpha, sigma = sym.symbols('alpha, sigma')
        rho, theta = sym.symbols('rho, theta')
        g, n, s, delta = sym.symbols('g, n, s, delta')

        # intensive output has the CES form
        rho = (sigma - 1) / sigma
        y = (alpha * k**rho + (1 - alpha))**(1 / rho)

        # define the Solow model
        k_dot = s * y - (g + n + delta) * k
        rhs = {k: k_dot}

        # set some randomly generated parameters
        self.params = {'g': np.random.uniform(),
                       's': np.random.uniform(),
                       'n': np.random.uniform(),
                       'alpha': np.random.uniform(),
                       'sigma': np.random.uniform(0.0, 10.0),
                       'delta': np.random.uniform()}

        # specify some boundary conditions
        kstar = self.steady_state(**self.params)
        if np.random.uniform() < 0.5:
            k0 = 0.5 * kstar
        else:
            k0 = 2.0 * kstar
        bcs = {'lower': [k - k0], 'upper': None}

        # set the model instance
        self.model = models.BoundaryValueProblem(dependent_vars=[k],
                                                 independent_var=t,
                                                 rhs=rhs,
                                                 boundary_conditions=bcs)

        # set the solver instance
        self.solver = orthogonal_collocation.OrthogonalCollocationSolver(self.model,
                                                                         self.params)

        # set an initial guess
        ts = np.linspace(0, 100, 1000)
        ks = kstar - (kstar - k0) * np.exp(-ts)
        initial_guess = np.polynomial.Chebyshev.fit(ts, ks, 50, [0, 100])
        self.initial_coefs = {k: initial_guess.coef}

    def test_orthogonal_polynomial_solver(self):

        solution = self.solver.solve(kind="Chebyshev",
                                     coefs_dict=self.initial_coefs,
                                     domain=[0, 100])

        solution.interpolation_knots = np.linspace(0, 100, 1000)
        mesg = "Residuals:\n{}\n\nDictionary of model params: {}"
        self.assertTrue(np.allclose(solution.residuals.values, 0, atol=1e-6),
                        msg=mesg.format(solution.residuals.values, self.params))

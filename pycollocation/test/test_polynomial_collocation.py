import unittest

import numpy as np
import sympy as sym

from .. import models
from .. import orthogonal_collocation


class SolowModel(unittest.TestCase):

    @staticmethod
    def steady_state(g, n, s, alpha, delta, sigma):
        """Steady state value for capital (per unit effective labor)."""
        rho = (sigma - 1) / sigma
        return ((1 - alpha) / (((g + n + delta) / s)**rho - alpha))**(1 / rho)

    def setUp(self):
        """Set up a Solow model to solve."""
        # define some variables
        t, k, c = sym.symbols('t, k, c')

        # define some parameters
        alpha, sigma = sym.symbols('alpha, sigma')
        rho, theta = sym.symbols('rho, theta')
        g, n, s, delta = sym.symbols('g, n, s, delta')

        # intensive output has the CES form
        rho = (sigma - 1) / sigma
        y = (alpha * k**rho + (1 - alpha))**(1 / rho)

        # define the equation of motion for capital
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

        # set the domain
        self.domain = [0, 100]

        # set an initial guess
        ts = np.linspace(self.domain[0], self.domain[1], 1000)
        ks = kstar - (kstar - k0) * np.exp(-ts)
        initial_guess = np.polynomial.Chebyshev.fit(ts, ks, 50, self.domain)
        self.initial_coefs = {k: initial_guess.coef}

    def test_chebyshev_collocation(self):
        """Test collocation solver using Chebyshev polynomials for basis."""
        solution = self.solver.solve(kind="Chebyshev",
                                     coefs_dict=self.initial_coefs,
                                     domain=self.domain)

        # compute the residuals
        solution.interpolation_knots = np.linspace(self.domain[0],
                                                   self.domain[1],
                                                   1000)
        residuals = solution.residuals.values

        # check that residuals are all close to zero
        mesg = "Residuals:\n{}\n\nDictionary of model params: {}"
        self.assertTrue(np.allclose(residuals, 0, atol=1e-6),
                        msg=mesg.format(residuals, self.params))

#     def test_legendre_collocation(self):
#         """Test collocation solver using Legendre polynomials for basis."""
#         solution = self.solver.solve(kind="Legendre",
#                                      coefs_dict=self.initial_coefs,
#                                      domain=self.domain)

#         # compute the residuals
#         solution.interpolation_knots = np.linspace(self.domain[0],
#                                                    self.domain[1],
#                                                    1000)
#         residuals = solution.residuals.values

#         # check that residuals are all close to zero
#         mesg = "Residuals:\n{}\n\nDictionary of model params: {}"
#         self.assertTrue(np.allclose(residuals, 0, atol=1e-6),
#                         msg=mesg.format(residuals, self.params))


# class RamseyModel(unittest.TestCase):

#     @staticmethod
#     def steady_state(g, n, s, alpha, delta, rho, sigma, theta):
#         """Steady state value for capital (per unit effective labor)."""
#         gamma = (sigma - 1) / sigma
#         kstar = ((1 - alpha)**(1 / gamma) * ((alpha / (delta + rho + theta * g))**(gamma / (gamma - 1)) - alpha)**(-1 / gamma)
#         cstar = 

#     def setUp(self):
#         """Set up a Solow model to solve."""
#         # define some variables
#         t, k, c = sym.symbols('t, k, c')

#         # define some parameters
#         alpha, sigma = sym.symbols('alpha, sigma')
#         rho, theta = sym.symbols('rho, theta')
#         g, n, delta = sym.symbols('g, n, delta')

#         # define functional form for intensive output
#         gamma = (sigma - 1) / sigma
#         f = (alpha * k**gamma + (1 - alpha))**(1 / gamma)

#         # define functional form for utility
#         u = c**(1 - theta) / (1 - theta)

#         # equation of motion for capital (per unit effective labor)
#         k_dot = f - c - (g + n + delta) * k

#         # equation of motion for consumption (per unit effective labor)
#         mpk = sym.diff(f, k, 1)
#         epsilon_mu = -(sym.diff(u, c, 2) * c) / sym.diff(u, c)
#         c_dot = ((1 / epsilon_mu) * (mpk - delta - rho) - g) * c
#         rhs = {k: k_dot, c: c_dot}


#         # set some randomly generated parameters
#         self.params = {'g': np.random.uniform(),
#                        's': np.random.uniform(),
#                        'n': np.random.uniform(),
#                        'alpha': np.random.uniform(),
#                        'sigma': np.random.uniform(0.0, 10.0),
#                        'delta': np.random.uniform()}

#         # specify some boundary conditions
#         kstar = self.steady_state(**self.params)
#         if np.random.uniform() < 0.5:
#             k0 = 0.5 * kstar
#         else:
#             k0 = 2.0 * kstar
#         bcs = {'lower': [k - k0], 'upper': None}

#         # set the model instance
#         self.model = models.BoundaryValueProblem(dependent_vars=[k],
#                                                  independent_var=t,
#                                                  rhs=rhs,
#                                                  boundary_conditions=bcs)

#         # set the solver instance
#         self.solver = orthogonal_collocation.OrthogonalCollocationSolver(self.model,
#                                                                          self.params)

#         # set the domain
#         self.domain = [0, 100]

#         # set an initial guess
#         ts = np.linspace(self.domain[0], self.domain[1], 1000)
#         ks = kstar - (kstar - k0) * np.exp(-ts)
#         initial_guess = np.polynomial.Chebyshev.fit(ts, ks, 50, self.domain)
#         self.initial_coefs = {k: initial_guess.coef}

#     def test_chebyshev_collocation(self):
#         """Test collocation solver using Chebyshev polynomials for basis."""
#         solution = self.solver.solve(kind="Chebyshev",
#                                      coefs_dict=self.initial_coefs,
#                                      domain=self.domain)

#         # compute the residuals
#         solution.interpolation_knots = np.linspace(self.domain[0],
#                                                    self.domain[1],
#                                                    1000)
#         residuals = solution.residuals.values

#         # check that residuals are all close to zero
#         mesg = "Residuals:\n{}\n\nDictionary of model params: {}"
#         self.assertTrue(np.allclose(residuals, 0, atol=1e-6),
#                         msg=mesg.format(residuals, self.params))


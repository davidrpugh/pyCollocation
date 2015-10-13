from nose import tools

import numpy as np
from scipy import stats

from . import models
from .. import basis_functions
from .. import solvers


def analytic_solution(t, k0, alpha, delta, g, n, s, **params):
    """Analytic solution for model with Cobb-Douglas production."""
    lmbda = (g + n + delta) * (1 - alpha)
    ks = (((s / (g + n + delta)) * (1 - np.exp(-lmbda * t)) +
           k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))
    return ks


def cobb_douglas_output(k, alpha, **params):
    """Intensive output has Cobb-Douglas functional form."""
    return k**alpha


def equilibrium_capital(alpha, delta, g, n, s, **params):
    """Equilibrium value of capital (per unit effective labor supply)."""
    return (s / (g + n + delta))**(1 / (1 - alpha))


def generate_random_params(scale, seed):
    np.random.seed(seed)

    # random g, n, delta such that sum of these params is positive
    g, n = stats.norm.rvs(0.05, scale, size=2)
    delta, = stats.lognorm.rvs(scale, loc=g + n, size=1)
    assert g + n + delta > 0

    # s and alpha must be on (0, 1) (but lower values are more reasonable)
    s, alpha = stats.beta.rvs(a=1, b=3, size=2)

    # choose k0 so that it is not too far from equilibrium
    kstar = equilibrium_capital(alpha, delta, g, n, s)
    k0, = stats.uniform.rvs(0.5 * kstar, 1.5 * kstar, size=1)
    assert k0 > 0

    params = {'g': g, 'n': n, 'delta': delta, 's': s, 'alpha': alpha,
              'k0': k0}

    return params


def initial_mesh(t, T, num, problem):
    ts = np.linspace(t, T, num)
    kstar = equilibrium_capital(**problem.params)
    ks = kstar - (kstar - problem.params['k0']) * np.exp(-ts)
    return ts, ks


random_seed = np.random.randint(2147483647)
random_params = generate_random_params(0.1, random_seed)
test_problem = models.SolowModel(cobb_douglas_output, equilibrium_capital,
                                 random_params)

polynomial_basis = basis_functions.PolynomialBasis()
solver = solvers.Solver(polynomial_basis)


def _test_polynomial_collocation(basis_kwargs, boundary_points, num=1000):
    """Helper function for testing various kinds of polynomial collocation."""
    ts, ks = initial_mesh(*boundary_points, num=num, problem=test_problem)
    k_poly = polynomial_basis.fit(ts, ks, **basis_kwargs)
    initial_coefs = k_poly.coef
    nodes = polynomial_basis.roots(**basis_kwargs)

    solution = solver.solve(basis_kwargs, boundary_points, initial_coefs,
                            nodes, test_problem)

    # check that solver terminated successfully
    msg = "Solver failed!\nSeed: {}\nModel params: {}\n"
    tools.assert_true(solution.result.success,
                      msg=msg.format(random_seed, test_problem.params))

    # compute the residuals
    normed_residuals = solution.normalize_residuals(ts)

    # check that residuals are close to zero on average
    tools.assert_true(np.mean(normed_residuals) < 1e-6,
                      msg=msg.format(random_seed, test_problem.params))

    # check that the numerical and analytic solutions are close
    numeric_soln = solution.evaluate_solution(ts)
    analytic_soln = analytic_solution(ts, **test_problem.params)
    tools.assert_true(np.mean(numeric_soln - analytic_soln) < 1e-6)


def test_chebyshev_collocation():
    """Test collocation solver using Chebyshev polynomials for basis."""
    boundary_points = (0, 100)
    basis_kwargs = {'kind': 'Chebyshev', 'degree': 50, 'domain': boundary_points}
    _test_polynomial_collocation(basis_kwargs, boundary_points)


def test_legendre_collocation():
    """Test collocation solver using Legendre polynomials for basis."""
    boundary_points = (0, 100)
    basis_kwargs = {'kind': 'Legendre', 'degree': 50, 'domain': boundary_points}
    _test_polynomial_collocation(basis_kwargs, boundary_points)

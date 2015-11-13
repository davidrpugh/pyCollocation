"""
Test case using the Ramsey-Cass-Koopmans model of optimal savings.

Model assumes Cobb-Douglas production technology and Constant Relative Risk
Aversion (CRRA) preferences.  I then generate random parameter values
consistent with a constant consumption-output ratio (i.e., constant savings
rate).

"""
from nose import tools

import numpy as np
from scipy import stats

from . import models
from .. import basis_functions
from .. import solvers


def analytic_solution(t, A0, alpha, delta, g, K0, n, N0, theta, **params):
    """Analytic solution for model with Cobb-Douglas production."""
    k0 = K0 / (A0 * N0)
    lmbda = (g + n + delta) * (1 - alpha)
    ks = ((k0**(1 - alpha) * np.exp(-lmbda * t) +
          (1 / (theta * (g + n + delta))) * (1 - np.exp(-lmbda * t)))**(1 / (1 - alpha)))
    ys = cobb_douglas_output(ks, alpha, **params)
    cs = ((theta - 1) / theta) * ys

    return [ks, cs]


def cobb_douglas_output(k, alpha, **params):
    """Cobb-Douglas output (per unit effective labor supply)."""
    return k**alpha


def cobb_douglas_mpk(k, alpha, **params):
    """Marginal product of capital with Cobb-Douglas production."""
    return alpha * k**(alpha - 1)


def equilibrium_capital(alpha, delta, g, n, rho, theta, **params):
    """Steady state value for capital stock (per unit effective labor)."""
    return (alpha / (delta + rho + theta * g))**(1 / (1 - alpha))


def generate_random_params(scale, seed):
    np.random.seed(seed)

    # random g, n, delta such that sum of these params is positive
    g, n = stats.norm.rvs(0.05, scale, size=2)
    delta, = stats.lognorm.rvs(scale, loc=g + n, size=1)
    assert g + n + delta > 0

    # choose alpha consistent with theta > 1
    upper_bound = 1 if (g + delta) < 0 else min(1, (g + delta) / (g + n + delta))
    alpha, = stats.uniform.rvs(scale=upper_bound, size=1)
    assert 0 < alpha < upper_bound

    lower_bound = delta / (alpha * (g + n + delta) - g)
    theta, = stats.lognorm.rvs(scale, loc=lower_bound, size=1)
    rho = alpha * theta * (g + n + delta) - (delta * theta * g)
    assert rho > 0

    # normalize A0 and N0
    A0 = 1.0
    N0 = A0

    # choose K0 so that it is not too far from balanced growth path
    kstar = equilibrium_capital(g, n, alpha, delta, rho, theta)
    K0, = stats.uniform.rvs(0.5 * kstar, 1.5 * kstar, size=1)
    assert K0 > 0

    params = {'g': g, 'n': n, 'delta': delta, 'rho': rho, 'alpha': alpha,
              'theta': theta, 'K0': K0, 'A0': A0, 'N0': N0}

    return params


def initial_mesh(t, T, num, problem):
    # compute equilibrium values
    cstar = problem.equilibrium_consumption(**problem.params)
    kstar = problem.equilibrium_capital(**problem.params)
    ystar = problem.intensive_output(kstar, **problem.params)

    # create the mesh for capital
    ts = np.linspace(t, T, num)
    k0 = problem.params['K0'] / (problem.params['A0'] * problem.params['N0'])
    ks = kstar - (kstar - k0) * np.exp(-ts)

    # create the mesh for consumption
    s = 1 - (cstar / ystar)
    y0 = cobb_douglas_output(k0, **problem.params)
    c0 = (1 - s) * y0
    cs = cstar - (cstar - c0) * np.exp(-ts)

    return ts, ks, cs


def pratt_arrow_risk_aversion(t, c, theta, **params):
    """Assume constant relative risk aversion"""
    return theta / c


random_seed = np.random.randint(2147483647)
random_params = generate_random_params(0.1, random_seed)
test_problem = models.RamseyCassKoopmansModel(pratt_arrow_risk_aversion,
                                              cobb_douglas_output,
                                              equilibrium_capital,
                                              cobb_douglas_mpk,
                                              random_params)

polynomial_basis = basis_functions.PolynomialBasis()
solver = solvers.Solver(polynomial_basis)


def _test_polynomial_collocation(basis_kwargs, boundary_points, num=1000):
    """Helper function for testing various kinds of polynomial collocation."""
    ts, ks, cs = initial_mesh(*boundary_points, num=num, problem=test_problem)
    k_poly = polynomial_basis.fit(ts, ks, **basis_kwargs)
    c_poly = polynomial_basis.fit(ts, cs, **basis_kwargs)
    initial_coefs = np.hstack([k_poly.coef, c_poly.coef])
    nodes = polynomial_basis.roots(**basis_kwargs)

    solution = solver.solve(basis_kwargs, boundary_points, initial_coefs,
                            nodes, test_problem)

    # check that solver terminated successfully
    msg = "Solver failed!\nSeed: {}\nModel params: {}\n".format(random_seed, test_problem.params)
    tools.assert_true(solution.result.success, msg=msg)

    # compute the residuals
    normed_residuals = solution.normalize_residuals(ts)

    # check that residuals are close to zero on average
    for normed_residual in normed_residuals:
        tools.assert_true(np.mean(normed_residual) < 1e-6, msg=msg)

    # check that the numerical and analytic solutions are close
    numeric_solns = solution.evaluate_solution(ts)
    analytic_solns = analytic_solution(ts, **test_problem.params)

    for numeric_soln, analytic_soln in zip(numeric_solns, analytic_solns):
        error = numeric_soln - analytic_soln
        tools.assert_true(np.mean(error) < 1e-6, msg=msg)


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

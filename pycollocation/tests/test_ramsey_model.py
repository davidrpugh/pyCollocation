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


def analytic_solution(t, k0, g, n, alpha, delta, theta, **params):
    """Analytic solution for model with Cobb-Douglas production."""
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


def equilibrium_capital(g, n, alpha, delta, rho, theta, **params):
    """Steady state value for capital stock (per unit effective labor)."""
    return (alpha / (delta + rho + theta * g))**(1 / (1 - alpha))


def generate_random_params(scale, seed):
    np.random.seed(seed)

    # random g, n, delta such that sum of these params is positive
    g, n = stats.norm.rvs(0.05, scale, size=2)
    delta, = stats.lognorm.rvs(scale, loc=g + n, size=1)
    assert g + n + delta > 0

    # choose alpha consistent with theta > 1
    alpha, = stats.uniform.rvs(scale=(g + delta) / (g + n + delta), size=1)
    assert alpha < (delta + g) / (g + n + delta)

    lower_bound = delta / (alpha * (g + n + delta) - g)
    theta, = stats.lognorm.rvs(scale, loc=lower_bound, size=1)
    rho = alpha * theta * (g + n + delta) - (delta * theta * g)
    print theta, lower_bound
    assert rho > 0

    # choose k0 so that it is not too far from equilibrium
    kstar = equilibrium_capital(g, n, alpha, delta, rho, theta)
    k0, = stats.uniform.rvs(0.5 * kstar, 1.5 * kstar, size=1)
    assert k0 > 0

    params = {'g': g, 'n': n, 'delta': delta, 'rho': rho, 'alpha': alpha,
              'theta': theta, 'k0': k0}

    return params


def initial_mesh(domain, num, problem):
    # compute equilibrium values
    cstar = problem.equilibrium_consumption(**problem.params)
    kstar = problem.equilibrium_capital(**problem.params)
    ystar = problem.intensive_output(kstar, **problem.params)

    # create the mesh for capital
    ts = np.linspace(domain[0], domain[1], num)
    ks = kstar - (kstar - problem.params['k0']) * np.exp(-ts)

    # create the mesh for consumption
    s = 1 - (cstar / ystar)
    y0 = cobb_douglas_output(problem.params['k0'], **problem.params)
    c0 = (1 - s) * y0
    cs = cstar - (cstar - c0) * np.exp(-ts)

    return ts, ks, cs


def pratt_arrow_risk_aversion(t, c, theta, **params):
    """Assume constant relative risk aversion"""
    return theta


random_seed = np.random.randint(2147483647)
random_params = generate_random_params(0.1, random_seed)
test_problem = models.RamseyCassKoopmansModel(pratt_arrow_risk_aversion,
                                              cobb_douglas_output,
                                              equilibrium_capital,
                                              cobb_douglas_mpk,
                                              random_params)

polynomial_basis = basis_functions.PolynomialBasis()
solver = solvers.Solver(polynomial_basis)


def _test_polynomial_collocation(basis_kwargs, num=1000):
    """Helper function for testing various kinds of polynomial collocation."""
    ts, ks, cs = initial_mesh(basis_kwargs['domain'], 1000, test_problem)
    k_poly = polynomial_basis.fit(ts, ks, **basis_kwargs)
    c_poly = polynomial_basis.fit(ts, cs, **basis_kwargs)
    initial_coefs = np.hstack([k_poly.coef, c_poly.coef])

    solution = solver.solve(basis_kwargs, initial_coefs, test_problem)

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
    basis_kwargs = {'kind': 'Chebyshev', 'degree': 50, 'domain': (0, 100)}
    _test_polynomial_collocation(basis_kwargs)


def test_legendre_collocation():
    """Test collocation solver using Legendre polynomials for basis."""
    basis_kwargs = {'kind': 'Legendre', 'degree': 50, 'domain': (0, 100)}
    _test_polynomial_collocation(basis_kwargs)

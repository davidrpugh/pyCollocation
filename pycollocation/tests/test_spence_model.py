from nose import tools

import numpy as np
from scipy import stats

from .. import basis_functions
from .. import problems
from .. import solvers


def analytic_solution(y, nL, alpha):
    """
    Analytic solution to the differential equation describing the signaling
    equilbrium of the Spence (1974) model.

    """
    D = ((1 + alpha) / 2) * (nL / yL(nL, alpha)**-alpha)**2 - yL(nL, alpha)**(1 + alpha)
    return y**(-alpha) * (2 * (y**(1 + alpha) + D) / (1 + alpha))**0.5


def spence_model(y, n, alpha, **params):
    return [(n**-1 - alpha * n * y**(alpha - 1)) / y**alpha]


def initial_condition(y, n, nL, alpha, **params):
    return [n - nL]


def yL(nL, alpha):
    return (nL**2 * alpha)**(1 / (1 - alpha))


def initial_mesh(yL, yH, num, problem):
    ys = np.linspace(yL, yH, num=num)
    ns = problem.params['nL'] + np.sqrt(ys)
    return ys, ns


random_seed = np.random.randint(2147483647)
params = {'nL': 1.0, 'alpha': 0.15}
test_problem = problems.IVP(initial_condition, 1, 1, params, spence_model)


def test_bspline_collocation():
    """Tests B-spline collocation."""
    bspline_basis = basis_functions.BSplineBasis()
    solver = solvers.Solver(bspline_basis)

    boundary_points = (yL(**params), 10)
    ys, ns = initial_mesh(*boundary_points, num=250, problem=test_problem)

    tck, u = bspline_basis.fit([ns], u=ys, k=5, s=0)
    knots, coefs, k = tck
    initial_coefs = np.hstack(coefs)

    basis_kwargs = {'knots': knots, 'degree': k, 'ext': 2}
    nodes = np.linspace(*boundary_points, num=249)
    solution = solver.solve(basis_kwargs, boundary_points, initial_coefs,
                            nodes, test_problem)

    # check that solver terminated successfully
    msg = "Solver failed!\nSeed: {}\nModel params: {}\n"
    tools.assert_true(solution.result.success,
                      msg=msg.format(random_seed, test_problem.params))

    # compute the residuals
    normed_residuals = solution.normalize_residuals(ys)

    # check that residuals are close to zero on average
    tools.assert_true(np.mean(normed_residuals) < 1e-6,
                      msg=msg.format(random_seed, test_problem.params))

    # check that the numerical and analytic solutions are close
    numeric_soln = solution.evaluate_solution(ys)
    analytic_soln = analytic_solution(ys, **test_problem.params)
    tools.assert_true(np.mean(numeric_soln - analytic_soln) < 1e-6)

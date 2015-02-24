import numpy as np
import pandas as pd
from scipy import optimize

import solvers


class OrthogonalCollocation(solvers.Solver):
    """Base class for OrthogonalCollocation classes."""

    _valid_kinds = ["Chebyshev", "Hermite", "Laguerre", "Legendre"]

    @classmethod
    def _basis_derivative_factory(cls, coef, kind, domain):
        """Return an orthogonal polynomial given some coefficients."""
        if kind == "Chebyshev":
            poly = np.polynomial.Chebyshev(coef, domain).deriv()
        elif kind == "Hermite":
            poly = np.polynomial.Hermite(coef, domain).deriv()
        elif kind == "Laguerre":
            poly = np.polynomial.Laguerre(coef, domain).deriv()
        elif kind == "Legendre":
            poly = np.polynomial.Legendre(coef, domain).deriv()
        else:
            mesg = "Attribute 'kind' must be one of {}, {}, {}, or {}."
            raise AttributeError(mesg.format(*cls._valid_kinds))
        return poly

    @classmethod
    def _basis_function_factory(cls, coef, kind, domain):
        """Return an orthogonal polynomial given some coefficients."""
        if kind == "Chebyshev":
            poly = np.polynomial.Chebyshev(coef, domain)
        elif kind == "Hermite":
            poly = np.polynomial.Hermite(coef, domain)
        elif kind == "Laguerre":
            poly = np.polynomial.Laguerre(coef, domain)
        elif kind == "Legendre":
            poly = np.polynomial.Legendre(coef, domain)
        else:
            mesg = "Attribute 'kind' must be one of {}, {}, {}, or {}."
            raise AttributeError(mesg.format(*cls._valid_kinds))
        return poly


class OrthogonalCollocationSolver(OrthogonalCollocation):

    @staticmethod
    def _basis_polynomial_coefs(degrees):
        """Return coefficients for the basis polynomial of a given degree."""
        basis_coefs = {}
        for var, degree in degrees.iteritems():
            tmp_coef = np.zeros(degree + 1)
            tmp_coef[-1] = 1
            basis_coefs[var] = tmp_coef
        return basis_coefs

    @staticmethod
    def _collocation_nodes(polynomials):
        """Return roots of suitable basis polynomial as collocation nodes."""
        return {var: poly.roots() for var, poly in polynomials.iteritems()}

    def _evaluate_collocation_residuals(self, coefs_array, kind, domain, degrees):
        """Return residuals given coefs and degrees for approximating polys."""
        # construct residual functions given new array of coefficients
        coefs = self._coefs_array_to_dict(coefs_array, degrees)
        funcs = self._construct_basis_funcs(coefs, kind, domain)
        derivs = self._construct_basis_derivs(coefs, kind, domain)
        residual_funcs = self._construct_residual_funcs(derivs, funcs)

        # find the appropriate collocation nodes
        basis_coefs = self._basis_polynomial_coefs(degrees)
        basis_polys = self._construct_basis_funcs(basis_coefs, kind, domain)
        nodes = self._collocation_nodes(basis_polys)

        # evaluate residual functions at the collocation nodes
        residuals = self._evaluate_residual_funcs(residual_funcs, nodes)
        boundary_residuals = self._evaluate_boundary_residuals(funcs, domain)
        collocation_residuals = residuals + boundary_residuals

        return np.hstack(collocation_residuals)

    def _infer_degrees(self, coefs_dict):
        """Return dict mapping a symbol to degree of its approximating poly."""
        degrees = {}
        for var in self.model.dependent_vars:
            coef_array = coefs_dict[var]
            degrees[var] = coef_array.size - 1
        return degrees

    def solve(self, kind, coefs_dict, domain, method="hybr", **kwargs):
        """Solve a boundary value problem using orthogonal collocation."""
        # solve for the optimal coefficients
        degrees = self._infer_degrees(coefs_dict)
        initial_guess = self._coefs_dict_to_array(coefs_dict)
        result = optimize.root(self._evaluate_collocation_residuals,
                               x0=initial_guess,
                               args=(kind, domain, degrees),
                               method=method,
                               **kwargs)
        solution = OrthogonalCollocationSolution(degrees, domain, kind,
                                                 self.model, self.params,
                                                 result)
        return solution


class OrthogonalCollocationSolution(OrthogonalCollocation):
    """Represents solution obtained using an OrthogonalCollocation solver."""

    def __init__(self, degrees, domain, kind, model, params, result):
        """Create an instance of the OrthogonalCollocationSolution class."""
        super(OrthogonalCollocationSolution, self).__init__(model, params)

        # initialize solution attributes
        self._degrees = degrees
        self._domain = domain
        self._kind = kind
        self._result = self._validate_result(result)

    @property
    def _interpolation_knots(self):
        """Interpolation knots to use when computing the final solution."""
        return np.linspace(self.domain[0], self.domain[1], self.number_knots)

    @property
    def coefficients(self):
        return self._coefs_array_to_dict(self.result.x, self.degrees)

    @property
    def degrees(self):
        return self._degrees

    @property
    def domain(self):
        return self._domain

    @property
    def derivatives(self):
        return self._construct_basis_derivs(self.coefficients, self.kind, self.domain)

    @property
    def functions(self):
        return self._construct_basis_funcs(self.coefficients, self.kind, self.domain)

    @property
    def kind(self):
        return self._kind

    @property
    def number_knots(self):
        """
        Number of interpolation knots to use when computing the final solution.

        :getter: Return the number of interpolation knots.
        :setter: Set a new number of interpolation knots.
        :type: int

        """
        return self._number_knots

    @number_knots.setter
    def number_knots(self, value):
        """Set a new number of interpolation knots."""
        self._number_knots = self._validate_number_knots(value)

    @property
    def result(self):
        """
        An instance of the `optimize.optimize.OptimizeResult` class that stores
        the raw output of the `OrthogonalCollocationSolver`.

        :getter: Return the `result` attribute.
        :type: `optimize.optimize.OptimizeResult`

        """
        return self._result

    @property
    def residual_functions(self):
        return self._construct_residual_funcs(self.derivatives, self.functions)

    @property
    def _solution(self):
        """Return the solution stored as a NumPy array."""
        tmp = []#[self._interpolation_knots]
        for var in self.polynomials.iteritems():
            tmp.append(poly(self._interpolation_knots)[:, np.newaxis])
        return np.hstack(tmp)

    @property
    def solution(self):
        """
        Solution to the model represented as a Pandas DataFrame.

        :getter: Return the DataFrame representing the current solution.
        :type: pandas.DataFrame

        """
        # col_names = ['x', r'$\mu(x)$', r'$\theta(x)$', '$w(x)$', r'$\pi(x)$']
        df = pd.DataFrame(self._solution, index=self._interpolation_knots)
        return df.set_index('x')

    @property
    def success(self):
        """
        True if a solution was found by the `OrthogonalCollocationSolver`;
        False otherwise.

        :getter: Return the `success` attribute.
        :type: boolean

        """
        return self.result.success

    @staticmethod
    def _validate_number_knots(number):
        """Validates the `number_knots` attribute."""
        if not isinstance(number, int):
            mesg = ("The 'number_knots' attribute must have type " +
                    "'int', not {}.")
            raise AttributeError(mesg.format(number.__class__))
        else:
            return number

    @staticmethod
    def _validate_result(result):
        """Validates the `result` attribute."""
        if not isinstance(result, optimize.optimize.OptimizeResult):
            mesg = ("The 'result' attribute must have type " +
                    "'optimize.optimize.OptimizeResult', not {}.")
            raise AttributeError(mesg.format(result.__class__))
        else:
            return result

if __name__ == '__main__':
    import sympy as sym
    import models

    # define some variables
    t, k, c = sym.symbols('t, k, c')

    # define some parameters
    alpha, sigma = sym.symbols('alpha, sigma')
    rho, theta = sym.symbols('rho, theta')
    g, n, s, delta = sym.symbols('g, n, s, delta')

    # intensive output has the CES form
    y = (alpha * k**((sigma - 1) / sigma) + (1 - alpha))**(sigma / (sigma - 1))

    # define the Solow model
    k_dot = s * y - (g + n + delta) * k
    rhs = {k: k_dot}

    # specify some boundary conditions
    k0 = 1.0
    bcs = {'lower': [k - k0], 'upper': None}  # hopefully this will work!

    solow = models.BoundaryValueProblem(dependent_vars=[k],
                                        independent_var=t,
                                        rhs=rhs,
                                        boundary_conditions=bcs)

    solow_params = {'g': 0.02, 's': 0.1, 'n': 0.02, 'alpha': 0.15, 'sigma': 2.0,
                    'delta': 0.04}
    solow_solver = OrthogonalCollocationSolver(solow, solow_params)

    # specify an initial guess
    def kstar(g, n, s, alpha, delta, sigma):
        rho = (sigma - 1) / sigma
        return ((1 - alpha) / (((g + n + delta) / s)**rho - alpha))**(1 / rho)
    xs = np.linspace(0, 100, 1000)
    ys = kstar(**solow_params) - (kstar(**solow_params) - k0) * np.exp(-xs)
    initial_poly = np.polynomial.Chebyshev.fit(xs, ys, 15, [0, 100])
    initial_solow_coefs = {k: initial_poly.coef}

    solow_result = solow_solver.solve(kind="Chebyshev",
                                      coefs_dict=initial_solow_coefs,
                                      domain=[0, 100])

    # define the Ramsey-Cass-Coopmans model
    mpk = sym.diff(y, k, 1)
    k_dot = y - c - (g + n) * k
    c_dot = ((mpk - delta - rho - theta * g) / theta) * c
    rhs = {k: k_dot, c: c_dot}

    kstar = ((1 - alpha)**((sigma - 1) / sigma) /
             ((alpha / (delta + rho + theta * g))**(1 - sigma) - (alpha / (1 - alpha)))**((sigma - 1) / sigma))
    cstar = y.subs({k: kstar}) - (g + n) * kstar
    bcs = {'lower': [k - 5.0], 'upper': [c - cstar]}

    ramsey = models.BoundaryValueProblem(dependent_vars=[k, c],
                                         independent_var=t,
                                         rhs=rhs,
                                         boundary_conditions=bcs)

    ramsey_params = {'g': 0.02, 's': 0.1, 'n': 0.02, 'alpha': 0.15, 'sigma': 2.0,
                     'theta': 2.05, 'rho': 0.03, 'delta': 0.04}

    ramsey_solver = OrthogonalCollocationSolver(ramsey, ramsey_params)


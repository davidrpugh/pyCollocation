import numpy as np
from scipy import optimize

from .. import solvers
from .. import solutions


class OrthogonalPolynomialBasis(solvers.Solver):
    """Class for constucting orthogonal polynomial basis functions."""

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


class OrthogonalPolynomialSolver(OrthogonalPolynomialBasis):

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
        solution = OrthogonalPolynomialSolution(degrees, domain, kind,
                                                self.model, self.params, result)
        return solution


class OrthogonalPolynomialSolution(solutions.Solution, OrthogonalPolynomialBasis):
    """Represents solution obtained using an OrthogonalCollocation solver."""

    def __init__(self, degrees, domain, kind, model, params, result):
        """Create an instance of the OrthogonalPolynomialSolution class."""
        super(OrthogonalPolynomialSolution, self).__init__(model, params)

        # initialize solution attributes
        self._degrees = degrees
        self._domain = domain
        self._kind = kind
        self._result = self._validate_result(result)

    @property
    def coefficients(self):
        """
        Coefficients to use when constructing the approximating polynomials.

        :getter: Return the `coefficients` attribute.
        :type: dict

        """
        return self._coefs_array_to_dict(self.result.x, self.degrees)

    @property
    def degrees(self):
        """
        Degrees to use when constructing the approximating polynomials.

        :getter: Return the `degrees` attribute.
        :type: dict

        """
        return self._degrees

    @property
    def derivatives(self):
        """
        Derivatives of the approximating polynomials.

        :getter: Return the `derivatives` attribute.
        :type: dict

        """
        return self._construct_basis_derivs(self.coefficients, self.kind, self.domain)

    @property
    def domain(self):
        """
        Domain over which the approximated solution is valid.

        :getter: Return the `domain` attribute.
        :type: list

        """
        return self._domain

    @property
    def functions(self):
        """
        The polynomial functions used to approximate the solution to the model.

        :getter: Return the `functions` attribute.
        :type: dict

        """
        return self._construct_basis_funcs(self.coefficients, self.kind, self.domain)

    @property
    def kind(self):
        """
        Kind of polynomials to use when constructing the approximation.

        :getter: Return the `kind` of orthogonal polynomials.
        :type: string

        """
        return self._kind

"""
Classes for solving models using collocation with orthogonal polynomials as the
underlying basis functions.

@author: davidrpugh

"""
import numpy as np
from scipy import optimize

from . import solvers


class OrthogonalPolynomialBasis(object):
    """Class for constucting orthogonal polynomial basis functions."""

    _valid_kinds = ["Chebyshev", "Hermite", "Laguerre", "Legendre"]

    @property
    def degrees(self):
        """
        Degrees used when constructing the orthogonal polynomials.

        :getter: Return the `degrees` attribute.
        :type: dict

        """
        return self._degrees

    @property
    def domain(self):
        """
        Domain over which the approximated solution is valid.

        :getter: Return the `domain` attribute.
        :type: list

        """
        return self._domain

    @property
    def kind(self):
        """
        Kind of polynomials to use when constructing the approximation.

        :getter: Return the `kind` of orthogonal polynomials.
        :type: string

        """
        return self._kind

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


class OrthogonalPolynomialSolver(OrthogonalPolynomialBasis, solvers.Solver):

    @staticmethod
    def _basis_polynomial_coefs(degrees):
        """Return coefficients for the basis polynomial of a given degree."""
        basis_coefs = {}
        for var, degree in degrees.items():
            tmp_coef = np.zeros(degree + 1)
            tmp_coef[-1] = 1
            basis_coefs[var] = tmp_coef
        return basis_coefs

    @staticmethod
    def _collocation_nodes(polynomials):
        """Return roots of suitable basis polynomial as collocation nodes."""
        return {var: poly.roots() for var, poly in polynomials.items()}

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
        for var in coefs_dict.keys():
            coef_array = coefs_dict[var]
            degrees[var] = coef_array.size - 1
        return degrees

    def solve(self, kind, coefs_dict, domain, method="hybr", **kwargs):
        """Solve a boundary value problem using orthogonal collocation."""
        # store the configuration
        self._kind = kind
        self._domain = domain
        self._degrees = self._infer_degrees(coefs_dict)

        # solve for the optimal coefficients
        initial_guess = self._coefs_dict_to_array(coefs_dict)
        self._result = optimize.root(self._evaluate_collocation_residuals,
                                     x0=initial_guess,
                                     args=(kind, domain, self.degrees),
                                     method=method,
                                     **kwargs)

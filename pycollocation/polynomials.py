"""
Classes for solving models using collocation with various polynomials as the
underlying basis functions.

@author: davidrpugh

"""
import numpy as np

from . import solvers


class PolynomialSolver(solvers.SolverBase):

    _valid_kinds = ["Standard", "Chebyshev", "Hermite", "Laguerre", "Legendre"]

    @staticmethod
    def _basis_polynomial_coefs(degree):
        """Return coefficients for a basis polynomial of a given degree."""
        return np.append(np.zeros(degree), 1)

    @classmethod
    def _basis_polynomial_factory(cls, coef, domain, kind, order):
        """Return a polynomial given some coefficients."""
        if kind == "Standard":
            poly = np.polynomial.Polynomial(coef, domain).deriv(order)
        elif kind == "Chebyshev":
            poly = np.polynomial.Chebyshev(coef, domain).deriv(order)
        elif kind == "Hermite":
            poly = np.polynomial.Hermite(coef, domain).deriv(order)
        elif kind == "Laguerre":
            poly = np.polynomial.Laguerre(coef, domain).deriv(order)
        elif kind == "Legendre":
            poly = np.polynomial.Legendre(coef, domain).deriv(order)
        else:
            mesg = "Parameter 'kind' must be one of {}, {}, {}, or {}."
            raise ValueError(mesg.format(*cls._valid_kinds))
        return poly

    @classmethod
    def basis_derivs_factory(cls, coef, domain, kind):
        """
        Given some coefficients, return a the derivative of a certain kind of
        orthogonal polynomial defined over a specific domain.

        """
        return cls._basis_polynomial_factory(coef, domain, kind, order=1)

    @classmethod
    def basis_funcs_factory(cls, coef, domain, kind):
        """
        Given some coefficients, return a certain kind of orthogonal polynomial
        defined over a specific domain.

        """
        return cls._basis_polynomial_factory(coef, domain, kind, order=0)

    @classmethod
    def collocation_nodes(cls, degree, domain, kind):
        """Return optimal collocation nodes for some orthogonal polynomial."""
        basis_coefs = cls._basis_polynomial_coefs(degree)
        basis_poly = cls.basis_funcs_factory(basis_coefs, domain, kind)
        return basis_poly.roots()

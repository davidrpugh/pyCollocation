"""
Class for approximating the solution to two-point boundary value problems using
either standard or orthogonal polynomials as basis functions.

@author: davidrpugh

"""
import numpy as np

from . import solvers


class PolynomialSolver(solvers.SolverBase):

    @staticmethod
    def _basis_monomial_coefs(degree):
        """Return coefficients for a monomial of a given degree."""
        return np.append(np.zeros(degree), 1)

    @classmethod
    def _basis_polynomial_factory(cls, coef, domain, kind, deriv):
        """Return a polynomial given some coefficients."""
        basis_polynomial = getattr(np.polynomial, kind)
        if not deriv:
            return basis_polynomial(coef, domain)
        else:
            return basis_polynomial(coef, domain).deriv()

    @classmethod
    def basis_derivs_factory(cls, coef, domain, kind, **kwargs):
        """
        Given some coefficients, return a the derivative of a certain kind of
        orthogonal polynomial defined over a specific domain.

        """
        return cls._basis_polynomial_factory(coef, domain, kind, True)

    @classmethod
    def basis_funcs_factory(cls, coef, domain, kind, **kwargs):
        """
        Given some coefficients, return a certain kind of orthogonal polynomial
        defined over a specific domain.

        """
        return cls._basis_polynomial_factory(coef, domain, kind, False)

    @classmethod
    def collocation_nodes(cls, degree, domain, kind):
        """Return optimal collocation nodes for some orthogonal polynomial."""
        basis_coefs = cls._basis_monomial_coefs(degree)
        basis_poly = cls.basis_funcs_factory(basis_coefs, domain, kind)
        return basis_poly.roots()

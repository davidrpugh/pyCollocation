"""
Class for approximating the solution to two-point boundary value problems using
B-splines as basis functions.

@author: davidrpugh

"""
import functools

from scipy import interpolate

from . import basis_functions


class BSplineBasis(basis_functions.BasisFunctionLike):

    @staticmethod
    def _basis_spline_factory(coef, degree, knots, der, ext):
        """Return a B-Spline given some coefficients."""
        return functools.partial(interpolate.splev, tck=(knots, coef, degree), der=der, ext=ext)

    @classmethod
    def derivatives_factory(cls, coef, degree, knots, ext, **kwargs):
        """
        Given some coefficients, return a the derivative of a B-spline.

        """
        return cls._basis_spline_factory(coef, degree, knots, 1, ext)

    @classmethod
    def fit(cls, *args, **kwargs):
        """Possibly just wrap interpolate.splprep?"""
        return interpolate.splprep(*args, **kwargs)

    @classmethod
    def functions_factory(cls, coef, degree, knots, ext, **kwargs):
        """
        Given some coefficients, return a B-spline.

        """
        return cls._basis_spline_factory(coef, degree, knots, 0, ext)

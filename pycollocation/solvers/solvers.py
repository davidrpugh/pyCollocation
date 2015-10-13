import functools

import numpy as np
from scipy import optimize

from . import solutions


class SolverLike(object):
    """
    Class describing the protocol the all SolverLike objects should satisfy.

    Notes
    -----
    Subclasses should implement `solve` method as described below.

    """

    @property
    def basis_functions(self):
        r"""
        Functions used to approximate the solution to a boundary value problem.

        :getter: Return the current basis functions.
        :type: `basis_functions.BasisFunctions`

        """
        return self._basis_functions

    @staticmethod
    def _array_to_list(coefs_array, indices_or_sections, axis=0):
        """Split an array into a list of arrays."""
        return np.split(coefs_array, indices_or_sections, axis)

    @staticmethod
    def _evaluate_functions(funcs, points):
        """Evaluate a list of functions at some points."""
        return [func(points) for func in funcs]

    @classmethod
    def _evaluate_rhs(cls, funcs, nodes, problem):
        """
        Compute the value of the right-hand side of the system of ODEs.

        Parameters
        ----------
        basis_funcs : list(function)
        nodes : numpy.ndarray
        problem : TwoPointBVPLike

        Returns
        -------
        evaluated_rhs : list(float)

        """
        evald_funcs = cls._evaluate_functions(funcs, nodes)
        evald_rhs = problem.rhs(nodes, *evald_funcs, **problem.params)
        return evald_rhs

    @classmethod
    def _lower_boundary_residual(cls, funcs, problem, ts):
        evald_funcs = cls._evaluate_functions(funcs, ts)
        return problem.bcs_lower(ts, *evald_funcs, **problem.params)

    @classmethod
    def _upper_boundary_residual(cls, funcs, problem, ts):
        evald_funcs = cls._evaluate_functions(funcs, ts)
        return problem.bcs_upper(ts, *evald_funcs, **problem.params)

    @classmethod
    def _compute_boundary_residuals(cls, boundary_points, funcs, problem):
        boundary_residuals = []
        if problem.bcs_lower is not None:
            residual = cls._lower_boundary_residual_factory(funcs, problem)
            boundary_residuals.append(residual(boundary_points[0]))
        if problem.bcs_upper is not None:
            residual = cls._upper_boundary_residual_factory(funcs, problem)
            boundary_residuals.append(residual(boundary_points[1]))
        return boundary_residuals

    @classmethod
    def _compute_interior_residuals(cls, derivs, funcs, nodes, problem):
        interior_residuals = cls._interior_residuals_factory(derivs, funcs, problem)
        residuals = interior_residuals(nodes)
        return residuals

    @classmethod
    def _interior_residuals(cls, derivs, funcs, problem, ts):
        evaluated_lhs = cls._evaluate_functions(derivs, ts)
        evaluated_rhs = cls._evaluate_rhs(funcs, ts, problem)
        return [lhs - rhs for lhs, rhs in zip(evaluated_lhs, evaluated_rhs)]

    @classmethod
    def _interior_residuals_factory(cls, derivs, funcs, problem):
        return functools.partial(cls._interior_residuals, derivs, funcs, problem)

    @classmethod
    def _lower_boundary_residual_factory(cls, funcs, problem):
        return functools.partial(cls._lower_boundary_residual, funcs, problem)

    @classmethod
    def _upper_boundary_residual_factory(cls, funcs, problem):
        return functools.partial(cls._upper_boundary_residual, funcs, problem)

    def _assess_approximation(self, boundary_points, derivs, funcs, nodes, problem):
        """
        Parameters
        ----------
        basis_derivs : list(function)
        basis_funcs : list(function)
        problem : TwoPointBVPLike

        Returns
        -------
        resids : numpy.ndarray

        """
        interior_residuals = self._compute_interior_residuals(derivs, funcs,
                                                              nodes, problem)
        boundary_residuals = self._compute_boundary_residuals(boundary_points,
                                                              funcs, problem)
        return np.hstack(interior_residuals + boundary_residuals)

    def _compute_residuals(self, coefs_array, basis_kwargs, boundary_points, nodes, problem):
        """
        Return collocation residuals.

        Parameters
        ----------
        coefs_array : numpy.ndarray
        basis_kwargs : dict
        problem : TwoPointBVPLike

        Returns
        -------
        resids : numpy.ndarray

        """
        coefs_list = self._array_to_list(coefs_array, problem.number_odes)
        derivs, funcs = self._construct_approximation(basis_kwargs, coefs_list)
        resids = self._assess_approximation(boundary_points, derivs, funcs,
                                            nodes, problem)
        return resids

    def _construct_approximation(self, basis_kwargs, coefs_list):
        """
        Construct a collection of derivatives and functions that approximate
        the solution to the boundary value problem.

        Parameters
        ----------
        basis_kwargs : dict(str: )
        coefs_list : list(numpy.ndarray)

        Returns
        -------
        basis_derivs : list(function)
        basis_funcs : list(function)

        """
        derivs = self._construct_derivatives(coefs_list, **basis_kwargs)
        funcs = self._construct_functions(coefs_list, **basis_kwargs)
        return derivs, funcs

    def _construct_derivatives(self, coefs, **kwargs):
        """Return a list of derivatives given a list of coefficients."""
        return [self.basis_functions.derivatives_factory(coef, **kwargs) for coef in coefs]

    def _construct_functions(self, coefs, **kwargs):
        """Return a list of functions given a list of coefficients."""
        return [self.basis_functions.functions_factory(coef, **kwargs) for coef in coefs]

    def _solution_factory(self, basis_kwargs, coefs_array, nodes, problem, result):
        """
        Construct a representation of the solution to the boundary value problem.

        Parameters
        ----------
        basis_kwargs : dict(str : )
        coefs_array : numpy.ndarray
        problem : TwoPointBVPLike
        result : OptimizeResult

        Returns
        -------
        solution : SolutionLike

        """
        soln_coefs = self._array_to_list(coefs_array, problem.number_odes)
        soln_derivs = self._construct_derivatives(soln_coefs, **basis_kwargs)
        soln_funcs = self._construct_functions(soln_coefs, **basis_kwargs)
        soln_residual_func = self._interior_residuals_factory(soln_derivs,
                                                              soln_funcs,
                                                              problem)
        solution = solutions.Solution(basis_kwargs, soln_funcs, nodes, problem,
                                      soln_residual_func, result)
        return solution

    def solve(self, basis_kwargs, boundary_points, coefs_array, nodes, problem,
              **solver_options):
        """
        Solve a boundary value problem using the collocation method.

        Parameters
        ----------
        basis_kwargs : dict
            Dictionary of keyword arguments used to build basis functions.
        coefs_array : numpy.ndarray
            Array of coefficients for basis functions defining the initial
            condition.
        problem : bvp.TwoPointBVPLike
            A two-point boundary value problem (BVP) to solve.
        solver_options : dict
            Dictionary of options to pass to the non-linear equation solver.

        Return
        ------
        solution: solutions.SolutionLike
            An instance of the SolutionLike class representing the solution to
            the two-point boundary value problem (BVP)

        Notes
        -----

        """
        raise NotImplementedError


class Solver(SolverLike):

    def __init__(self, basis_functions):
        self._basis_functions = basis_functions

    def solve(self, basis_kwargs, boundary_points, coefs_array, nodes, problem,
              **solver_options):
        """
        Solve a boundary value problem using the collocation method.

        Parameters
        ----------
        basis_kwargs : dict
            Dictionary of keyword arguments used to build basis functions.
        coefs_array : numpy.ndarray
            Array of coefficients for basis functions defining the initial
            condition.
        problem : bvp.TwoPointBVPLike
            A two-point boundary value problem (BVP) to solve.
        solver_options : dict
            Dictionary of options to pass to the non-linear equation solver.

        Return
        ------
        solution: solutions.SolutionLike
            An instance of the SolutionLike class representing the solution to
            the two-point boundary value problem (BVP)

        Notes
        -----

        """
        result = optimize.root(self._compute_residuals,
                               x0=coefs_array,
                               args=(basis_kwargs, boundary_points, nodes, problem),
                               **solver_options)
        solution = self._solution_factory(basis_kwargs, result.x, nodes,
                                          problem, result)
        return solution

import functools

import numpy as np
from scipy import optimize

from . import solutions


class SolverLike(object):

    @property
    def basis_functions(self):
        return self._basis_functions

    @staticmethod
    def _array_to_list(coefs_array, indices_or_sections, axis=0):
        """Splits an array into sections."""
        return np.split(coefs_array, indices_or_sections, axis)

    def _approximate_soln(self, basis_kwargs, coefs_list):
        """
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

    @classmethod
    def _assess_approximate_soln(cls, derivs, funcs, basis_kwargs, nodes, problem):
        """
        Parameters
        ----------
        basis_derivs : list(function)
        basis_funcs : list(function)
        nodes : numpy.ndarray
        problem : TwoPointBVPLike

        Returns
        -------
        resids : numpy.ndarray

        """
        resid_func = cls._residual_function_factory(derivs, funcs, problem)
        interior_resids = resid_func(nodes)
        boundary_resids = cls._compute_boundary_resids(funcs, basis_kwargs, problem)
        resids = np.hstack(interior_resids + boundary_resids)
        return resids

    @classmethod
    def _compute_boundary_resids(cls, funcs, basis_kwargs, problem):
        boundary_resids = []
        if problem.bcs_lower is not None:
            boundary_pt = basis_kwargs['domain'][0]
            evald_basis_funcs = [func(boundary_pt) for func in funcs]
            evald_bcs_lower = problem.bcs_lower(boundary_pt, *evald_basis_funcs, **problem.params)
            boundary_resids.append(evald_bcs_lower)
        if problem.bcs_upper is not None:
            boundary_pt = basis_kwargs['domain'][1]
            evald_basis_funcs = [func(boundary_pt) for func in funcs]
            evald_bcs_upper = problem.bcs_upper(boundary_pt, *evald_basis_funcs, **problem.params)
            boundary_resids.append(evald_bcs_upper)
        return boundary_resids

    def _compute_collocation_resids(self, coefs_array, basis_kwargs, nodes, problem):
        """Return collocation residuals."""
        coefs_list = self._array_to_list(coefs_array, problem.number_odes)
        basis_derivs, basis_funcs = self._approximate_soln(basis_kwargs,
                                                           coefs_list)
        collocation_resids = self._assess_approximate_soln(basis_derivs,
                                                           basis_funcs,
                                                           basis_kwargs,
                                                           nodes,
                                                           problem)
        return collocation_resids

    @classmethod
    def _compute_rhs(cls, funcs, nodes, problem):
        """
        Compute the value of the right-hand side (RHS) of the...

        Parameters
        ----------
        basis_funcs : list(function)
        nodes : numpy.ndarray
        problem : TwoPointBVPLike

        Returns
        -------
        evaluated_rhs : list(float)

        """
        evald_basis_funcs = [func(nodes) for func in funcs]
        evaluated_rhs = problem.rhs(nodes, *evald_basis_funcs, **problem.params)
        return evaluated_rhs

    @classmethod
    def _residual_function(cls, ts, derivs, funcs, problem):
        evaluated_lhs = [deriv(ts) for deriv in derivs]
        evaluated_rhs = cls._compute_rhs(funcs, ts, problem)
        return [lhs - rhs for lhs, rhs in zip(evaluated_lhs, evaluated_rhs)]

    @classmethod
    def _residual_function_factory(cls, derivs, funcs, problem):
        return functools.partial(cls._residual_function, derivs=derivs, funcs=funcs, problem=problem)

    def _construct_derivatives(self, coefs, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [self.basis_functions.derivatives_factory(coef, **kwargs) for coef in coefs]

    def _construct_functions(self, coefs, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [self.basis_functions.functions_factory(coef, **kwargs) for coef in coefs]

    def _construct_soln(self, basis_kwargs, nodes, problem, result):
        """
        Construct a representation of the solution to the boundary value problem.

        Parameters
        ----------
        basis_kwargs : dict(str : )
        nodes : numpy.ndarray
        problem : TwoPointBVPLike
        result : OptimizeResult

        Returns
        -------
        solution : SolutionLike

        """
        soln_coefs = self._array_to_list(result.x, problem.number_odes)
        soln_derivs = self._construct_derivatives(soln_coefs, **basis_kwargs)
        soln_funcs = self._construct_functions(soln_coefs, **basis_kwargs)
        soln_residual_func = self._residual_function_factory(soln_derivs, soln_funcs, problem)
        solution = solutions.Solution(basis_kwargs, soln_funcs, nodes, problem,
                                      soln_residual_func, result)
        return solution

    def solve(self, basis_kwargs, coefs_array, problem, **solver_options):
        """
        Solve a boundary value problem using orthogonal collocation.

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
        nodes = self.basis_functions.nodes(**basis_kwargs)
        result = optimize.root(self._compute_collocation_resids,
                               x0=coefs_array,
                               args=(basis_kwargs, nodes, problem),
                               **solver_options)
        solution = self._construct_soln(basis_kwargs, nodes, problem, result)
        return solution


class Solver(SolverLike):

    def __init__(self, basis_functions):
        self._basis_functions = basis_functions

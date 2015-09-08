import numpy as np
from scipy import optimize

from . import solutions


class SolverBase(object):

    @staticmethod
    def _array_to_list(coefs_array, indices_or_sections, axis=0):
        """Splits an array into equal sections."""
        return np.split(coefs_array, indices_or_sections, axis)

    @staticmethod
    def _evaluate_functions(functions, xs):
        """Return a list of functions evaluated at specified points."""
        return [f(xs) for f in functions]

    @classmethod
    def _approximate_soln(cls, basis_kwargs, coefs_list, domain):
        basis_derivs = cls._construct_basis_derivs(coefs_list, domain, **basis_kwargs)
        basis_funcs = cls._construct_basis_funcs(coefs_list, domain, **basis_kwargs)
        return basis_derivs, basis_funcs

    @classmethod
    def _assess_approximate_soln(cls, basis_derivs, basis_funcs, domain, nodes, problem):
        resid_func = cls._construct_residual_func(basis_derivs, basis_funcs, problem)
        interior_resids = resid_func(nodes)
        boundary_resids = cls._compute_boundary_resids(basis_funcs, domain, problem)
        resids = np.hstack(interior_resids + boundary_resids)
        return resids

    @classmethod
    def _compute_bcs_lower(cls, basis_funcs, boundary, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_funcs, boundary)
        return problem.bcs_lower(boundary, *evaluated_basis_funcs, **problem.params)

    @classmethod
    def _compute_bcs_upper(cls, basis_funcs, boundary, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_funcs, boundary)
        return problem.bcs_upper(boundary, *evaluated_basis_funcs, **problem.params)

    @classmethod
    def _compute_boundary_resids(cls, basis_funcs, domain, problem):
        boundary_resids = []
        if problem.bcs_lower is not None:
            args = (basis_funcs, domain[0], problem)
            boundary_resids.append(cls._compute_bcs_lower(*args))
        if problem.bcs_upper is not None:
            args = (basis_funcs, domain[1], problem)
            boundary_resids.append(cls._compute_bcs_upper(*args))
        return boundary_resids

    @classmethod
    def _compute_collocation_resids(cls, coefs_array, basis_kwargs, domain,
                                    nodes, problem):
        """Return collocation residuals."""
        coefs_list = cls._array_to_list(coefs_array, problem.number_odes)
        basis_derivs, basis_funcs = cls._approximate_soln(basis_kwargs,
                                                          coefs_list,
                                                          domain)
        collocation_resids = cls._assess_approximate_soln(basis_derivs,
                                                          basis_funcs,
                                                          domain,
                                                          nodes,
                                                          problem)
        return collocation_resids

    @classmethod
    def _compute_rhs(cls, basis_funcs, nodes, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_funcs, nodes)
        evaluated_rhs = problem.rhs(nodes, *evaluated_basis_funcs, **problem.params)
        return evaluated_rhs

    @classmethod
    def _construct_basis_derivs(cls, coefs, domain, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [cls.basis_derivs_factory(coef, domain, **kwargs) for coef in coefs]

    @classmethod
    def _construct_basis_funcs(cls, coefs, domain, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [cls.basis_funcs_factory(coef, domain, **kwargs) for coef in coefs]

    @classmethod
    def _construct_residual_func(cls, basis_derivs, basis_funcs, problem):

        def residual_func(points):
            evaluated_lhs = cls._evaluate_functions(basis_derivs, points)
            evaluated_rhs = cls._compute_rhs(basis_funcs, points, problem)
            return [lhs - rhs for lhs, rhs in zip(evaluated_lhs, evaluated_rhs)]

        return residual_func

    @classmethod
    def _construct_soln(cls, basis_kwargs, domain, nodes, problem, result):
        soln_coefs = cls._array_to_list(result.x, problem.number_odes)
        soln_derivs = cls._construct_basis_derivs(soln_coefs, domain, **basis_kwargs)
        soln_funcs = cls._construct_basis_funcs(soln_coefs, domain, **basis_kwargs)
        soln_residual_func = cls._construct_residual_func(soln_derivs, soln_funcs, problem)
        solution = solutions.Solution(basis_kwargs, domain, soln_funcs, nodes,
                                      problem, soln_residual_func, result)
        return solution

    @classmethod
    def basis_derivs_factory(cls, coef, domain, **kwargs):
        raise NotImplementedError

    @classmethod
    def basis_funcs_factory(cls, coef, domain, **kwargs):
        raise NotImplementedError

    @classmethod
    def solve(cls, basis_kwargs, coefs_array, domain, nodes, problem,
              **solver_options):
        """
        Solve a boundary value problem using orthogonal collocation.

        Parameters
        ----------
        basis_kwargs : dict
            Dictionary of keyword arguments used to build basis functions.
        coefs_array : numpy.ndarray
            Array of coefficients for basis functions defining the initial
            condition.
        domain : array_like
        nodes : numpy.ndarray
        problem : bvp.BVPLike
            Instance of a BVP problem to solve.
        solver_options : dict
            Dictionary of options to pass to the non-linear equation solver.

        Return
        ------
        result : scipy.optimize.Result
            Result object.

        Notes
        -----
        Collocation method converts functional equation into a system of
        non-linear equations.

        """
        result = optimize.root(cls._compute_collocation_resids,
                               x0=coefs_array,
                               args=(basis_kwargs, domain, nodes, problem),
                               **solver_options)
        solution = cls._construct_soln(basis_kwargs, domain, nodes, problem, result)
        return solution

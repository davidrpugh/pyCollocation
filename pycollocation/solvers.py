import warnings

import numpy as np
from scipy import optimize


class SolverBase(object):

    @staticmethod
    def _array_to_list(coefs_array, indices_or_sections, axis=0):
        """Splits an array into equal sections."""
        return np.split(coefs_array, indices_or_sections, axis)

    @staticmethod
    def _evaluate_functions(functions, xs):
        """Return a list of functions evaluated at given nodes."""
        return [f(xs) for f in functions]

    @classmethod
    def _assess_approx_soln(cls, approx_soln, domain, nodes, params, problem):
        interior_resids = cls._evaluate_interior_resids(approx_soln, nodes, params, problem)
        boundary_resids = cls._evaluate_boundary_resids(approx_soln, domain, params, problem)
        return np.hstack(interior_resids + boundary_resids)

    @classmethod
    def _construct_approx_soln(cls, basis_kwargs, coefs_list, domain):
        basis_derivs = cls._construct_basis_derivs(coefs_list, domain, **basis_kwargs)
        basis_funcs = cls._construct_basis_funcs(coefs_list, domain, **basis_kwargs)
        return basis_derivs, basis_funcs

    @classmethod
    def _evaluate_bcs_lower(cls, basis_funcs, boundary, params, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_funcs, boundary)
        return problem.bcs_lower(boundary, *evaluated_basis_funcs, **params)

    @classmethod
    def _evaluate_bcs_upper(cls, basis_funcs, boundary, params, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_funcs, boundary)
        return problem.bcs_upper(boundary, *evaluated_basis_funcs, **params)

    @classmethod
    def _evaluate_boundary_resids(cls, approx_soln, domain, params, problem):
        boundary_resids = []
        _, basis_funcs = approx_soln
        if problem.bcs_lower is not None:
            args = (basis_funcs, domain[0], params, problem)
            boundary_resids.append(cls._evaluate_bcs_lower(*args))
        if problem.bcs_upper is not None:
            args = (basis_funcs, domain[1], params, problem)
            boundary_resids.append(cls._evaluate_bcs_upper(*args))
        return boundary_resids

    @classmethod
    def _evaluate_interior_resids(self, approx_soln, nodes, params, problem):
        """Return a list of residuals evaluated at the interior nodes."""
        basis_derivs, basis_funcs = approx_soln
        evaluated_lhs = self._evaluate_functions(basis_derivs, nodes)
        evaluated_rhs = self._evaluate_rhs(basis_funcs, nodes, params, problem)
        return [lhs - rhs for lhs, rhs in zip(evaluated_lhs, evaluated_rhs)]

    @classmethod
    def _evaluate_rhs(cls, basis_funcs, nodes, params, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_funcs, nodes)
        evaluated_rhs = problem.rhs(nodes, *evaluated_basis_funcs, **params)
        return evaluated_rhs

    @classmethod
    def _construct_basis_derivs(self, coefs, domain, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [self.basis_derivs_factory(coef, domain, **kwargs) for coef in coefs]

    @classmethod
    def _construct_basis_funcs(self, coefs, domain, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [self.basis_funcs_factory(coef, domain, **kwargs) for coef in coefs]

    @classmethod
    def basis_derivs_factory(cls, coef, domain, **kwargs):
        raise NotImplementedError

    @classmethod
    def basis_funcs_factory(cls, coef, domain, **kwargs):
        raise NotImplementedError

    @classmethod
    def _evaluate_collocation_resids(cls, coefs_array, basis_kwargs, domain,
                                     nodes, params, problem):
        """Return collocation residuals."""
        coefs_list = cls._array_to_list(coefs_array, problem.number_odes)
        approx_soln = cls._construct_approx_soln(basis_kwargs, coefs_list, domain)
        collocation_resids = cls._assess_approx_soln(approx_soln, domain, nodes, params, problem)
        return collocation_resids

    @classmethod
    def solution_funcs_factory(cls, result):
        if not result.success:
            mesg = ("Looks like the solver did not converge, interpret " +
                    "resulting functions with caution!")
            warnings.warn(mesg, RuntimeWarning)
        soln_coefs = cls._array_to_list(result.x, result.problem.number_odes)
        soln_funcs = cls._construct_basis_funcs(soln_coefs, result.domain, **result.basis_kwargs)
        return soln_funcs

    @classmethod
    def solution_residuals(cls, points, result):
        if not result.success:
            mesg = ("Looks like the solver did not converge, interpret " +
                    "residuals with caution!")
            warnings.warn(mesg, RuntimeWarning)
        args = (result.x, result.basis_kwargs, result.domain, points,
                result.params, result.problem)
        residuals = cls._evaluate_collocation_resids(*args)
        return residuals

    @classmethod
    def solve(cls, basis_kwargs, coefs_array, domain, nodes, params, problem, **solver_options):
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
        params : dict
            Dictionary of model parameters.
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
        values = (basis_kwargs, domain, nodes, params, problem)
        result = optimize.root(cls._evaluate_collocation_resids,
                               x0=coefs_array,
                               args=values,
                               **solver_options)

        # modify result object...
        keys = ['basis_kwargs', 'domain', 'nodes', 'params', 'problem']
        attributes = {k: v for k, v in zip(keys, values)}
        result.update(attributes)

        return result

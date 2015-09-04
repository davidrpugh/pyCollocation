import numpy as np
from scipy import optimize


class SolverBase(object):

    @staticmethod
    def _array_to_list(coefs_array, sections):
        """Splits an array into equal sections."""
        return np.split(coefs_array, sections)

    @staticmethod
    def _evaluate_functions(functions, xs):
        """Return a list of functions evaluated at given nodes."""
        return [f(xs) for f in functions]

    @classmethod
    def _evaluate_bcs_lower(cls, basis_funcs, boundary, params, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_funcs, boundary)
        return problem.bcs_lower(boundary, *evaluated_basis_funcs, **params)

    @classmethod
    def _evaluate_bcs_upper(cls, basis_funcs, boundary, params, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_funcs, boundary)
        return problem.bcs_upper(boundary, *evaluated_basis_funcs, **params)

    @classmethod
    def _evaluate_boundary_resids(cls, basis_funcs, domain, params, problem):
        boundary_resids = []
        if problem.bcs_lower is not None:
            args = (basis_funcs, domain[0], params, problem)
            boundary_resids.append(cls._evaluate_bcs_lower(*args))
        if problem.bcs_upper is not None:
            args = (basis_funcs, domain[1], params, problem)
            boundary_resids.append(cls._evaluate_bcs_upper(*args))
        return boundary_resids

    @classmethod
    def _evaluate_interior_resids(self, basis_derivs, basis_funcs, nodes, params, problem):
        """Return a list of residuals evaluated at the interior nodes."""
        evaluated_lhs = self._evaluate_functions(basis_derivs, nodes)
        evaluated_rhs = self._evaluate_rhs(basis_funcs, nodes, params, problem)
        return [lhs - rhs for lhs, rhs in zip(evaluated_lhs, evaluated_rhs)]

    @classmethod
    def _evaluate_rhs(cls, basis_funcs, nodes, params, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_funcs, nodes)
        evaluated_rhs = problem.rhs(nodes, *evaluated_basis_funcs, **params)
        return evaluated_rhs

    def _construct_basis_derivs(self, coefs, domain, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [self.basis_derivs_factory(coef, domain, **kwargs) for coef in coefs]

    def _construct_basis_funcs(self, coefs, domain, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [self.basis_funcs_factory(coef, domain, **kwargs) for coef in coefs]

    def _evaluate_collocation_resids(self, coefs_array, basis_kwargs, domain, nodes, params, problem):
        """Return collocation residuals."""
        # construct an approximation of the solution to the BVP
        coefs_list = self._array_to_list(coefs_array, problem.number_odes)
        basis_derivs = self._construct_basis_derivs(coefs_list, domain, **basis_kwargs)
        basis_funcs = self._construct_basis_funcs(coefs_list, domain, **basis_kwargs)

        # assess the quality of the approximation
        interior_resids = self._evaluate_interior_resids(basis_derivs, basis_funcs, nodes, params, problem)
        boundary_resids = self._evaluate_boundary_resids(basis_funcs, domain, params, problem)
        collocation_resids = np.hstack(interior_resids + boundary_resids)

        return collocation_resids

    def basis_derivs_factory(self, coef, domain, **kwargs):
        raise NotImplementedError

    def basis_funcs_factory(self, coef, domain, **kwargs):
        raise NotImplementedError

    def collocation_nodes(self, *args, **kwargs):
        raise NotImplementedError

    def solution_funcs_factory(self, basis_kwargs, domain, problem, result):
        if result.success:
            soln_coefs = self._array_to_list(result.x, problem.number_odes)
            return self._construct_basis_funcs(soln_coefs, domain, **basis_kwargs)
        else:
            raise ValueError

    def solution_residuals(self, basis_kwargs, domain, nodes, params, problem, result):
        if result.success:
            args = (basis_kwargs, domain, nodes, params, problem)
            residuals = self._evaluate_collocation_resids(result.x, *args)
            return residuals
        else:
            raise ValueError

    def solve(self, basis_kwargs, coefs_array, domain, nodes, params, problem,
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
        # solving for solution coefficients is "just" a root-finding problem!
        args = (basis_kwargs, domain, nodes, params, problem)
        result = optimize.root(self._evaluate_collocation_resids,
                               x0=coefs_array,
                               args=args,
                               **solver_options)
        return result

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

    @staticmethod
    def _evaluate_interior_resids(left_hand_sides, right_hand_sides):
        """Return a list of residuals evaluated at the interior nodes."""
        return [lhs - rhs for lhs, rhs in zip(left_hand_sides, right_hand_sides)]

    @classmethod
    def _evaluate_bcs_lower(cls, basis_functions, boundary, params, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_functions, boundary)
        return problem.bcs_lower(boundary, *evaluated_basis_funcs, **params)

    @classmethod
    def _evaluate_bcs_upper(cls, basis_functions, boundary, params, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_functions, boundary)
        return problem.bcs_upper(boundary, *evaluated_basis_funcs, **params)

    @classmethod
    def _evaluate_rhs(cls, basis_functions, nodes, params, problem):
        evaluated_basis_funcs = cls._evaluate_functions(basis_functions, nodes)
        evaluated_rhs = problem.rhs(nodes, *evaluated_basis_funcs, **params)
        return evaluated_rhs

    def _construct_basis_derivatives(self, coefs, domain, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [self.basis_derivative_factory(coef, domain, **kwargs) for coef in coefs]

    def _construct_basis_functions(self, coefs, domain, **kwargs):
        """Return a list of basis functions given a list of coefficients."""
        return [self.basis_function_factory(coef, domain, **kwargs) for coef in coefs]

    def _evaluate_collocation_resids(self, coefs_array, domain, nodes, params, problem, kwargs):
        """Return collocation residuals associated with the current set of coefficients."""
        coefs_list = self._array_to_list(coefs_array, problem.number_odes)
        basis_derivs = self._construct_basis_derivatives(coefs_list, domain, **kwargs)
        basis_funcs = self._construct_basis_functions(coefs_list, domain, **kwargs)

        evaluated_lhs = self._evaluate_functions(basis_derivs, nodes)
        evaluated_rhs = self._evaluate_rhs(basis_funcs, nodes, params, problem)

        interior_resids = self._evaluate_interior_resids(evaluated_lhs, evaluated_rhs)
        boundary_resids = (self._evaluate_bcs_lower(basis_funcs, domain[0], params, problem) +
                           self._evaluate_bcs_upper(basis_funcs, domain[1], params, problem))
        collocation_resids = interior_resids + boundary_resids

        return np.hstack(collocation_resids)

    def basis_derivative_factory(self, coef, domain, **kwargs):
        raise NotImplementedError

    def basis_functions_factory(self, coef, domain, **kwargs):
        raise NotImplementedError

    def collocation_nodes(self, *args, **kwargs):
        raise NotImplementedError

    def solve(self, coefs_array, domain, nodes, params, problem,
              solver_options=None, **kwargs):
        """Solve a boundary value problem using orthogonal collocation."""
        if solver_options is None:
            solver_options = {}

        result = optimize.root(self._evaluate_collocation_resids,
                               x0=coefs_array,
                               args=(domain, nodes, params, problem, kwargs),
                               **solver_options)

        if result.success:
            solution_coefs = self._array_to_list(result.x, problem.number_odes)
            return self._construct_basis_functions(solution_coefs, domain, **kwargs)
        else:
            return result

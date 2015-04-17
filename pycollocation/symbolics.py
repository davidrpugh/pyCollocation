"""
Classes for constructing symbolic models.

@author : davidrpugh

"""
import numpy as np
import sympy as sym


class SymbolicBase(object):

    _modules = [{'ImmutableMatrix': np.array}, 'numpy']

    @property
    def _symbolic_args(self):
        """List of symbolic arguments used to lambdify expressions."""
        raise NotImplementedError

    def _lambdify_factory(self, expr):
        """Lambdify a symbolic expression."""
        return sym.lambdify(self._symbolic_args, expr, self._modules)


class SymbolicModelLike(SymbolicBase):

    __symbolic_jacobian = None

    @property
    def _symbolic_args(self):
        """List of symbolic arguments used to lambdify expressions."""
        return self._symbolic_vars + self._symbolic_params

    @property
    def _symbolic_jacobian(self):
        """Symbolic Jacobian matrix of partial derivatives."""
        if self.__symbolic_jacobian is None:
            args = self.dependent_vars
            self.__symbolic_jacobian = self._symbolic_system.jacobian(args)
        return self.__symbolic_jacobian

    @property
    def _symbolic_params(self):
        """List of symbolic model parameters."""
        return sym.var(list(self.params.keys()))

    @property
    def _symbolic_system(self):
        """Represents rhs as a symbolic matrix."""
        return sym.Matrix([self.rhs[var] for var in self.dependent_vars])

    @property
    def _symbolic_vars(self):
        """List of symbolic model variables."""
        return sym.var([self.independent_var] + self.dependent_vars)

    def _clear_cache(self):
        """Clear cached symbolic Jacobian."""
        self.__symbolic_jacobian = None


class SymbolicBoundaryValueProblemLike(SymbolicModelLike):

    __lower_boundary_condition = None

    __upper_boundary_condition = None

    @property
    def _lower_boundary_condition(self):
        """Cache lambdified lower boundary condition for numerical evaluation."""
        condition = self.boundary_conditions['lower']
        if condition is not None:
            if self.__lower_boundary_condition is None:
                self.__lower_boundary_condition = self._lambdify_factory(condition)
            return self.__lower_boundary_condition
        else:
            return None

    @property
    def _upper_boundary_condition(self):
        """Cache lambdified upper boundary condition for numerical evaluation."""
        condition = self.boundary_conditions['upper']
        if condition is not None:
            if self.__upper_boundary_condition is None:
                self.__upper_boundary_condition = self._lambdify_factory(condition)
            return self.__upper_boundary_condition
        else:
            return None

    def _validate_boundary(self, conditions):
        """Validate a dictionary of lower and upper boundary conditions."""
        bcs = {'lower': self._validate_boundary_exprs(conditions['lower']),
               'upper': self._validate_boundary_exprs(conditions['upper'])}
        return bcs

    def _validate_boundary_exprs(self, expressions):
        """Check that lower/upper boundary_conditions are expressions."""
        if expressions is None:
            return None
        else:
            return [self._validate_expression(expr) for expr in expressions]

import numpy as np
import sympy as sym


class SymbolicModelLike(object):

    __symbolic_jacobian = None

    _modules = [{'ImmutableMatrix': np.array}, 'numpy']

    @property
    def _symbolic_args(self):
        """List of symbolic arguments used to lambdify expressions."""
        return self._symbolic_vars + self._symbolic_params

    @property
    def _symbolic_params(self):
        """List of symbolic model parameters."""
        return sym.var(list(self.params.keys()))

    @property
    def _symbolic_system(self):
        """Represents rhs as a symbolic matrix."""
        return sym.Matrix([self.rhs[var] for var in self.dependent_vars])

    @property
    def _symbolic_jacobian(self):
        """Symbolic Jacobian matrix of partial derivatives."""
        if self.__symbolic_jacobian is None:
            args = self.dependent_vars
            self.__symbolic_jacobian = self._symbolic_system.jacobian(args)
        return self.__symbolic_jacobian

    @property
    def _symbolic_vars(self):
        """List of symbolic model variables."""
        return [self.independent_var] + self.dependent_vars

    def _clear_cache(self):
        """Clear cached symbolic Jacobian."""
        self.__symbolic_jacobian = None

    def _lambdify_factory(self, expr):
        """Lambdify a symbolic expression."""
        return sym.lambdify(self._symbolic_args, expr, self._modules)

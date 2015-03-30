import numpy as np
import sympy as sym


class Symbolics(object):

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
    def _symbolic_vars(self):
        """List of symbolic model variables."""
        return [self.independent_var] + self.dependent_vars

    def _lambdify_factory(self, expr):
        """Lambdify a symbolic expression."""
        return sym.lambdify(self._symbolic_args, expr, self._modules)

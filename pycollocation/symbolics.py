"""
Classes for constructing symbolic models.

@author : David R. Pugh

"""
import numpy as np
import sympy as sym


class SymbolicLike(object):

    _modules = [{'ImmutableMatrix': np.array}, 'numpy']

    @property
    def _symbolic_args(self):
        """List of symbolic arguments used to lambdify expressions."""
        return self.__symbolic_args

    @_symbolic_args.setter
    def _symbolic_args(self, args):
        """Set new values for the symbolic arguments."""
        self.__symbolic_args = args

    def _lambdify_factory(self, expr):
        """Lambdify a symbolic expression."""
        return sym.lambdify(self._symbolic_args, expr, self._modules)

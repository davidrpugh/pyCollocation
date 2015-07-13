"""
Classes for finding fixed point equilibria of a system of difference or
differential equations.

@author : David R. Pugh

"""
import numpy as np
from scipy import optimize


class Equilibrium(object):
    """
    Class for computing fixed-point equilibria for a system of difference or
    differential equations.

    """

    def __init__(self, problem):
        """
        Create an instance of the Equilbrium class.

        Parameters
        ----------

        problem : bvp.SymbolicTwoPointBVPLike

        @TODO : Need to extend this class so that it will work with arbitrary
        bvp.TwoPointBVPLike object.

        """
        self.problem = problem

    def _equilibrium_system(self, X, t):
        """
        System of equations representing the right-hand side of a system of
        difference or differential equations.

        """
        args = np.hstack((np.array([t]), X, np.array(self.problem.params.values())))
        residuals = [self.problem._rhs_functions(var)(*args) for var in self.problem.dependent_vars]
        return np.array(residuals)

    def find_equilibrium(self, initial_guess, method='hybr', **solver_opts):
        """
        Compute the steady state values of capital and consumption
        (per unit effective labor).

        Parameters
        ----------
        initial_guess : np.ndarray
            Array of values representing the initial guess for an equilibrium.
        method : string (default='hybr')
            Method used to solve the system of non-linear equations. See
            `scipy.optimize.root` for more details.
        solver_opts : dictionary
            Dictionary of optional keyword arguments to pass to the non-linear
            equation solver.  See `scipy.optimize.root` for details.

        Returns
        -------
        result : scipy.optimze.Result
            Object representing the result of the non-linear equation solver.
            See `scipy.optimize.root` for details.

        """
        result = optimize.root(self._equilibrium_system,
                               x0=initial_guess,
                               args=(0.0,),  # independent var irrevelant at equilibrium?
                               method=method,
                               **solver_opts)
        return result

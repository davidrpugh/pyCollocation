"""
Solvers for over-identified systems.

@author : davidrpugh

"""
from scipy import optimize

from . import solvers


class LeastSquaresSolver(solvers.Solver):

    def solve(self, basis_kwargs, boundary_points, coefs_array, nodes, problem,
              **solver_options):
        """
        Solve a boundary value problem using the collocation method.

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
        result = optimize.leastsq(self._compute_residuals,
                                  x0=coefs_array,
                                  args=(basis_kwargs, boundary_points, nodes, problem),
                                  **solver_options)
        solution = self._solution_factory(basis_kwargs, result[0], nodes,
                                          problem, result)
        return solution

"""
Classes for constructing two-point boundary value problems.

@author : davidrpugh

"""
from . import differential_equations
from . import symbolics


class BoundaryValueProblem(object):

    @property
    def boundary_conditions(self):
        """
        Boundary conditions for the problem.

        :getter: Return the boundary conditions for the problem.
        :setter: Set new boundary conditions for the problem.
        :type: dict

        """
        return self._boundary_conditions

    @boundary_conditions.setter
    def boundary_conditions(self, conditions):
        """Set new boundary conditions for the model."""
        self._boundary_conditions = self._validate_boundary(conditions)

    def _sufficient_boundary(self, conditions):
        """Check that there are sufficient boundary conditions."""
        number_conditions = 0
        if conditions['lower'] is not None:
            number_conditions += len(conditions['lower'])
        if conditions['upper'] is not None:
            number_conditions += len(conditions['upper'])
        return number_conditions == len(self.dependent_vars)

    def _validate_boundary(self, conditions):
        """Validate a dictionary of lower and upper boundary conditions."""
        if not isinstance(conditions, dict):
            mesg = "Attribute `boundary_conditions` must have type `dict` not {}"
            raise AttributeError(mesg.format(conditions.__class__))
        elif not (set(conditions.keys()) < set(['lower', 'upper', None])):
            mesg = "Keys for `boundary_conditions` dict must be {}, {}, or {}"
            raise AttributeError(mesg.format('lower', 'upper', 'None'))
        elif not self._sufficient_boundary(conditions):
            mesg = "Number of conditions must equal number of dependent vars."
            raise ValueError(mesg)
        else:
            return conditions


class SymbolicBoundaryValueProblem(BoundaryValueProblem,
                                   symbolics.SymbolicBoundaryValueProblemLike,
                                   differential_equations.SymbolicDifferentialEquation):
    """Class for representing two-point boundary value problems."""

    def __init__(self, boundary_conditions, dependent_vars, independent_var,
                 rhs, params):
        """Create an instance of a two-point boundary value problem (BVP)."""
        super(SymbolicBoundaryValueProblem, self).__init__(dependent_vars,
                                                           independent_var,
                                                           rhs,
                                                           params)
        self.boundary_conditions = boundary_conditions

    def _validate_boundary(self, conditions):
        """Validate a dictionary of lower and upper boundary conditions."""
        super(SymbolicBoundaryValueProblem, self)._validate_boundary(conditions)
        bcs = {'lower': self._validate_boundary_exprs(conditions['lower']),
               'upper': self._validate_boundary_exprs(conditions['upper'])}
        return bcs

    def _validate_boundary_exprs(self, expressions):
        """Check that lower/upper boundary_conditions are expressions."""
        if expressions is None:
            return None
        else:
            return [self._validate_expression(expr) for expr in expressions]

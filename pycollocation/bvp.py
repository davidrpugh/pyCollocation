"""
Classes for representing two-point boundary value problems.

@author : David R. Pugh

"""
from . import equilibria
from . import models


class TwoPointBVPLike(object):

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


class SymbolicTwoPointBVPLike(TwoPointBVPLike, models.SymbolicModelLike):

    __lower_boundary_condition = None

    __upper_boundary_condition = None

    @property
    def equilibrium(self):
        """
        Object representing the model equilibrium.

        :getter: Return the current object.
        :type: equilibria.Equilibrium

        """
        return equilibria.Equilibrium(self)

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


class SymbolicTwoPointBVP(SymbolicTwoPointBVPLike):
    """Class for representing symbolic two-point boundary value problems."""

    def __init__(self, boundary_conditions, dependent_vars, independent_var, params, rhs):
        """Create an instance of a two-point boundary value problem (BVP)."""
        self.boundary_conditions = boundary_conditions
        self.dependent_vars = dependent_vars
        self.independent_var = independent_var
        self.params = params
        self.rhs = rhs

    def _validate_boundary(self, conditions):
        """Validate a dictionary of lower and upper boundary conditions."""
        super(SymbolicTwoPointBVP, self)._validate_boundary(conditions)
        bcs = {'lower': self._validate_boundary_exprs(conditions['lower']),
               'upper': self._validate_boundary_exprs(conditions['upper'])}
        return bcs

    def _validate_boundary_exprs(self, expressions):
        """Check that lower/upper boundary_conditions are expressions."""
        if expressions is None:
            return None
        else:
            return [self._validate_expression(expr) for expr in expressions]


class TwoPointBVP(TwoPointBVPLike):
    """Class for representing two-point boundary value problems."""

    def __init__(self, boundary_conditions, dependent_vars, independent_var, params, rhs):
        """Create an instance of a two-point boundary value problem (BVP)."""
        self.boundary_conditions = boundary_conditions
        self.dependent_vars = dependent_vars
        self.independent_var = independent_var
        self.params = params
        self.rhs = rhs

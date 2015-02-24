import sympy as sym


class SymbolicModel(object):
    """Base class for all symbolic models."""

    @property
    def dependent_vars(self):
        """
        Model dependent variables.

        :getter: Return the model dependent variables.
        :setter: Set new model dependent variables.
        :type: list

        """
        return self._dependent_variables

    @dependent_vars.setter
    def dependent_vars(self, symbols):
        """Set new list of dependent variables."""
        self._dependent_variables = self._validate_symbols(symbols)

    @property
    def independent_var(self):
        """
        Symbolic variable representing the independent variable.

        :getter: Return the symbol representing the independent variable.
        :setter: Set a new symbol to represent the independent variable.
        :type: sympy.Symbol

        """
        return self._independent_var

    @independent_var.setter
    def independent_var(self, symbol):
        """Set a new symbol to represent the independent variable."""
        self._independent_var = self._validate_symbol(symbol)

    @property
    def rhs(self):
        """
        Symbolic representation of the right-hand side of a system of
        differential/difference equations.

        :getter: Return the right-hand side of the system of equations.
        :setter: Set a new right-hand side of the system of equations.
        :type: dict

        """
        return self._rhs

    @rhs.setter
    def rhs(self, equations):
        """Set a new right-hand side of the system of equations."""
        self._rhs = self._validate_rhs(equations)
        self._clear_cache

    @staticmethod
    def _validate_expression(expression):
        """Validates a symbolic expression."""
        if not isinstance(expression, sym.Basic):
            mesg = "Attribute must be of type `sympy.Basic` not {}"
            raise AttributeError(mesg.format(expression.__class__))
        else:
            return expression

    def _validate_rhs(self, rhs):
        """Validate a the rhs attribute."""
        if not isinstance(rhs, dict):
            mesg = "Attribute `rhs` must be of type `dict` not {}"
            raise AttributeError(mesg.format(rhs.__class__))
        elif not (len(rhs) == len(self.dependent_vars)):
            mesg = "Number of equations must equal number of dependent vars."
            raise ValueError(mesg)
        else:
            exprs = {}
            for var, expr in rhs.iteritems():
                exprs[var] = self._validate_expression(expr)
            return exprs

    @staticmethod
    def _validate_symbol(symbol):
        """Validate the independent_var attribute."""
        if not isinstance(symbol, sym.Symbol):
            mesg = "Attribute must be of type `sympy.Symbol` not {}"
            raise AttributeError(mesg.format(symbol.__class__))
        else:
            return symbol

    @classmethod
    def _validate_symbols(cls, symbols):
        """Validate the dependent_vars attribute."""
        return [cls._validate_symbol(symbol) for symbol in symbols]


class DifferentialEquation(SymbolicModel):

    __symbolic_jacobian = None

    def __init__(self, dependent_vars, independent_var, rhs):
        """Create an instance of the DifferentialEquation class."""
        self.dependent_vars = dependent_vars
        self.independent_var = independent_var
        self.rhs = rhs

    @property
    def _symbolic_system(self):
        """Represents rhs as a symbolic matrix."""
        return sym.Matrix([self.rhs[var] for var in self.dependent_vars])

    @property
    def jacobian(self):
        """
        Symbolic Jacobian matrix of partial derivatives.

        :getter: Return the Jacobian matrix.
        :type: sympy.Matrix

        """
        if self.__symbolic_jacobian is None:
            args = self.dependent_vars
            self.__symbolic_jacobian = self._symbolic_system.jacobian(args)
        return self.__symbolic_jacobian

    def _clear_cache(self):
        """Clear cached symbolic Jacobian."""
        self.__symbolic_jacobian = None


class BoundaryValueProblem(DifferentialEquation):
    """Class for representing two-point boundary value problems."""

    def __init__(self, boundary_conditions, dependent_vars, independent_var, rhs):
        """
        Create an instance of a two-point boundary value problem (BVP).

        """
        super(BoundaryValueProblem, self).__init__(dependent_vars,
                                                   independent_var,
                                                   rhs)
        self.boundary_conditions = boundary_conditions

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
            bcs = {'lower': self._validate_boundary_exprs(conditions['lower']),
                   'upper': self._validate_boundary_exprs(conditions['upper'])}
            return bcs

    def _validate_boundary_exprs(self, expressions):
        """Check that lower/upper boundary_conditions are expressions."""
        if expressions is None:
            return None
        else:
            return [self._validate_expression(expr) for expr in expressions]

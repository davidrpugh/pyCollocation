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
        :type: sympy.Matrix

        """
        return self._rhs

    @rhs.setter
    def rhs(self, system):
        """Set a new right-hand side of the system of equations."""
        self._rhs = self._validate_rhs(system)
        self._clear_cache

    @staticmethod
    def _validate_equation(equation):
        """Validates a symbolic equation."""
        if not isinstance(equation, sym.Basic):
            mesg = "Attribute must be of type `sympy.Basic` not {}"
            raise AttributeError(mesg.format(equation.__class__))
        else:
            return equation

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

    def _validate_rhs(self, rhs):
        """Validate the rhs attribute."""
        if not isinstance(rhs, list):
            mesg = "Attribute must be of type `list` not {}"
            raise AttributeError(mesg.format(rhs.__class__))
        elif not (len(rhs) == len(self.dependent_vars)):
            mesg = "Number of equations must equal number of dependent vars."
            raise ValueError(mesg)
        else:
            return [self._validate_equation(eqn) for eqn in rhs]


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
        return sym.Matrix(self.rhs)

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


if __name__ == '__main__':

    # define some variables
    t, k, c = sym.symbols('t, k, c')

    # define some parameters
    alpha, sigma = sym.symbols('alpha, sigma')
    rho, theta = sym.symbols('rho, theta')
    g, n, s, delta = sym.symbols('g, n, s, delta')

    # intensive output has the CES form
    y = (alpha * k**((sigma - 1) / sigma) + (1 - alpha))**(sigma / (sigma - 1))

    # define the Solow model
    k_dot = s * y - (g + n + delta) * k

    solow = DifferentialEquation(dependent_vars=[k],
                                 independent_var=t,
                                 rhs=[k_dot])

    # define the Ramsey-Cass-Coopmans model
    mpk = sym.diff(y, k, 1)
    k_dot = y - c - (g + n) * k
    c_dot = ((mpk - rho - theta * g) / theta) * c

    ramsey = DifferentialEquation(dependent_vars=[k, c],
                                  independent_var=t,
                                  rhs=[k_dot, c_dot])

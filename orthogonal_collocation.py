import numpy as np
from scipy import optimize

import solvers


class OrthogonalCollocation(solvers.Solver):
    """Base class for OrthogonalCollocation classes."""

    @staticmethod
    def _basis_coef(degree):
        """Return coefficients for the basis polynomial of a given degree."""
        basis_coef = np.zeros(degree + 1)
        basis_coef[-1] = 1
        return basis_coef

    @property
    def _coefs(self):
        """Dictionary mapping a symbol to the coefs of its polynomial."""
        return self.__coefs

    @_coefs.setter
    def _coefs(self, coefs):
        """Set a new dictionary of polynomial coefs."""
        self.__coefs = coefs

    @property
    def _orthogonal_polys(self):
        """Dictionary mapping a symbol to its approximating poly."""
        polys = {}
        for symbol, coef in self._coefs.iteritems():
            polys[symbol] = self._orthogonal_poly_factory(coef)
        return polys

    @property
    def _orthogonal_poly_derivs(self):
        """Dictionary mapping a symbol to the derivative of its polynomial."""
        derivs = {}
        for symbol, poly in self._orthogonal_polys.iteritems():
            derivs[symbol] = poly.deriv()
        return derivs

    @property
    def _residual_funcs(self):
        """Dictionary mappng a symbol to its residual function."""
        residual_funcs = {}
        for var in self.model.dependent_vars:
            residual_funcs[var] = self._residual_function_factory(var)
        return residual_funcs

    @property
    def domain(self):
        """
        Domain over which the polynomial approximation is valid.

        :getter: Return the approximation domain.
        :setter: Set a new approximation domain.
        :type: list

        """
        return self._domain

    @domain.setter
    def domain(self, value):
        """Set a new approximation domain."""
        self._domain = self._validate_domain(value)

    @property
    def kind(self):
        """
        Kind of polynomials to use when constructing the approximation.

        :getter: Return the current kind of orthogonal polynomials.
        :setter: Set a new kind of orthogonal polynomials.
        :type: string

        """
        return self._kind

    @kind.setter
    def kind(self, value):
        """Set a new kind of orthogonal polynomials."""
        self._kind = self._validate_kind(value)

    @staticmethod
    def _coefs_array_to_dict(coefs_array, degrees):
        """Split array of coefs into dict mapping symbols to coef arrays."""
        precondition = coefs_array.size == sum(degrees.values()) + len(degrees)
        assert precondition, "The coefs array must conform with degree list!"

        coefs_dict = {}
        for var, degree in degrees.iteritems():
            coefs_dict[var] = coefs_array[:degree+1]
            coefs_array = coefs_array[degree+1:]

        postcondition = len(coefs_dict) == len(degrees)
        assert postcondition, "Length of coefs and degree lists must be equal!"

        return coefs_dict

    def _coefs_dict_to_array(self, coefs_dict):
        """Cast dict mapping symbol to coef arrays into array of coefs."""
        coefs_list = []
        for var in self.model.dependent_vars:
            coef_array = coefs_dict[var]
            coefs_list.append(coef_array)
        return np.hstack(coefs_list)

    def _collocation_nodes(self, degrees):
        """Return roots of suitable basis polynomial as collocation nodes."""
        basis_polys = self._construct_basis_polys(degrees)
        nodes = {}
        for var in self.model.dependent_vars:
            poly = basis_polys[var]
            nodes[var] = poly.roots()
        return nodes

    def _evaluate_functions(self, funcs_dict, arrays_dict):
        """Return a list of functions evaluated at arrays."""
        evaluated_funcs = []
        for var in self.model.dependent_vars:
            evaluated_funcs.append(funcs_dict[var](arrays_dict[var]))
        return evaluated_funcs

    def _orthogonal_poly_factory(self, coef):
        """Returns an orthogonal polynomial given some coefficients."""
        if self.kind == "Chebyshev":
            poly = np.polynomial.Chebyshev(coef, self.domain)
        elif self.kind == "Hermite":
            poly = np.polynomial.Hermite(coef, self.domain)
        elif self.kind == "Laguerre":
            poly = np.polynomial.Laguerre(coef, self.domain)
        else:
            poly = np.polynomial.Legendre(coef, self.domain)
        return poly

    def _residual_function_factory(self, symbol):
        """Generate the residual function for a given variable."""
        # extract left and right hand sides of the residual function
        lhs = self._orthogonal_poly_derivs[symbol]
        rhs = self._rhs_functions(symbol)

        # extract polys (ordering of this list is important!)
        polys = [self._orthogonal_polys[var] for var in self.model.dependent_vars]

        def residual_function(t):
            """Residual function evaluated at array of points t."""
            evaluated_polys = [poly(t) for poly in polys]
            params = self.params.values()
            args = evaluated_polys + params
            return lhs(t) - rhs(t, *args)

        return residual_function

    @staticmethod
    def _validate_domain(domain):
        """Validate the domain attribute."""
        lower, upper = domain
        if not isinstance(domain, list):
            mesg = ("Attribute 'domain' must have type list, not {}.")
            raise AttributeError(mesg.format(domain.__class__))
        elif not (lower < upper):
            mesg = "Lower bound of the domain must be less than upper bound."
            raise AttributeError(mesg)
        else:
            return domain

    @staticmethod
    def _validate_kind(kind):
        """Validate the kind attribute."""
        valid_kinds = ['Chebyshev', 'Hermite', 'Legendre', 'Laguerre']
        if not isinstance(kind, str):
            mesg = ("Attribute 'kind' must have type str, not {}.")
            raise AttributeError(mesg.format(kind.__class__))
        elif kind not in valid_kinds:
            mesg = "Attribute 'kind' must be one of {}, {}, {}, or {}."
            raise AttributeError(mesg.format(*valid_kinds))
        else:
            return kind


class OrthogonalCollocationSolver(OrthogonalCollocation):

    def solve(self, coefs_dict, method="hybr", **kwargs):
        degrees = self._infer_degrees(coefs_dict)
        initial_guess = self._coefs_dict_to_array(coefs_dict)
        result = optimize.root(self._evaluate_collocation_residuals,
                               x0=initial_guess,
                               args=(degrees,),
                               method=method,
                               **kwargs)
        return result

    def _boundary_condition_factory(self, symbol):
        """Generate the boundary condition for a given symbol."""
        raise NotImplementedError

    def _construct_basis_polys(self, degrees):
        """Return a list of suitable basis polynomial of a given degree."""
        basis_polys = {}
        for symbol, degree in degrees.iteritems():
            tmp_coef = self._basis_coef(degree)
            basis_poly = self._orthogonal_poly_factory(tmp_coef)
            basis_polys[symbol] = basis_poly

        postcondition = len(basis_polys) == len(degrees)
        assert postcondition, "Length of polys and degree lists must be equal!"

        return basis_polys

    def _evaluate_collocation_residuals(self, coefs_array, degrees):
        # update the dictionary of polynomial coefs
        self._coefs = self._coefs_array_to_dict(coefs_array, degrees)

        # get collocation nodes
        nodes = self._collocation_nodes(degrees)

        # evaluate the residual and boundary functions at collocation nodes
        residuals = self._evaluate_functions(self._residual_funcs, nodes)

        return residuals

    def _infer_degrees(self, coefs_dict):
        """Return dict mapping a symbol to degree of its approximating poly."""
        degrees = {}
        for var in self.model.dependent_vars:
            coef_array = coefs_dict[var]
            degrees[var] = coef_array.size - 1
        return degrees


if __name__ == '__main__':
    import sympy as sym
    import models

    # define some variables
    t, k, c = sym.symbols('t, k, c')

    # define some parameters
    alpha, sigma = sym.symbols('alpha, sigma')
    g, n, s, delta = sym.symbols('g, n, s, delta')

    # intensive output has the CES form
    y = (alpha * k**((sigma - 1) / sigma) + (1 - alpha))**(sigma / (sigma - 1))

    # define the Solow model
    k_dot = s * y - (g + n + delta) * k

    solow = models.DifferentialEquation(dependent_vars=[k],
                                        independent_var=t,
                                        rhs={k: k_dot})

    solow_params = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.15,
                    'sigma': 2.0, 'delta': 0.04}
    solow_solver = OrthogonalCollocationSolver(solow, solow_params)

    # define the Ramsey-Cass-Coopmans model
    rho, theta = sym.symbols('rho, theta')

    mpk = sym.diff(y, k, 1)
    k_dot = y - c - (g + n) * k
    c_dot = ((mpk - rho - theta * g) / theta) * c

    ramsey = models.DifferentialEquation(dependent_vars=[k, c],
                                         independent_var=t,
                                         rhs={k: k_dot, c: c_dot})

    ramsey_params = {'g': 0.02, 'n': 0.02, 's': 0.15, 'alpha': 0.15,
                     'sigma': 2.0, 'delta': 0.04, 'rho': 0.03, 'theta': 2.5}
    ramsey_solver = OrthogonalCollocationSolver(ramsey, ramsey_params)


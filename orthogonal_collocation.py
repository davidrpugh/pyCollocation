"""
TODO: GET RID OF SIDE EFFECTS!

"""
import numpy as np
from scipy import optimize

import solvers


class OrthogonalCollocation(solvers.Solver):
    """Base class for OrthogonalCollocation classes."""

    @property
    def _coefs(self):
        """Dictionary mapping a symbol to the coefs of its polynomial."""
        print str(self.__coefs) + "get"
        return self.__coefs

    @_coefs.setter
    def _coefs(self, coefs):
        """Set a new dictionary of polynomial coefs."""
        print str(coefs) + "set"
        self.__coefs = coefs

    @property
    def _lower_boundary_residual(self):
        if self._lower_boundary_condition is not None:
            evaluated_polys = self._evaluate_orthogonal_polys(self.domain[0])
            params = self.params.values()
            args = evaluated_polys + params
            return self._lower_boundary_condition(t, *args)
        else:
            return None

    @property
    def _upper_boundary_residual(self):
        if self._upper_boundary_condition is not None:
            evaluated_polys = self._evaluate_orthogonal_polys(self.domain[1])
            params = self.params.values()
            args = evaluated_polys + params
            return self._upper_boundary_condition(t, *args)
        else:
            return None

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

    def _evaluate_orthogonal_polys(self, t):
        """Evaluate approximating polynomials at an array of points t."""
        evaluated_polys = []
        for var in self.model.dependent_vars:
            evaluated_polys.append(self._orthogonal_polys[var](t))
        return evaluated_polys

    def _evaluate_residual_funcs(self, residuals, nodes):
        """Return residual function at some collocation nodes."""
        evaluated_funcs = []
        for var in self.model.dependent_vars:
            evaluated_funcs.append(residuals[var](nodes[var]))
        return np.hstack(evaluated_funcs)

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

        def residual_function(t):
            """Residual function evaluated at array of points t."""
            evaluated_polys = self._evaluate_orthogonal_polys(t)
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

    @staticmethod
    def _basis_coef(degree):
        """Return coefficients for the basis polynomial of a given degree."""
        basis_coef = np.zeros(degree + 1)
        basis_coef[-1] = 1
        return basis_coef

    def _collocation_nodes(self, degrees):
        """Return roots of suitable basis polynomial as collocation nodes."""
        basis_polys = self._construct_basis_polys(degrees)
        nodes = {}
        for var in self.model.dependent_vars:
            poly = basis_polys[var]
            nodes[var] = poly.roots()
        return nodes

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
        """Return residuals given coefs and degrees for approximating polys."""
        # set the new values for polynomial coefs
        self._coefs = self._coefs_array_to_dict(coefs_array, degrees)

        # evaluate the residual funcs at collocation nodes
        nodes = self._collocation_nodes(degrees)
        resids = [self._evaluate_residual_funcs(self._residual_funcs, nodes)]

        # add the boundary condition residuals
        if self._lower_boundary_residual is not None:
            resids += self._lower_boundary_residual
        if self._upper_boundary_residual is not None:
            resids += self._upper_boundary_residual

        return np.hstack(resids)

    def _infer_degrees(self, coefs_dict):
        """Return dict mapping a symbol to degree of its approximating poly."""
        degrees = {}
        for var in self.model.dependent_vars:
            coef_array = coefs_dict[var]
            degrees[var] = coef_array.size - 1
        return degrees

    def solve(self, coefs_dict, method="hybr", **kwargs):
        """Solve a boundary value problem using orthogonal collocation."""
        degrees = self._infer_degrees(coefs_dict)
        initial_guess = self._coefs_dict_to_array(coefs_dict)
        result = optimize.root(self._evaluate_collocation_residuals,
                               x0=initial_guess,
                               args=(degrees,),
                               method=method,
                               **kwargs)
        return result


if __name__ == '__main__':
    import sympy as sym
    import models

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
    rhs = {k: k_dot}

    # specify some boundary conditions
    k0 = 1.0
    bcs = {'lower': [k - k0], 'upper': None}  # hopefully this will work!

    solow = models.BoundaryValueProblem(dependent_vars=[k],
                                        independent_var=t,
                                        rhs=rhs,
                                        boundary_conditions=bcs)

    solow_params = {'g': 0.02, 's': 0.1, 'n': 0.02, 'alpha': 0.15, 'sigma': 2.0,
                    'delta': 0.04}
    solow_solver = OrthogonalCollocationSolver(solow, solow_params)
    solow_solver.kind = "Chebyshev"
    solow_solver.domain = [0, 100]

    # specify an initial guess
    def kstar(g, n, s, alpha, delta, sigma):
        rho = (sigma - 1) / sigma
        return ((1 - alpha) / (((g + n + delta) / s)**rho - alpha))**(1 / rho)
    xs = np.linspace(0, 100, 1000)
    ys = kstar(**solow_params) - (kstar(**solow_params) - k0) * np.exp(-xs)
    initial_poly = np.polynomial.Chebyshev.fit(xs, ys, 15, solow_solver.domain)
    initial_solow_coefs = {k: initial_poly.coef}
    solow_result = solow_solver.solve(initial_solow_coefs)

    # define the Ramsey-Cass-Coopmans model
    mpk = sym.diff(y, k, 1)
    k_dot = y - c - (g + n) * k
    c_dot = ((mpk - delta - rho - theta * g) / theta) * c
    rhs = {k: k_dot, c: c_dot}

    kstar = ((1 - alpha)**((sigma - 1) / sigma) /
             ((alpha / (delta + rho + theta * g))**(1 - sigma) - (alpha / (1 - alpha)))**((sigma - 1) / sigma))
    cstar = y.subs({k: kstar}) - (g + n) * kstar
    bcs = {'lower': [k - 5.0], 'upper': [c - cstar]}

    ramsey = models.BoundaryValueProblem(dependent_vars=[k, c],
                                         independent_var=t,
                                         rhs=rhs,
                                         boundary_conditions=bcs)

    ramsey_params = {'g': 0.02, 's': 0.1, 'n': 0.02, 'alpha': 0.15, 'sigma': 2.0,
                     'theta': 2.05, 'rho': 0.03, 'delta': 0.04}

    ramsey_solver = OrthogonalCollocationSolver(ramsey, ramsey_params)

    # specify kind of polynomials to use and the spproximation domain
    ramsey_solver.kind = "Chebyshev"
    ramsey_solver.domain = [0, 100]

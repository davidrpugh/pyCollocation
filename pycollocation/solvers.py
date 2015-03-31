import numpy as np

from . import boundary_value_problems


class Solver(object):
    """Base class for all Solvers."""

    def __init__(self, model):
        """
        Create an instance of the Solver class.

        Parameters
        ----------
        model : models.Model
            An instance of models.Model to solve.

        """
        self.model = model

    @property
    def coefficients(self):
        """
        Coefficients to use when constructing the approximating polynomials.

        :getter: Return the `coefficients` attribute.
        :type: dict

        """
        return self._coefs_array_to_dict(self.result.x, self.degrees)

    @property
    def derivatives(self):
        """
        Derivatives of the approximating basis functions.

        :getter: Return the `derivatives` attribute.
        :type: dict

        """
        return self._construct_basis_derivs(self.coefficients, self.kind, self.domain)

    @property
    def functions(self):
        """
        The basis functions used to approximate the solution to the model.

        :getter: Return the `functions` attribute.
        :type: dict

        """
        return self._construct_basis_funcs(self.coefficients, self.kind, self.domain)

    @property
    def model(self):
        """
        Symbolic representation of the model to solve.

        :getter: Return the current model.
        :setter: Set a new model to solve.
        :type: models.Model

        """
        return self._model

    @model.setter
    def model(self, model):
        """Set a new model to solve."""
        self._model = self._validate_model(model)

    @property
    def result(self):
        """
        Result object

        :getter: Return the current result object.
        :type: optimize.Result

        """
        return self._result

    @property
    def residual_functions(self):
        """
        Residual functions

        :getter: Return the current residual functions.

        """
        return self._construct_residual_funcs(self.derivatives, self.functions)

    @classmethod
    def _basis_derivative_factory(cls, *args, **kwargs):
        """Factory method for constructing derivatives of basis functions."""
        raise NotImplementedError

    @classmethod
    def _basis_function_factory(cls, *args, **kwargs):
        """Factory method for constructing basis functions."""
        raise NotImplementedError

    @staticmethod
    def _coefs_array_to_dict(coefs_array, degrees):
        """Split array of coefs into dict mapping symbols to coef arrays."""
        precondition = coefs_array.size == sum(degrees.values()) + len(degrees)
        assert precondition, "The coefs array must conform with degree list!"

        coefs_dict = {}
        for var, degree in degrees.items():
            coefs_dict[var] = coefs_array[:degree+1]
            coefs_array = coefs_array[degree+1:]

        postcondition = len(coefs_dict) == len(degrees)
        assert postcondition, "Length of coefs and degree lists must be equal!"

        return coefs_dict

    def _coefs_dict_to_array(self, coefs_dict):
        """Cast dict mapping symbol to coef arrays into array of coefs."""
        coefs_list = []
        for var in coefs_dict.keys():
            coef_array = coefs_dict[var]
            coefs_list.append(coef_array)
        return np.hstack(coefs_list)

    def _construct_basis_funcs(self, coefs, *args, **kwargs):
        """Return dict of basis functions given coefficients."""
        basis_funcs = {}
        for var, coef in coefs.items():
            basis_funcs[var] = self._basis_function_factory(coef, *args, **kwargs)
        return basis_funcs

    def _construct_basis_derivs(self, coefs, *args, **kwargs):
        """Return dict of basis function derivatives given coefficients."""
        basis_derivs = {}
        for var, coef in coefs.items():
            basis_derivs[var] = self._basis_derivative_factory(coef, *args, **kwargs)
        return basis_derivs

    def _construct_residual_funcs(self, basis_derivs, basis_funcs):
        """Return dict of residual functions for given basis funcs and derivs."""
        residual_funcs = {}
        for var, basis_deriv in basis_derivs.items():
            residual_funcs[var] = self._residual_function_factory(var, basis_derivs, basis_funcs)
        return residual_funcs

    def _evaluate_basis_funcs(self, basis_funcs, points):
        """Return a list of basis functions evaluated at some points."""
        return [basis_funcs[var](points) for var in self.model.dependent_vars]

    def _evaluate_boundary_residuals(self, basis_funcs, domain):
        """Return a list of boundary conditions evaluated on the domain."""
        lower_residual = self._evaluate_lower_boundary_residual(basis_funcs, domain[0])
        upper_residual = self._evaluate_upper_boundary_residual(basis_funcs, domain[1])

        residuals = []
        if lower_residual is not None:
            residuals.append(lower_residual)
        if upper_residual is not None:
            residuals.append(upper_residual)

        return residuals

    def _evaluate_lower_boundary_residual(self, basis_funcs, lower_bound):
        """Return the lower boundary condition evaluated on the domain."""
        if self.model._lower_boundary_condition is not None:
            args = (self._evaluate_basis_funcs(basis_funcs, lower_bound) +
                    list(self.model.params.values()))
            return self.model._lower_boundary_condition(lower_bound, *args)

    def _evaluate_residual_funcs(self, residual_funcs, nodes):
        """Return a list of residual functions evaluated at collocation nodes."""
        return [residual_funcs[var](nodes[var]) for var in self.model.dependent_vars]

    def _evaluate_upper_boundary_residual(self, basis_funcs, upper_bound):
        """Return the upper boundary condition evaluated on the domain."""
        if self.model._upper_boundary_condition is not None:
            args = (self._evaluate_basis_funcs(basis_funcs, upper_bound) +
                    list(self.model.params.values()))
            return self.model._upper_boundary_condition(upper_bound, *args)

    def _residual_function_factory(self, var, basis_derivs, basis_funcs):
        """Generate the residual function for a given variable."""

        def residual_function(t):
            """Residual function evaluated at array of points t."""
            args = (self._evaluate_basis_funcs(basis_funcs, t) +
                    list(self.model.params.values()))
            return basis_derivs[var](t) - self.model._rhs_functions(var)(t, *args)

        return residual_function

    @staticmethod
    def _validate_model(model):
        """Validate the dictionary of parameters."""
        if not isinstance(model, boundary_value_problems.BoundaryValueProblem):
            mesg = "Attribute 'model' must have type BoundaryValueProblem, not {}"
            raise AttributeError(mesg.format(model.__class__))
        else:
            return model

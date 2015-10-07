"""
Objects imported here will live in the `pycollocation` namespace

"""
from pkg_resources import get_distribution

from . import basis_functions
from . import problems
from . import solvers

__all__ = ["basis_functions", "problems", "solvers"]
__version__ = get_distribution('pyCollocation').version

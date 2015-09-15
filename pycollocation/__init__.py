"""
Objects imported here will live in the `pycollocation` namespace

"""
from pkg_resources import get_distribution

from . import problems
from . import solvers

__all__ = ["problems", "solvers"]
__version__ = get_distribution('pyCollocation').version

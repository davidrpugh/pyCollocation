import nose

from .. import symbolics


def test_symbolic_base():
    """SymbolicsBase._symbolic_args attribute is not implemented."""
    with nose.tools.assert_raises(ValueError):
        base = symbolics.SymbolicBase()
        base._symbolic_args

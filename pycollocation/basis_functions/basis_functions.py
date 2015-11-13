class BasisFunctionLike(object):

    @classmethod
    def derivatives_factory(cls, coef, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def functions_factory(cls, coef, *args, **kwargs):
        raise NotImplementedError

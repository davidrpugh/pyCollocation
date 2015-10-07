class BasisFunctionLike(object):

    @classmethod
    def derivatives_factory(cls, coef, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def functions_factory(cls, coef, domain, kind, **kwargs):
        raise NotImplementedError

    @classmethod
    def nodes(cls, *args, **kwargs):
        raise NotImplementedError

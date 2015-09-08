import collections


SolutionBase = collections.namedtuple("SolutionBase",
                                      field_names=['basis_kwargs',
                                                   'domain',
                                                   'functions',
                                                   'nodes',
                                                   'problem',
                                                   'residuals',
                                                   'result',
                                                   ],
                                      )


class Solution(SolutionBase):
    pass

from setuptools import setup

import pycollocation

setup(
    name="pyCollocation",
    version=pycollocation.__version__,
    license="MIT License",
    author="davidrpugh",
    install_requires=["numpy",
                      "scipy",
                      "sympy",
                      "pandas",
                      ],
    author_email="david.pugh@maths.ox.ac.uk",
    classifiers=["Programming Language :: Python",
                 ]
    )

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
    classifiers=['Development Status :: 1 - Planning',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 ]
    )

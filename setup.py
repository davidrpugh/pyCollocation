from setuptools import setup


# Meta information
DESCRIPTION = ("Python package for solving initial value problems (IVP) " +
               "and two-point boundary value problems (2PBVP) using the " +
               "collocation method ")

CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Intended Audience :: Education',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.3',
               'Programming Language :: Python :: 3.4',
               'Topic :: Scientific/Engineering',
               ]

setup(
    name="pyCollocation",
    version='0.2.2-alpha',
    license="MIT License",
    author="davidrpugh",
    install_requires=["numpy",
                      "scipy",
                      "sympy",
                      "pandas",
                      "ipython",
                      ],
    description=DESCRIPTION,
    author_email="david.pugh@maths.ox.ac.uk",
    classifiers=CLASSIFIERS,
    url='https://github.com/davidrpugh/pyCollocation',
    )

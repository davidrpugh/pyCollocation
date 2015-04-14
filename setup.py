import os

from distutils.core import setup


def read(*paths):
    """Build a file path from *paths* and return the contents."""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

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
    name="pycollocation",
    packages=['pycollocation'],
    version='0.3.0-alpha',
    description=DESCRIPTION,
    long_description=read('README.rst'),
    license="MIT License",
    author="davidrpugh",
    author_email="david.pugh@maths.ox.ac.uk",
    url='https://github.com/davidrpugh/pyCollocation',
    classifiers=CLASSIFIERS,
    )

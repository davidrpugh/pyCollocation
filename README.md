# pyCollocation

[![Build Status](https://travis-ci.org/davidrpugh/pyCollocation.svg?branch=master)](https://travis-ci.org/davidrpugh/pyCollocation)
[![Coverage Status](https://coveralls.io/repos/davidrpugh/pyCollocation/badge.svg?branch=master)](https://coveralls.io/r/davidrpugh/pyCollocation?branch=master)
[![Development Status](https://pypip.in/status/pyCollocation/badge.svg)](https://pypi.python.org/pypi/pyCollocation/)
[![Latest Version](https://pypip.in/version/pyCollocation/badge.svg)](https://pypi.python.org/pypi/pyCollocation/)
[![Downloads](https://pypip.in/download/pyCollocation/badge.svg)](https://pypi.python.org/pypi/pyCollocation/)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.16932.svg)](http://dx.doi.org/10.5281/zenodo.16932)

Python package for solving initial value problems (IVP) and two-point boundary value problems (2PBVP) using the collocation method with various basis functions. Currently I have implemented the following basis functions:

* Orthogonal polynomials: [Chebyshev](http://en.wikipedia.org/wiki/Chebyshev_polynomials), [Laguerre](http://en.wikipedia.org/wiki/Laguerre_polynomials), [Legendre](http://en.wikipedia.org/wiki/Legendre_polynomials), and [Hermite](http://en.wikipedia.org/wiki/Hermite_polynomials).


## Example notebooks

### Economics
There are a number of example notebooks that demonstrate how to use the library to solve seminal models in the economics literature.

* [Solow model of economic growth](http://nbviewer.ipython.org/github/davidrpugh/pyCollocation/blob/master/examples/solow-model.ipynb)
* [Ramsey-Cass-Koopmans model of optimal savings](http://nbviewer.ipython.org/github/davidrpugh/pyCollocation/blob/master/examples/ramsey-model.ipynb)
* [Spence model of costly signaling](http://nbviewer.ipython.org/github/davidrpugh/pyCollocation/blob/master/examples/spence-model.ipynb)
* [Credit cycles](http://nbviewer.ipython.org/github/davidrpugh/pyCollocation/blob/master/examples/credit-cycles.ipynb)

### Physics

* [A simple heat exchanger](http://nbviewer.ipython.org/github/davidrpugh/pyCollocation/blob/master/examples/heat-exchanger.ipynb)

More notebooks will be added in the near future (hopefully!)

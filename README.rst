pyCollocation
=============

|Build Status| |Coverage Status| |Latest Version| |Downloads| |DOI|

.. |Build Status| image:: https://travis-ci.org/davidrpugh/pyCollocation.svg?branch=master
   :target: https://travis-ci.org/davidrpugh/pyCollocation
.. |Coverage Status| image:: https://coveralls.io/repos/davidrpugh/pyCollocation/badge.svg?branch=master
   :target: https://coveralls.io/r/davidrpugh/pyCollocation?branch=master
.. |Latest Version| image:: https://img.shields.io/pypi/v/pyCollocation.svg
   :target: https://pypi.python.org/pypi/pyCollocation/
.. |Downloads| image:: https://img.shields.io/pypi/dm/pyCollocation.svg
   :target: https://pypi.python.org/pypi/pyCollocation/
.. |DOI| image:: https://zenodo.org/badge/doi/10.5281/zenodo.17283.svg
   :target: http://dx.doi.org/10.5281/zenodo.17283

Python package for solving initial value problems (IVP) and two-point boundary value problems (2PBVP) using the collocation method with various basis functions. Currently I have implemented the following basis functions:

- Orthogonal polynomials: Chebyshev_, Laguerre_, Legendre_, and Hermite_.

.. _Chebyshev: http://en.wikipedia.org/wiki/Chebyshev_polynomials
.. _Laguerre: http://en.wikipedia.org/wiki/Laguerre_polynomials
.. _Legendre: http://en.wikipedia.org/wiki/Legendre_polynomials
.. _Hermite: http://en.wikipedia.org/wiki/Hermite_polynomials

Installation
------------

Assuming you have `pip`_ on your computer (as will be the case if you've `installed Anaconda`_) you can install the latest stable release of ``pycollocation`` by typing
    
.. code:: bash

    pip install pycollocation

at a terminal prompt.

.. _pip: https://pypi.python.org/pypi/pip
.. _`installed Anaconda`: http://quant-econ.net/getting_started.html#installing-anaconda

Example notebooks
-----------------

Economics
~~~~~~~~~

There are a number of example notebooks that demonstrate how to use the library to solve seminal models in the economics literature.

- `Solow model of economic growth`_
- `Ramsey model of optimal savings`_
- `Spence model of costly signaling`_
- `Kiyotaki and Moore model of credit cycles`_

.. _`Solow model of economic growth`: http://nbviewer.ipython.org/github/davidrpugh/pyCollocation/blob/master/examples/solow-model.ipynb
.. _`Ramsey model of optimal savings`: http://nbviewer.ipython.org/github/ramseyPy/ramseyPy/blob/master/examples/ramsey-model.ipynb
.. _`Spence model of costly signaling`: http://nbviewer.ipython.org/github/davidrpugh/pyCollocation/blob/master/examples/spence-model.ipynb
.. _`Kiyotaki and Moore model of credit cycles`: http://nbviewer.ipython.org/github/davidrpugh/pyCollocation/blob/master/examples/credit-cycles.ipynb

Physics
~~~~~~~

- `A simple heat exchanger`_ 

.. _`A simple heat exchanger`: http://nbviewer.ipython.org/github/davidrpugh/pyCollocation/blob/master/examples/heat-exchanger.ipynb

More notebooks will be added in the near future (hopefully!)

pyCollocation
=============

|Build Status| |Coverage Status| |Documentation Status| |Code Climate| |Latest Version| |Downloads| |DOI|

.. |Build Status| image:: https://travis-ci.org/davidrpugh/pyCollocation.svg?branch=master
   :target: https://travis-ci.org/davidrpugh/pyCollocation
.. |Coverage Status| image:: https://coveralls.io/repos/davidrpugh/pyCollocation/badge.svg?branch=master
   :target: https://coveralls.io/r/davidrpugh/pyCollocation?branch=master
.. |Code Climate| image:: https://codeclimate.com/github/davidrpugh/pyCollocation/badges/gpa.svg
   :target: https://codeclimate.com/github/davidrpugh/pyCollocation
.. |Latest Version| image:: https://img.shields.io/pypi/v/pyCollocation.svg
   :target: https://pypi.python.org/pypi/pyCollocation/
.. |Downloads| image:: https://img.shields.io/pypi/dm/pyCollocation.svg
   :target: https://pypi.python.org/pypi/pyCollocation/
.. |DOI| image:: https://zenodo.org/badge/doi/10.5281/zenodo.31910.svg
   :target: http://dx.doi.org/10.5281/zenodo.31910
.. |Documentation Status| image:: https://readthedocs.org/projects/pycollocation/badge/?version=latest
   :target: https://readthedocs.org/projects/pycollocation/?badge=latest


Python package for solving initial value problems (IVP) and two-point boundary value problems (2PBVP) using the collocation method with various basis functions. Currently I have implemented the following basis functions:

- Polynomials: Standard_, Chebyshev_, Laguerre_, Legendre_, and Hermite_.

.. _Standard: https://en.wikipedia.org/wiki/Polynomial
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

- `Solow model`_ of economic growth
- `Ramsey model`_ of optimal savings
- Various `auction models`_ (currently symmetric and asymmetric IPVP)

.. _`Solow model` : https://github.com/davidrpugh/pyCollocation/blob/master/examples/solow-model.ipynb
.. _`Ramsey model`: https://github.com/davidrpugh/pyCollocation/blob/master/examples/ramsey-cass-koopmans-model.ipynb
.. _`auction models` : https://github.com/davidrpugh/pyCollocation/blob/master/examples/auction-models.ipynb


Physics
~~~~~~~

- `A simple heat exchanger`_ 

.. _`A simple heat exchanger`: https://github.com/davidrpugh/pyCollocation/blob/master/examples/heat-exchanger.ipynb

More notebooks will be added in the near future (hopefully!)...and suggestions for example notebooks are very welcome!

Roadmap to 1.0
--------------
Ultimately I am hoping to contribute this package to either SciPy or QuantEcon, depending.  Ideally, version 1.0 of pyCollocation would include the following functionality...

- Support for at least two additional classes of basis functions: B-Splines, what else? Next obvious candidate would be some basis functions specifically used to approximate periodic functions.

- Support a solver for over-identified systems of equations.  Currently the Solver class requires the system of equations defined by the collocation residuals to be exactly identified.  Many economic applications, particularly models of optimal bidding functions for various types of auctions, naturally lead to over-identified systems.

- Built-in support for computing Jacobian matrix for the system of equations defined by the collocation residuals.  Given a user-supplied Jacobian for the BVP, one can apply the chain rule to construct the Jacobian matrix for system of equations defined by the collocation residuals.

- Support for solving models with unknown parameters (similar to `scikits.bvp_solver`_). This would allow for the possibility to simultaneously solve and calibrate a model.

- Support for free boundary conditions.  This comes up a lot in auction theory applications where the upper bounds on the bidder valuation distributions are unknown.

.. _`scikits.bvp_solver` : https://github.com/jsalvatier/scikits.bvp_solver 

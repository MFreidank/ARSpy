========
ARSPY
========

|Build Status|
|Docs_|
|Coverage_|
|Pypi_|

This package provides a pure python/numpy 
implementation of adaptive rejection sampling as 
proposed by P. Wild, W.R. Gilks in 
*Algorithm AS 287: Adaptive Rejection Sampling from Log Concave Density 
functions*.

Under the (frequently satisfied) assumption that the target distribution to 
sample from has a log-concave density function, this algorithm allows us 
to sample without calculating any integrals. 

This sampling method is *exact* (all resulting samples are i.i.d) and 
*efficient* and our implementation can handle any univariate log-concave 
distribution. 

One prime use case is Gibbs sampling, where one frequently encounters many 
1D log-concave distributions.

Install
=======

Simply run::

   pip3 install ARSpy

.. |Build Status| image:: https://travis-ci.org/MFreidank/ARSpy.svg?branch=master
    :target: https://travis-ci.org/MFreidank/ARSpy

.. |Coverage_| image:: https://coveralls.io/repos/github/MFreidank/pyARS/badge.svg
   :target: https://coveralls.io/github/MFreidank/pyARS
   :alt: Coverage

.. |Docs_| image:: https://readthedocs.org/projects/ARSpy/badge/?version=latest
   :target: http://ARSpy.readthedocs.io/en/latest/
   :alt: Docs

.. |Pypi_| image:: https://badge.fury.io/py/ARSpy.svg
    :target: https://badge.fury.io/py/ARSpy

Documentation
=============
Our documentation can be found at http://arspy.readthedocs.io/en/latest/.

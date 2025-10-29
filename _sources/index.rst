.. stream_sim documentation master file, created by
   sphinx-quickstart on Wed Oct 29 11:50:35 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Stream_Sim's documentation!
===================================

.. warning::
   This package is under active development. The API and features may change in future releases.

Stream_Sim is a Python package designed for stellar stream data generation. It provides essential tools for generate mock stellar stream data, and convert them into observable quantities.


Key Features
------------

- Generate mock stellar stream data based on chosen parametric models, including:
   
   - Positions (phi1, phi2) / (ra, dec)
   - Distances
   - magnitudes in various photometric bands
- Convert data mock into observed quantity by a chosen survey (eg. LSST)


Example Usage
-------------

This package can be used to convert the results of dynamic simulations of stellar streams as they would be observed by a survey such as LSST.
 

Installation
------------

To install the package and its dependencies, see the :doc:`installation` page.

Citation
--------

If you use this package in your work, please see the :doc:`citation` page for citation information.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   about
   quickstart
   installation
   citation
   modules
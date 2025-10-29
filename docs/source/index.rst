.. stream_sim documentation master file

Welcome to StreamSim's Documentation!
======================================

.. warning::
   This package is under active development. The API and features may change in future releases.

**StreamSim** is a Python package for stellar stream data generation and observation simulation. 
It provides a complete pipeline to transform theoretical or dynamical stream models into realistic 
mock observations as they would appear in astronomical surveys.

What StreamSim Does
-------------------

StreamSim enables you to:

1. **Generate mock stellar stream data** from parametric models
   
   - Create positions of stars along streams (phi1, phi2 or RA, Dec)
   - Model various stream morphologies (linear, sinusoidal, spline-based)
   - Assign distances and velocities to stream stars

2. **Assign photometric properties** using stellar isochrones
   
   - Generate magnitudes in multiple photometric bands (eg. g and r)

3. **Simulate realistic survey observations**
   
   - Apply survey footprint and selection functions
   - Include Galactic extinction effects
   - Add photometric errors and detection completeness
   - Simulate observational uncertainties

Use Cases
---------

StreamSim is particularly useful for people who:

- Run **dynamical simulations** and need to convert results into observable quantities
- Develop **stream detection algorithms** and require realistic test data
- Plan **survey strategies** for detecting stellar streams
- Need to generate **mock catalogs** for pipeline validation

.. note::
   StreamSim focuses on parametric stream models and observation simulation. 
   It is **not** designed for N-body simulations or dynamical modeling.


Getting Started
---------------

New to StreamSim? Start here:

1. **Installation**: :doc:`installation` - Set up StreamSim on your system
2. **About**: :doc:`about` - Understand StreamSim's purpose and design
3. **Quickstart**: :doc:`quickstart` - Learn the basics with simple examples
4. **API Reference**: :doc:`modules` - Explore the complete API documentation

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   about
   installation
   quickstart
   citation

.. toctree::
   :maxdepth: 2
   :caption: Content documentation
   
   modules

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

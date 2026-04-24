face-rhythm documentation
=========================

**face-rhythm** is a Python package for extracting and decomposing rhythmic
facial movements from video. It combines point-based optic flow, spectral
analysis (variable-Q transforms), and tensor component analysis to turn raw
behavior videos into compact, interpretable factors capturing spatial and
spectral structure.

The pipeline is designed to be:

- *stable*, needing no pre-trained models and few sensitive hyperparameters;
- *spectrally aware*, exposing rhythmic content that frame-difference or
  keypoint methods typically ignore;
- *session-scalable*, with alignment utilities for comparing across sessions.

See the linked sections below to install the package, get a pipeline running
on a single video, and explore the API reference.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   standards
   organization
   notebooks
   support

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api

.. toctree::
   :maxdepth: 1
   :caption: About

   citation


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

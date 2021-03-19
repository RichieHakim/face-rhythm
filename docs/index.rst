face-rhythm documentation!
==============================================

While many tools exist to transform behavioral videos into useful low-dimensional representations,
very few both stay stable to small alterations in the recording conditions and require no pre-training of large models.
Furthermore, few techniques capitalize on rich spectral data in creating factors. To tackle these existing restraints,
we propose Face Rhythm, a novel behavioral quantification pipeline combining optic flow, spectral analysis, and
tensor component analysis to robustly generate meaningful, low-dimensional factors capturing both spectral and
spatial patterns in behavioral videos.

User Guide
-------------------

.. toctree::

   installation
   standards
   organization
   support

Programmer Reference
--------------------

.. toctree::

   face_rhythm.util
   face_rhythm.optic_flow
   face_rhythm.analysis
   face_rhythm.visualize


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

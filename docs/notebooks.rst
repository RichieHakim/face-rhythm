Demo Notebooks
==============

Three annotated demo notebooks live in the ``notebooks/`` directory of the
source repository. They are the recommended way to learn the pipeline and
to adapt it to new data.

Clone the repository (see :doc:`installation`) and launch Jupyter from the
repository root to open them:

.. code-block:: console

   jupyter notebook notebooks/

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Notebook
     - What it covers
   * - ``demo_pipeline.ipynb``
     - End-to-end walkthrough of the basic pipeline: loading videos,
       drawing ROIs, running point tracking, spectral decomposition, and
       tensor component analysis.
   * - ``demo_set_rois_multisession.ipynb``
     - Interactively drawing and saving ROIs for a multi-session dataset
       where each session needs its own ROI file.
   * - ``demo_event_alignment.ipynb``
     - Aligning the extracted factors to external event timestamps (e.g.
       task events, stimulus onsets) for downstream analysis.

Links to the notebooks on GitHub:

* `demo_pipeline.ipynb <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_pipeline.ipynb>`_

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/RichieHakim/face-rhythm/blob/release/notebooks/demo_pipeline.ipynb
     :alt: Open In Colab

* `demo_set_rois_multisession.ipynb <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_set_rois_multisession.ipynb>`_

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/RichieHakim/face-rhythm/blob/release/notebooks/demo_set_rois_multisession.ipynb
     :alt: Open In Colab

* `demo_event_alignment.ipynb <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_event_alignment.ipynb>`_

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/RichieHakim/face-rhythm/blob/release/notebooks/demo_event_alignment.ipynb
     :alt: Open In Colab

.. note::

   If the notebook names above do not match the files in your checkout,
   you may be on an older branch. Pull ``release`` (or ``dev``) to get
   the current demo set.

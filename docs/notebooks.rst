Demo Notebooks
==============

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

* `demo_set_rois_multisession.ipynb <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_set_rois_multisession.ipynb>`_

* `demo_event_alignment.ipynb <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_event_alignment.ipynb>`_
Quick Start
===========

face-rhythm has three entry points, ordered from lowest friction to most
flexible:

1. Interactive notebook (recommended for new users)
---------------------------------------------------

The end-to-end demo runs a complete face-rhythm pipeline on a sample
recording in roughly 5 minutes:

* `demo_pipeline.ipynb <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_pipeline.ipynb>`_
  on GitHub

  .. image:: https://colab.research.google.com/assets/colab-badge.svg
     :target: https://colab.research.google.com/github/RichieHakim/face-rhythm/blob/release/notebooks/demo_pipeline.ipynb
     :alt: Open In Colab

Other notebooks:
`demo_set_rois_multisession <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_set_rois_multisession.ipynb>`_
for cross-session ROI alignment, and
`demo_event_alignment <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_event_alignment.ipynb>`_
for event-aligned trace analysis. See :doc:`notebooks` for the full list.

2. Command-line script
----------------------

For batch runs across many sessions:

.. code-block:: bash

   python scripts/run_pipeline_basic.py \
       --path_params params.json \
       --directory_save /path/to/project/

A ready-to-edit template lives at ``scripts/params_pipeline_basic.json``.

3. Python API
-------------

.. include:: ../README.md
   :start-after: <!-- start-quickstart -->
   :end-before: <!-- end-quickstart -->
   :parser: myst_parser.sphinx_

For the full parameter contract, see
:func:`face_rhythm.util.get_default_parameters`.

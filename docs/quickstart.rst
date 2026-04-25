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

.. code-block:: python

   import json
   import face_rhythm as fr

   with open("params_pipeline_basic.json", "r") as f:
       params = json.load(f)

   params["project"]["directory_project"] = "/path/to/new/project/"
   params["paths_videos"]["directory_videos"] = "/path/to/videos/"
   params["ROIs"]["initialize"]["path_file"] = "/path/to/ROIs.h5"

   results = fr.pipelines.pipeline_basic(params)

Copy ``scripts/params_pipeline_basic.json`` as a template, edit the three
paths, and run. Results land in the project directory as HDF5 files plus
summary plots.

For the full parameter contract, see
:func:`face_rhythm.util.get_default_parameters`.

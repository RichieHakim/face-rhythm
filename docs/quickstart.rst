Quick Start
===========

face-rhythm has three entry points:

Interactive notebooks
---------------------

- `demo_pipeline.ipynb <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_pipeline.ipynb>`_
  — end-to-end demo on a single session. Start here.
- `demo_set_rois_multisession.ipynb <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_set_rois_multisession.ipynb>`_
  — draw and align ROIs across multiple sessions of the same subject.
- `demo_event_alignment.ipynb <https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_event_alignment.ipynb>`_
  — align extracted factors to event timestamps and view trial-averaged
  traces.

See :doc:`notebooks` for the full list.

Command line
------------

For batch runs across many sessions:

.. code-block:: shell

   python scripts/run_pipeline_basic.py --path_params params.json --directory_save /path/to/project/

``scripts/params_pipeline_basic.json`` is a ready-to-edit template.

Python API
----------

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

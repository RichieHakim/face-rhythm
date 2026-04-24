Quick Start
===========

This page shows the minimum code needed to run the face-rhythm basic
pipeline end-to-end. The canonical, fully-worked walkthrough (including
the interactive ROI-drawing step) lives in
``notebooks/demo_pipeline.ipynb`` — see :doc:`notebooks`.

Run the basic pipeline from Python
----------------------------------

Build a parameters dictionary from the defaults, point it at your data,
then hand it to :func:`face_rhythm.pipelines.pipeline_basic`:

.. code-block:: python

   import face_rhythm as fr

   params = fr.util.get_default_parameters(
       directory_project="/path/to/output/project_dir",
       directory_videos="/path/to/videos",
       filename_videos_strMatch=r"\.mp4$",   # regex that videos must match
       path_ROIs=None,                       # set to a .h5 to skip the GUI step
   )

   fr.pipelines.pipeline_basic(params)

The pipeline writes every intermediate artefact (point trajectories,
spectral decompositions, TCA factors) and a ``params_used.json`` snapshot
into the project directory.

Run from the command line
-------------------------

A thin CLI wrapper lives in ``scripts/run_pipeline_basic.py`` (available
when you install from source — see :ref:`install-source`). Point it at a
JSON parameters file derived from ``scripts/params_pipeline_basic.json``:

.. code-block:: console

   python scripts/run_pipeline_basic.py \
       --path_params scripts/params_pipeline_basic.json \
       --directory_save /path/to/output/project_dir

The ``--directory_save`` flag is optional; if omitted, the pipeline uses
``directory_project`` from the JSON file.

Next steps
----------

- Read :doc:`standards` for recommended video acquisition settings.
- Read :doc:`organization` for how multi-session datasets should be laid
  out.
- Browse the :doc:`api` to see every module and function.

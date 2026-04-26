Pipeline Outputs and Concepts
=============================

This page documents the data structures and terminology used throughout
face-rhythm: how a session is organized, what a pipeline run produces, and
how to interpret the files written to disk.

Project layout on disk
----------------------

A face-rhythm project is a single directory created by
:func:`face_rhythm.project.prepare_project`. After a run completes, that
directory contains:

::

    <directory_project>/
    ├── config.yaml          # parameters used by every module
    ├── run_info.json        # lightweight metadata produced by every module
    ├── analysis_files/      # per-module HDF5 outputs (the data products)
    │   ├── Dataset_videos.h5
    │   ├── ROIs.h5
    │   ├── PointTracker.h5
    │   ├── VQT_Analyzer.h5
    │   └── TCA.h5
    └── visualizations/      # figures saved by Figure_Saver

Each pipeline module appends a section keyed by class name (for example,
``PointTracker``) to ``config.yaml`` and ``run_info.json``, and writes its
data arrays to the matching ``<ClassName>.h5`` file under ``analysis_files/``.

Run output dict
---------------

:func:`face_rhythm.pipelines.pipeline_basic` returns a small ``results``
dictionary that points at the artifacts on disk; the substantive arrays are
not held in memory. The keys are:

* ``path_config`` (``str``) — path to ``config.yaml``.
* ``path_run_info`` (``str``) — path to ``run_info.json``.
* ``directory_project`` (``str``) — root project directory.
* ``SEED`` (``int``) — random seed used for the run.
* ``params`` (``dict``) — the full parameter dictionary that was executed.

Pipeline stage outputs
----------------------

``Dataset_videos.h5`` — :mod:`face_rhythm.data_importing`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A handle on the raw video set used for the run.

* ``example_image`` — a single representative frame, used downstream for ROI
  drawing and visualization.

The accompanying ``run_info.json`` block records ``frame_rate``,
``num_frames_total``, ``frame_height_width``, ``num_channels``, and the
per-video ``metadata``.

``ROIs.h5`` — :mod:`face_rhythm.rois`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The face mask(s) and the seed grid of points to track.

* ``mask_images`` — boolean masks for each ROI.
* ``roi_points`` — vertex coordinates of each user-drawn ROI polygon.
* ``point_positions`` — ``(n_points, 2)`` array of ``(x, y)`` coordinates
  used as initial point locations.
* ``exampleImage`` — the frame the ROIs were drawn on.
* ``num_points`` — total number of tracked points.

``PointTracker.h5`` — :mod:`face_rhythm.point_tracking`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Output of optical-flow point tracking across all videos in the session.

* ``points_tracked`` — point-tracking time series dict keyed by video index (``"0"``, ``"1"``, ...);
  each value is an array of shape ``(n_frames, n_points, 2)``.
* ``violations`` — sparse COO arrays (``row``, ``col``, ``data``,
  ``shape``) per video flagging frames where displacements crossed the
  outlier threshold and points were frozen.
* ``point_positions`` — initial point grid (carried through from
  ``ROIs.h5``).
* ``neighbors``, ``mesh_d0``, ``mask`` — internal mesh structure used by
  the elastic point tracker.

``run_info`` adds ``num_videos``, per-video ``num_frames``,
``num_frames_total``, ``num_points``, and ``violation_fraction``.

``VQT_Analyzer.h5`` — :mod:`face_rhythm.spectral_analysis`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Variable-Q transform spectrograms of the per-point displacement traces.

* ``spectrograms`` — dict keyed by video index; each value has shape
  ``(2, n_points, n_freq_bins, n_frames_downsampled)``, where the leading
  axis is the ``(x, y)`` displacement direction.
* ``x_axis`` — frame indices corresponding to each spectrogram column
  (after temporal downsampling).
* ``frequencies`` — center frequencies of the VQT filter bank.
* ``VQT`` — the filter bank itself (``filters`` and ``wins``) used to
  compute the spectrograms.
* ``point_positions`` — point grid carried through from earlier stages.

``TCA.h5`` — :mod:`face_rhythm.decomposition`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tensor component analysis (CP-style) decomposition of the spectrogram
tensor.

* ``factors`` — the raw factor matrices, one per tensor dimension, in the
  same dimension order used during fitting.
* ``factors_rearranged`` — the factors after any concatenations along the
  ``xy``, complex, and dictionary-element axes have been undone, giving
  per-dimension factors in their natural shape.
* ``names_dims_array`` / ``names_dims_array_preDecomp`` — the axis names
  (e.g. ``['xy', 'points', 'frequency', 'time']``) corresponding to each
  factor, before and after rearrangement.
* ``name_dim_dictElements`` / ``name_dim_dictElements_preDecomp`` — the
  name of the dictionary axis (typically ``'trials'``) before and after
  rearrangement.

``run_info`` adds ``num_dictElements`` along with the dimension-name
metadata.

Cross-references
----------------

* :doc:`api` — full module reference.
* :doc:`standards` — video acquisition recommendations (frame rate, codec,
  exposure, occlusion handling).

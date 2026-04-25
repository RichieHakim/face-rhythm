Installation
============

If you have any issues during installation, please open a
`GitHub issue <https://github.com/RichieHakim/face-rhythm/issues>`_.

0. Requirements
---------------

- `Anaconda <https://www.anaconda.com/distribution/>`_ or
  `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (any
  modern install).

1. Install face-rhythm
----------------------

**Linux / macOS** (recommended - torchcodec backend, all distros):

.. code-block:: shell

   conda create -n face_rhythm -c conda-forge python=3.12 'torchcodec=*=cpu*' ffmpeg libstdcxx-ng
   conda activate face_rhythm
   pip install face-rhythm

On macOS, drop ``libstdcxx-ng`` from the command (it is Linux-only).

For NVDEC GPU video decoding, swap ``cpu*`` for ``cuda126*``,
``cuda129*``, or ``cuda130*`` to match your driver (Linux x86_64 only).

**Windows** (decord backend):

.. code-block:: shell

   conda create -n face_rhythm python=3.12
   conda activate face_rhythm
   pip install face-rhythm

torchcodec has no PyPI Windows wheels, so on Windows construct readers
with ``BufferedVideoReader(..., backend='decord')`` explicitly. The
torchcodec wrapper raises a helpful error pointing to this if you forget.

You will need to activate the environment with
``conda activate face_rhythm`` each time you want to use face-rhythm.

**Why conda-forge for torchcodec?** The conda-forge torchcodec package
bakes RPATH so FFmpeg is found automatically. The PyPI wheel does not,
which causes load errors in many conda environments. Conda-forge avoids
this entirely.

- **Headless servers** (no display, e.g. compute clusters): after the
  install above, run ``pip uninstall opencv_contrib_python`` and
  ``pip install opencv_contrib_python_headless``.

2. Clone the repo to get the notebooks
--------------------------------------

.. code-block:: shell

   git clone https://github.com/RichieHakim/face-rhythm.git

Then open the notebooks in ``face-rhythm/notebooks/``.

Troubleshooting Installation
----------------------------

The recipe at the top of this page avoids the common pitfalls by
construction. The notes below cover failure modes that still surface
on existing environments, unusual platforms, or GPU setups.

torchcodec FFmpeg / shared-library load errors
----------------------------------------------

Symptom: ``import torchcodec`` raises::

   OSError: libavutil.so.NN: cannot open shared object file: No such file or directory

Cause: the PyPI ``torchcodec`` wheel ships its native ``.so`` files with
**no RPATH or RUNPATH**, so the dynamic loader never looks in
``$CONDA_PREFIX/lib`` even when ``ffmpeg`` is installed via conda. The
conda-forge build bakes ``$ORIGIN/../../..`` into the RPATH of every
``libtorchcodec_core*.so``, which is the entire fix.

**Fix:** uninstall the PyPI wheel and install from conda-forge.

.. code-block:: bash

   pip uninstall -y torchcodec
   conda install -c conda-forge 'torchcodec=*=cpu*' ffmpeg libstdcxx-ng

(Drop ``libstdcxx-ng`` on macOS.)

.. note::

   ``LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`` is a workaround
   that lets the PyPI wheel find FFmpeg, but it is fragile (does not
   survive subprocess re-execs, easy to forget on cluster jobs) and on
   RHEL-family systems triggers a follow-on ``GLIBCXX_3.4.34 not found``
   error from the system libstdc++. Use the conda-forge package; tracked
   upstream at `pytorch/torchcodec#905
   <https://github.com/pytorch/torchcodec/issues/905>`_.

Headless OpenCV on servers
--------------------------

On compute nodes, SLURM jobs, Docker containers, and other display-free
hosts, ``import cv2`` may abort with a Qt, ``xcb``, or display-related
error. The default ``opencv_contrib_python`` wheel links against Qt and
expects an X11 display. Swap it for the ABI-compatible headless build:

.. code-block:: bash

   pip uninstall -y opencv_contrib_python
   pip install opencv_contrib_python_headless

Windows: torchcodec unavailable
-------------------------------

torchcodec has no Windows wheels on PyPI. ``face-rhythm``'s
``pyproject.toml`` declares the dependency with the marker
``torchcodec; sys_platform != 'win32'``, so ``pip install face-rhythm``
on Windows skips it cleanly — no install error.

You do need to construct ``BufferedVideoReader`` with the decord backend
explicitly, since the default resolves to torchcodec on Linux/macOS:

.. code-block:: python

   import face_rhythm as fr

   reader = fr.helpers.BufferedVideoReader(
       paths_videos=["video.mp4"],
       backend="decord",
   )

The torchcodec wrapper raises a helpful error pointing at this line if
you forget.

GPU NVDEC decoding
------------------

For NVDEC-accelerated video decoding on a CUDA GPU, install the matching
CUDA build of torchcodec from conda-forge instead of the CPU build:

.. code-block:: bash

   conda create -n face_rhythm -c conda-forge python=3.12 \
       'torchcodec=*=cuda130*' ffmpeg libstdcxx-ng
   conda activate face_rhythm
   pip install face-rhythm

Replace ``cuda130*`` with ``cuda126*`` or ``cuda129*`` to match your
NVIDIA driver / CUDA toolkit. Conda-forge ships CUDA torchcodec builds
only for ``linux-64``.

To activate NVDEC at runtime, pass ``device='cuda'`` (or
``device='cuda:0'``) when constructing a reader:

.. code-block:: python

   reader = fr.helpers.BufferedVideoReader(
       paths_videos=["video.mp4"],
       device="cuda",
   )

GPU decoding additionally requires that FFmpeg be built with
``--enable-cuda``; the conda-forge ``ffmpeg`` package satisfies this. See
the `NVIDIA Video Codec SDK
<https://developer.nvidia.com/video-codec-sdk>`_ for codec-support
details (NVDEC supports H.264, HEVC, VP9, and AV1 on recent GPUs).

Verifying the install
---------------------

Confirm that the package imports and that a video reader constructs
cleanly:

.. code-block:: python

   import face_rhythm as fr
   print(fr.__version__)

   reader = fr.helpers.BufferedVideoReader(
       paths_videos=["/path/to/any/video.mp4"],
   )
   print("frames:", len(reader), "shape:", reader[0].shape)

If you do not have a test video on hand, the repository ships a small
demo bundle at ``tests/data_test.zip``. Unzip it and point the reader at
one of the ``.mp4`` files inside:

.. code-block:: bash

   unzip tests/data_test.zip -d tests/data_test/

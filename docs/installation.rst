Installation
============

If you have issues during installation, please open a
`GitHub issue <https://github.com/RichieHakim/face-rhythm/issues>`_.

0. Requirements
---------------

- `Anaconda <https://www.anaconda.com/distribution/>`_ or
  `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.
- Run commands in Terminal (Linux/macOS) or Anaconda Prompt (Windows).
- Windows users should use the ``decord`` video backend.

1. Create a conda environment
-----------------------------

.. code-block:: shell

   conda create -n face_rhythm python=3.12
   conda activate face_rhythm
   python -m pip install --upgrade pip

Activate the env with ``conda activate face_rhythm`` each time you use
face-rhythm.

2. Install video packages
-------------------------

Linux
~~~~~

.. code-block:: shell

   conda install -c conda-forge 'torchcodec=*=cpu*' ffmpeg libstdcxx-ng

macOS
~~~~~

.. code-block:: shell

   conda install -c conda-forge 'torchcodec=*=cpu*' ffmpeg

Windows
~~~~~~~

Skip this step. The default ``pip install`` installs the decord backend. If you want GPU speedups, you can try ``conda install -c conda-forge 'torchcodec=*=cpu*' ffmpeg``, but Windows torchcodec support is not first-class.

3. Install face-rhythm
----------------------

.. code-block:: shell

   pip install face-rhythm

For headless servers, see :ref:`installation-troubleshooting`. For GPU
acceleration, finish the CPU install first, then see
:ref:`gpu-acceleration`.

4. Clone the repo to get the notebooks
--------------------------------------

.. code-block:: shell

   git clone https://github.com/RichieHakim/face-rhythm.git

Then open the notebooks in ``face-rhythm/notebooks/``.

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

.. code-block:: shell

   unzip tests/data_test.zip -d tests/data_test/

.. _gpu-acceleration:

GPU acceleration
----------------

face-rhythm runs on CPU by default. Install the CPU setup above first.

PyTorch compute
~~~~~~~~~~~~~~~

Set ``project.use_GPU: true`` in your params. Check CUDA with:

.. code-block:: shell

   python -c "import torch; print(torch.cuda.is_available())"

OpenCV CUDA
~~~~~~~~~~~

face-rhythm uses OpenCV CUDA for point-tracking optical flow and CLAHE
when ``cv2.cuda.getCudaEnabledDeviceCount() > 0``.

- Build OpenCV plus ``opencv_contrib`` in this Python environment.
- Enable CUDA in CMake with ``WITH_CUDA=ON``.
- Set ``OPENCV_EXTRA_MODULES_PATH=/path/to/opencv_contrib/modules``.
- If a pip OpenCV wheel shadows your build, uninstall
  ``opencv_contrib_python``, ``opencv_python``, and
  ``opencv_contrib_python_headless``.
- Verify:

  .. code-block:: shell

     python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"

Useful links:
`OpenCV CUDA build options
<https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html#cuda-support>`_
and `opencv_contrib <https://github.com/opencv/opencv_contrib>`_.

NVDEC video decoding
~~~~~~~~~~~~~~~~~~~~

NVDEC support is experimental and depends on the GPU, driver, codec,
TorchCodec build, and FFmpeg build.

- Use Linux with a supported NVIDIA GPU.
- Install a CUDA TorchCodec build:

  .. code-block:: shell

     conda install -c conda-forge 'torchcodec=*=cuda130*' ffmpeg libstdcxx-ng

- Use ``cuda126*``, ``cuda129*``, or ``cuda130*`` to match your driver.
- Set ``params["BufferedVideoReader"]["device"] = "cuda"`` or pass
  ``device="cuda"`` when constructing a reader.

Useful links:
`TorchCodec CUDA decoding
<https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/basic_cuda_example.html>`_,
`NVIDIA Video Codec SDK <https://developer.nvidia.com/video-codec-sdk>`_,
and the `NVIDIA decode/encode support matrix
<https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new>`_.

.. _installation-troubleshooting:

Installation troubleshooting
----------------------------

torchcodec FFmpeg / shared-library load errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symptom: ``import torchcodec`` raises an error such as
``libavutil.so.NN: cannot open shared object file``.

Fix on Linux:

.. code-block:: shell

   pip uninstall -y torchcodec
   conda install -c conda-forge 'torchcodec=*=cpu*' ffmpeg libstdcxx-ng

Fix on macOS:

.. code-block:: shell

   pip uninstall -y torchcodec
   conda install -c conda-forge 'torchcodec=*=cpu*' ffmpeg

Headless OpenCV on servers
~~~~~~~~~~~~~~~~~~~~~~~~~~

If ``import cv2`` raises a Qt, ``xcb``, or display-related error on a
server, swap to the headless OpenCV wheel:

.. code-block:: shell

   pip uninstall -y opencv_contrib_python opencv_python
   pip install opencv_contrib_python_headless

OpenCV package conflicts
~~~~~~~~~~~~~~~~~~~~~~~~

If both regular and headless OpenCV wheels are installed, keep only one:

.. code-block:: shell

   pip uninstall -y opencv_contrib_python opencv_contrib_python_headless opencv_python opencv_python_headless
   pip install opencv_contrib_python

Windows video backend
~~~~~~~~~~~~~~~~~~~~~

If a Windows run tries to use torchcodec, switch the video backend to
``decord`` in your params or API call.

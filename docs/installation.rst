Installation
============

face-rhythm is published on PyPI and can also be installed directly from the
GitHub repository. We recommend using a dedicated `conda
<https://docs.conda.io/en/latest/miniconda.html>`_ or ``venv`` environment to
keep the scientific Python stack isolated.

Requirements
------------

- Python 3.10, 3.11, 3.12, or 3.13
- A recent ``pip`` (``pip install --upgrade pip``)
- (Optional) A CUDA-capable GPU. The ``[all]`` install works on CPU; for
  hardware-accelerated video decoding (NVDEC) and optical flow, see the
  "Optional: GPU-accelerated video decoding and tracking" section at the
  end of the repository ``README.md``.

.. _install-pypi:

1. Install from PyPI (recommended)
----------------------------------

The fastest path is to install the release from PyPI into a fresh
environment:

.. code-block:: console

   conda create -n face_rhythm python=3.12 -y
   conda activate face_rhythm
   pip install --upgrade pip
   pip install "face-rhythm[all]"

The ``[all]`` extra installs the notebook/GUI stack, test tooling, and the
docs toolchain alongside the core dependencies. For a minimal install use
``pip install face-rhythm``.

.. _install-source:

2. Install from source
----------------------

To work against the development branch or to contribute changes, clone the
repository and install in editable mode:

.. code-block:: console

   conda create -n face_rhythm python=3.12 -y
   conda activate face_rhythm
   git clone https://github.com/RichieHakim/face-rhythm.git
   cd face-rhythm
   pip install -e ".[all]"

.. _install-notebooks:

3. Get the demo notebooks
-------------------------

The demo notebooks are not packaged on PyPI; they live in the
``notebooks/`` directory of the source tree. Clone the repository (step 2)
or download the folder directly from GitHub to follow along.

See :doc:`notebooks` for a guided list.

Verifying the install
---------------------

To confirm the package imports cleanly, run:

.. code-block:: console

   python -c "import face_rhythm; print(face_rhythm.__version__)"

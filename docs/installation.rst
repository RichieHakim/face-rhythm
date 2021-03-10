Installation
============
I prefer to use conda for package management, so I'll explain set up using conda

1. Clone this repo to a good location:

.. code-block:: console

    git clone https://github.com/akshay-jaggi/face-rhythm/
    git checkout testing


2. Create a conda environment:

.. code-block:: console

    conda create -n face-rhythm python=3.8
    conda activate face-rhythm


3. Run the set up script:

.. code-block:: console

    cd face-rhythm
    pip install -e .

4. Install the correct version of cuda toolkit (if you plan on using a gpu).
`This link <https://anaconda.org/anaconda/cudatoolkit>`_ and `this link <https://pytorch.org/get-started/locally/>`_ are useful for figuring that out:

.. code-block:: console

    conda install cudatoolkit=10.2

5. Go ahead and create a "project directory" where you'd like your intermediate files, videos, and config files saved. Ideally outside of this repo so you don't have to worry about it when pushing and pulling. Again, given that your ipynb will change a lot (get populated with plots and new parameters, it's good to copy this out of the repo while you're doing analysis.

.. code-block:: console

    cd ..
    mkdir face_rhythm_runs
    mkdir face_rhythm_runs/run_00
    cp face-rhythm/notebooks/opticflow_notebook_new.ipynb face_rhythm_runs/run_00/

6. Get started! I like to use jupyter notebook for this stuff:

.. code-block:: console

    jupyter notebook


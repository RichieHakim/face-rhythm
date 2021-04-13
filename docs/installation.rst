Installation
============
I prefer to use conda for package management, so I'll explain set up using conda

1. Clone this repo to a good location:

.. code-block:: console

    git clone https://github.com/RichieHakim/face-rhythm/
    cd face-rhythm
    git checkout dev


2. Create a conda environment:

.. code-block:: console

    conda create -n face-rhythm python=3.8
    conda activate face-rhythm


3. Run the set up script:

.. code-block:: console

    pip install -e .

4. Install the correct version of cuda toolkit (if you plan on using a gpu).
`This link <https://anaconda.org/anaconda/cudatoolkit>`_ and `this link <https://pytorch.org/get-started/locally/>`_ are useful for figuring that out:

.. code-block:: console

    conda install cudatoolkit=10.2

5. Create a "project directory" where we will save intermediate files, videos, and config files.
This project directory should ideally be outside of the repo, and you'll create a new one each time
you analyze a new dataset.
Again, given that your ipynb will change a lot (get populated with plots and new parameters,
it's good to copy this out of the repo while you're doing analysis. I typically put one notebook in
each of my project folders.

.. code-block:: console

    cd ..
    mkdir face_rhythm_run
    cp face-rhythm/notebooks/face_rhythm_notebook.ipynb face_rhythm_run/

6. Get started! I like to use jupyter notebook for this stuff:

.. code-block:: console

    jupyter notebook


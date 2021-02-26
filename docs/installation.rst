Installation
============
I prefer to use conda for package management, so I'll explain set up using conda

1. Clone this repo to a good location:

.. code-block:: console

    git clone https://github.com/RichieHakim/face-rhythm/
    git checkout akshay


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

5. Copy your video(s) into the raw video folder

6. Get started! I like to use jupyter lab for this stuff:

.. code-block:: console

    jupyter lab


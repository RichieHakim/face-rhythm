Data Organization
=================
Here I'll both define the terminology we use for talking about data, and
I'll also give some recommendations on how to structure your data when using Face Rhythm.

Terminology
-----------
Session: A session corresponds to a specific continuous experiment. This experiment was recorded and then saved as either
one single video or a series of chunked videos. If you've saved your session as multiple videos, we assume that the videos are
continuous. In other words, we assume that the videos can be ordered and that the last frame of one video is immediately followed
by the first frame of the next. This is important for when we run tracking of points from frame to frame.

Trial: Within a session, you might have multiple trials when the experimental subject is asked to perform different tasks.
If you have a trial-like structure, we assume that you provide a list of lists of indices where each list of indices
specifies the frames of the session that correspond to the trial. If your session is chunked, the trial indices should be absolute
(aka correspond to the frame number wrt to all the frames in the entire session).

Organization
------------

1. Basic
This method of data organization works great if you're just trying analyze a single video or a set of videos from a single session.
We just assume that you provide us a folder with videos in it and a file name stem that helps us narrow down what videos you want.
The file name stem is useful if there are other videos in this folder that you want to ignore.

Example:

Video_Folder
- session1_chunk1.avi
- session1_chunk2.avi
- session1_test.avi



1. Clone this repo to a good location:

.. code-block:: console

    git clone https://github.com/akshay-jaggi/face-rhythm/
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

5. Go ahead and create a "project directory" where you'd like your intermediate files, videos, and config files saved. Ideally outside of this repo so you don't have to worry about it when pushing and pulling. Again, given that your ipynb will change a lot (get populated with plots and new parameters, it's good to copy this out of the repo while you're doing analysis.

.. code-block:: console

    cd ..
    mkdir face_rhythm_runs
    mkdir face_rhythm_runs/run_00
    cp face-rhythm/notebooks/face_rhythm_notebook.ipynb face_rhythm_runs/run_00/

6. Get started! I like to use jupyter notebook for this stuff:

.. code-block:: console

    jupyter notebook


face-rhythm
==============================

Project structure for Rich Hakim's Face Rhythms

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── face-rhythm        <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── util           <- Utility scripts for data loading, tracking, saving
    │   │   
    │   │
    │   ├── optic_flow     <- Main library of functions for optic flow computations
    │   │   
    │   │
    │   ├── analysis       <- PCA, TCA, and spectral decomposition                
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Installation
------------
I prefer to use conda for package management, so I'll explain set up using conda

1. Clone this repo to a good location 
```
git clone https://github.com/RichieHakim/face-rhythm/
git checkout release
```

2. Create a conda environment 
```
conda create -n face-rhythm python=3.8
conda activate face-rhythm
```
3. Run the set up script
```
cd face-rhythm
pip install -e . 
```
4. Install the correct version of cuda toolkit (if you plan on using a gpu). [This link](https://anaconda.org/anaconda/cudatoolkit) and [this link](https://pytorch.org/get-started/locally/) are useful for figuring that out
```
conda install cudatoolkit=10.2
```
5. Go ahead and create a "project directory" where you'd like your intermediate files, videos, and config files saved. Ideally outside of this repo so you don't have to worry about it when pushing and pulling. Again, given that your ipynb will change a lot (get populated with plots and new parameters, it's good to copy this out of the repo while you're doing analysis. 
```
cd ..
mkdir face_rhythm_runs
cp face-rhythm/notebooks/opticflow_notebook_new.ipynb face_rhythm_runs/
```

6. Get started! I like to use jupyter lab for this stuff
```
jupyter lab
```

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

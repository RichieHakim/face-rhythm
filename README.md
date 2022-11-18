face-rhythm
# Face-Rhythm

Learn more at https://face-rhythm.readthedocs.io/

--------

# Installation

1. Clone this repo to a good location 
```
git clone https://github.com/RichieHakim/face-rhythm/
cd face-rhythm
```

2. Create a conda environment 
```
conda create -n face-rhythm python=3.9
conda activate face-rhythm
```
3. Run the set up script
```
pip install -e . 
```
4. Install the correct version of cuda toolkit (if you plan on using a gpu). [This link](https://anaconda.org/anaconda/cudatoolkit) and [this link](https://pytorch.org/get-started/locally/) are useful for figuring that out
```
conda install cudatoolkit=10.2
```
5. Create a "project directory" where we will save intermediate files, videos, and config files.
This project directory should ideally be outside of the repo, and you'll create a new one each time
you analyze a new dataset.
Again, given that your ipynb will change a lot (get populated with plots and new parameters,
it's good to copy this out of the repo while you're doing analysis. I typically put one notebook in
each of my project folders.

```
cd ..
mkdir face_rhythm_run
cp face-rhythm/notebooks/face_rhythm_notebook.ipynb face_rhythm_run/
```

6. Get started! The plots display better using Jupyter Notebook
```
jupyter notebook
```
If you run into a kernel error at this stage and are a Windows user, check out: 
https://jupyter-notebook.readthedocs.io/en/stable/troubleshooting.html#pywin32-issues

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Project Organization
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


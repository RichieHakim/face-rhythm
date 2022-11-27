face-rhythm
# Face-Rhythm

Learn more at https://face-rhythm.readthedocs.io/

--------

# Installation

### 0. Requirements
- [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)<br>
- GCC >= 5.4.0, ideally == 9.2.0. Google how to do this on your operating system. For unix/linux: check with `gcc --version`.<br>
- For GPU support, you just need a CUDA compatible NVIDIA GPU and the relevant [drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us). There is no need to download CUDA or CUDNN as PyTorch takes care of this during the installation. Using a GPU is not required, but can increase speeds 2-20x depending on the GPU and your data. See https://developer.nvidia.com/cuda-gpus for a list of compatible GPUs.
- On some Linux servers (like Harvard's O2 server), you may need to load modules instead of installing. To load conda, gcc, try: `module load conda3/latest gcc/9.2.0` or similar.<br>

### 1. Clone this repo 

**`git clone https://github.com/RichieHakim/face-rhythm/`**<br>
**`cd face-rhythm`**<br>

2. Create a conda environment 
#### 3A. Install dependencies with GPU support (recommended)<br>
**`conda env create --file environment_GPU.yml`**<br>

#### 3B. Install dependencies with only CPU support<br>
**`conda env create --file environment_CPU_only.yml`**<br>

3. Run the set up script
```
pip install -e . 
```
4. Create a "project directory" where we will save intermediate files, videos, and config files.
This project directory should ideally be outside of the repo, and you'll create a new one each time
you analyze a new dataset. You may want to save a copy of the .ipynb file you use for the run there.
```
cd ..
mkdir face_rhythm_run
```

5. Get started! The plots display better using Jupyter Notebook
```
jupyter notebook
```
If you run into a kernel error at this stage and are a Windows user, check out: 
https://jupyter-notebook.readthedocs.io/en/stable/troubleshooting.html#pywin32-issues

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


# Repository Organization
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

# Project Directory Organization
------------

    Project Directory
    ├── config.yaml           <- Configuration parameters to run each module in the pipeline. Dictionary.
    ├── run_info.yaml         <- Output information from each module. Dictionary.
    │
    ├── run_data              <- Output data from each module.
    │   ├── point_tracking.h5 <- Output data from optic flow module.
    │   ├── spectral.h5       <- Output data from spectral decomposition module.
    │   └── decomposition.h5  <- Output data from PCA/TCA modules.
    │   
    └── visualizations        <- Output visualizations.
        ├── example_plot.png  <- Example plot.
        └── example_video.mp4 <- Example video.

    

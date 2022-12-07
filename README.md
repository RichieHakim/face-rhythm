face-rhythm
# Face-Rhythm

Learn more at https://face-rhythm.readthedocs.io/

--------

<br>
<br>

# Installation

#### 0. Requirements <br>
- [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)<br>
- GCC >= 5.4.0, ideally == 9.2.0. Google how to do this on your operating system. For unix/linux: check with `gcc --version`.<br>
- For GPU support, you just need a CUDA compatible NVIDIA GPU and the relevant [drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us). There is no need to download CUDA or CUDNN as PyTorch takes care of this during the installation. Using a GPU is not required, but can increase speeds 2-20x depending on the GPU and your data. See https://developer.nvidia.com/cuda-gpus for a list of compatible GPUs.
- On some Linux servers (like Harvard's O2 server), you may need to load modules instead of installing. To load conda, gcc, try: `module load conda3/latest gcc/9.2.0` or similar.<br>

#### 1. Clone this repo <br>
**`git clone https://github.com/RichieHakim/face-rhythm/`**<br>
**`cd face-rhythm`**<br>

#### 2. Create a conda environment 
#### 3A. Install dependencies with GPU support (recommended)<br>
**`conda env create --file environment_GPU.yml`**<br>

#### 3B. Install dependencies with only CPU support<br>
**`conda env create --file environment_CPU_only.yml`**<br>

#### 3. Run the set up script <br>
**`pip install -e .`**<br>

<br>
<br>

# Usage

#### 1. Create a "project directory" where we will save intermediate files, videos, and config files. <br>
This project directory should ideally be outside of the repo, and you'll create a new one each time
you analyze a new dataset. You may want to save a copy of the .ipynb file you use for the run there.
**`cd directory/where/you/want/to/save/your/project`**<br>
**`mkdir face_rhythm_run`**<br>

#### 2. Open up jupyter notebook! The plots display better using Jupyter Notebook than Jupyter Lab or VSCode. <br>
**`jupyter notebook`**<br>
If you run into a kernel error at this stage and are a Windows user, check out: 
https://jupyter-notebook.readthedocs.io/en/stable/troubleshooting.html#pywin32-issues

#### 3. Open up a demo notebook and run it! <br>
- `basic_face_rhythm_notebook.ipynb` is a basic demo notebook that runs through the entire pipeline.
- `demo_align_temporal_factors.ipynb` is a demo notebook that shows how to align the temporal factors that are output from the basic pipeline.

<br>
<br>

# Repository Organization
    face-rhythm
    ├── notebooks  <- Jupyter notebooks containing the main pipeline and some demos.
    |   ├── basic_face_rhythm_notebook.ipynb  <- Main pipeline notebook.
    |   └── demo_align_temporal_factors.ipynb <- Demo notebook for aligning temporal factors.
    |
    ├── face-rhythm  <- Source code for use in this project.
    │   ├── project.py           <- Contains methods for project directory organization and preparation
    │   ├── data_importing.py    <- Contains classes for importing data (like videos)
    |   ├── rois.py              <- Contains classes for defining regions of interest (ROIs) to analyze
    |   ├── point_tracking.py    <- Contains classes for tracking points in videos
    |   ├── spectral_analysis.py <- Contains classes for spectral decomposition
    |   ├── decomposition.py     <- Contains classes for TCA decomposition
    |   ├── utils.py             <- Contains utility functions for face-rhythm
    |   ├── visualization.py     <- Contains classes for visualizing data
    |   ├── helpers.py           <- Contains general helper functions (non-face-rhythm specific)
    |   ├── h5_handling.py       <- Contains classes for handling h5 files
    │   └── __init__.py          <- Makes src a Python module    
    |
    ├── setup.py   <- makes project pip installable (pip install -e .) so src can be imported
    ├── LICENSE    <- License file
    ├── Makefile   <- Makefile with commands like `make data` or `make train`
    ├── README.md  <- The top-level README for developers using this project.
    ├── docs       <- A default Sphinx project; see sphinx-doc.org for details
    └── tox.ini    <- tox file with settings for running tox; see tox.readthedocs.io

<br>
<br>

# Project Directory Organization

    Project Directory
    ├── config.yaml           <- Configuration parameters to run each module in the pipeline. Dictionary.
    ├── run_info.json         <- Output information from each module. Dictionary.
    │
    ├── run_data              <- Output data from each module.
    │   ├── point_tracking.h5 <- Output data from optic flow module.
    │   ├── spectral.h5       <- Output data from spectral decomposition module.
    │   └── decomposition.h5  <- Output data from PCA/TCA modules.
    │   
    └── visualizations        <- Output visualizations.
        ├── example_plot.png  <- Example plot.
        └── example_video.mp4 <- Example video.

    

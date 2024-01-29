# Face-Rhythm

Learn more at https://face-rhythm.readthedocs.io/

--------

<br>
<br>

# Installation

### 0. Requirements <br>
- Operating system:
  - Ubuntu >= 18.04 (other linux versions usually okay but not actively maintained)
  - Windows >= 10
  - Mac >= 12
- [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- If using linux/unix: GCC >= 5.4.0, ideally == 9.2.0. Google how to do this on your operating system. Check with: `gcc --version`.
- **Optional:** [CUDA compatible NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) and [drivers](https://developer.nvidia.com/cuda-toolkit-archive). Using a GPU can increase the speeds for the TCA step, but is not necessary.
- The below commands should be run in the terminal (Mac/Linux) or Anaconda Prompt (Windows).
<br>

### 1. Clone this repo <br>
This will create a folder called **face-rhythm** in your current directory. This repository folder contains the source code AND the interactive notebooks needed to run the pipeline. <br>
**`git clone https://github.com/RichieHakim/face-rhythm/`**<br>
**`cd face-rhythm`**<br>

### 2. Create a conda environment
This will also install the **face-rhythm** package and all of its dependencies into the environment. <br>
**`conda env create --file environment.yml`**<br>

Activate the environment: <br>
**`conda activate face_rhythm`** <br>

### Optional Direct installation <br>
You can also directly install the **face-rhythm** package from PyPI into the environment of your choice. Note that you will still need to download/clone the repository for the notebooks. <br>
##### Option 1: Install from PyPI <br>
**`pip install face-rhythm`**<br>
##### Option 2: Install from source <br>
**`pip install -e .`**<br>

<br>
<br>

# Usage

#### Notebooks
The easiest way to use **face-rhythm** is through the interactive notebooks. They are found in the following directory: `face-rhythm/notebooks/`. <br>
- The `interactive_pipeline_basic.ipynb` notebook contains the main pipeline and instructions on how to use it. <br>
- The `interactive_set_ROIs_only.ipynb` notebook is useful for when you want to run a batch job of many videos/sessions and need to set the ROIs for each video/session ahead of time. <br>

#### Command line
The basic pipeline in the interactive notebook is also provided as a function within the `face_rhythm/pipelines.py` module. In the `scripts` folder, you'll find a script called `run_pipeline_basic.py` that can be used to run the pipeline from the command line. An example `params.json` file is also in that folder to use as a template for your runs. <br>



<br>
<br>

# Repository Organization
    face-rhythm
    ├── notebooks  <- Jupyter notebooks containing the main pipeline and some demos.
    |   ├── basic_face_rhythm_notebook.ipynb  <- Main pipeline notebook.
    |   └── interactive_set_ROIs_only.ipynb   <- Notebook for setting ROIs only.
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
    │   ├── Dataset_videos.h5 <- Output data from Dataset_videos class. Contains metadata about the videos.
    │   ├── ROIs.h5           <- Output data from ROIs class. Contains ROI masks.
    │   ├── PointTracker.h5   <- Output data from PointTracker class. Contains point tracking data.
    |   ├── VQT_Analyzer.h5   <- Output data from VQT_Analyzer class. Contains spectral decomposition data.
    │   ├── TCA.h5            <- Output data from TCA class. Contains TCA decomposition data.
    │   
    └── visualizations        <- Output visualizations.
        ├── factors_rearranged_[frequency].png  <- Example of a rearranged factor plot.
        └── point_tracking_demo.avi             <- Example video.

    

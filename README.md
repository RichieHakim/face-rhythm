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
#### 2A. Install dependencies with GPU support (recommended)<br>
**`conda env create --file environment_GPU.yml`**<br>

#### 2B. Install dependencies with only CPU support<br>
**`conda env create --file environment_CPU_only.yml`**<br>

In either case, this step will create a conda environment named face-rhythm. Activate it: 
**`conda activate face-rhythm`** <br>

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

#### 2. Copy the interactive notebook to your project directory 
We recommend copying the interactive notebook from your face-rhythm repository to your project folder each time you make a new project. This will allow you to have one notebook per project, which will keep your analyses from potentially conflicting if you run different datasets through the same notebooks. 
**`cp /path to face-rhythm repo/face-rhythm/notebooks/interactive_pipeline_basic.ipynb.ipynb /path to project/face_rhythm_run/`**<br>

`interactive_pipeline_basic.ipynb.ipynb` is a basic demo notebook that runs through the entire pipeline.
See the `notebooks/other` folder for some notebooks demonstrating other kinds of analyses. These are more experimental and are subject to change as we develop new analyses. 

#### 3. Open up jupyter notebook! The plots display better using Jupyter Notebook than Jupyter Lab or VSCode. <br>
**`jupyter notebook`**<br>
If you run into a kernel error at this stage and are a Windows user, check out: 
https://jupyter-notebook.readthedocs.io/en/stable/troubleshooting.html#pywin32-issues

Navigate to your folder containing your interactive notebook and launch it by clicking on it! 


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
    │   ├── Dataset_videos.h5 <- Output data from Dataset_videos class. Contains metadata about the videos.
    │   ├── ROIs.h5           <- Output data from ROIs class. Contains ROI masks.
    │   ├── PointTracker.h5   <- Output data from PointTracker class. Contains point tracking data.
    |   ├── VQT_Analyzer.h5   <- Output data from VQT_Analyzer class. Contains spectral decomposition data.
    │   ├── TCA.h5            <- Output data from TCA class. Contains TCA decomposition data.
    │   
    └── visualizations        <- Output visualizations.
        ├── factors_rearranged_[frequency].png  <- Example of a rearranged factor plot.
        └── point_tracking_demo.avi             <- Example video.

    

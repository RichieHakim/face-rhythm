# Welcome to face-rhythm

[![PyPI version](https://badge.fury.io/py/face-rhythm.svg)](https://badge.fury.io/py/face-rhythm)
[![Downloads](https://pepy.tech/badge/face-rhythm)](https://pepy.tech/project/face-rhythm)
[![Python versions](https://img.shields.io/pypi/pyversions/face-rhythm.svg)](https://pypi.org/project/face-rhythm/)
[![build](https://github.com/RichieHakim/face-rhythm/actions/workflows/build.yml/badge.svg)](https://github.com/RichieHakim/face-rhythm/actions/workflows/build.yml)
[![Documentation Status](https://readthedocs.org/projects/face-rhythm/badge/?version=latest)](https://face-rhythm.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

- **Documentation:** [https://face-rhythm.readthedocs.io](https://face-rhythm.readthedocs.io)
- **Preprint:** [Hakim et al. (2025), *bioRxiv*](https://doi.org/10.1101/2025.09.10.675423)
- **Issues / support:** [GitHub Issues](https://github.com/RichieHakim/face-rhythm/issues)

## What is face-rhythm

A Python package that turns videos of facial or other behavior into a small set of interpretable behavioral components.

**Why use face-rhythm?**
- **Unsupervised.** No labels, no model zoo.
- **Interpretable.** Each component is a (space × frequency × time) factor
  you can plot and read off directly.

## How to use it

**Interactive notebooks:**

- [`demo_pipeline.ipynb`](https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_pipeline.ipynb)
  — end-to-end demo on a single session. Start here.
- [`demo_set_rois_multisession.ipynb`](https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_set_rois_multisession.ipynb)
  — draw and align ROIs across multiple sessions of the same subject.
- [`demo_event_alignment.ipynb`](https://github.com/RichieHakim/face-rhythm/blob/release/notebooks/demo_event_alignment.ipynb)
  — align extracted factors to event timestamps and view trial-averaged
  traces.

**Command line** for batch runs across many sessions:
```shell
python scripts/run_pipeline_basic.py --path_params params.json --directory_save /path/to/project/
```
`scripts/params_pipeline_basic.json` is a ready-to-edit template.

**Python API:** see [Quick start](#quick-start) below, or the
[full API reference](https://face-rhythm.readthedocs.io/en/latest/api.html).

## Installation

### 0. Requirements

- [Anaconda](https://www.anaconda.com/distribution/) or
  [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)

### 1. Create a conda environment

```shell
conda create -n face_rhythm python=3.12
conda activate face_rhythm
python -m pip install --upgrade pip
```

Activate the env (`conda activate face_rhythm`) each time you use
face-rhythm.

### 2. Install video packages

**Linux:**
```shell
conda install -c conda-forge 'torchcodec=*=cpu*' ffmpeg libstdcxx-ng
```

**macOS:**
```shell
conda install -c conda-forge 'torchcodec=*=cpu*' ffmpeg
```

**Windows:** skip this step. `torchcodec` doesn't explicitly support Windows. Installing it often works, but is not guaranteed. Unless you need ultrafast GPU speeds, just use the `'decord'` backend, instead.

### 3. Install face-rhythm

```shell
pip install face-rhythm
```

For headless servers, GPU acceleration, and installation troubleshooting,
see the [installation docs](https://face-rhythm.readthedocs.io/en/latest/installation.html).

### 4. Clone the repo to get the notebooks

```shell
git clone https://github.com/RichieHakim/face-rhythm.git
```
<!-- end-install -->

## CLI Quick start

<!-- start-quickstart -->
```python
import json
import face_rhythm as fr

with open("params_pipeline_basic.json", "r") as f:
    params = json.load(f)

params["project"]["directory_project"] = "/path/to/new/project/"
params["paths_videos"]["directory_videos"] = "/path/to/videos/"
params["ROIs"]["initialize"]["path_file"] = "/path/to/ROIs.h5"

results = fr.pipelines.pipeline_basic(params)
```

Copy [`scripts/params_pipeline_basic.json`](scripts/params_pipeline_basic.json)
as a template, edit the three paths, and run. Results land in the project
directory as HDF5 files plus summary plots.
<!-- end-quickstart -->

## Upgrading

```shell
pip install --upgrade face-rhythm
```

To update the cloned notebooks/scripts: `cd face-rhythm && git pull`.

## Pipeline at a glance

1. Read the video frames ([`face_rhythm.helpers.BufferedVideoReader`](https://face-rhythm.readthedocs.io/en/latest/api.html)).
2. Draw ROIs that pick (a) where to track and (b) what region to crop
   ([`face_rhythm.rois`](https://face-rhythm.readthedocs.io/en/latest/api.html)).
3. Track a dense grid of points via optical flow
   ([`face_rhythm.point_tracking`](https://face-rhythm.readthedocs.io/en/latest/api.html)).
4. Compute a spectrogram for each point's trajectory
   ([`face_rhythm.spectral_analysis`](https://face-rhythm.readthedocs.io/en/latest/api.html)).
5. Factorize the (points × frequency × time) tensor with non-negative TCA
   ([`face_rhythm.decomposition`](https://face-rhythm.readthedocs.io/en/latest/api.html)).

## GPU acceleration (optional)

face-rhythm runs on CPU by default. Install the CPU setup above first.

**PyTorch compute:** set `project.use_GPU: true` in your params. Check CUDA
with:
```shell
python -c "import torch; print(torch.cuda.is_available())"
```

**OpenCV CUDA:** build OpenCV plus `opencv_contrib` with CUDA enabled, then
make sure that build is the `cv2` imported in this env. Useful links:
[OpenCV CUDA build options](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html#cuda-support)
and [opencv_contrib](https://github.com/opencv/opencv_contrib).

**NVDEC video decoding:** (uses experimental libraries). On Linux/NVIDIA systems, try a CUDA torchcodec package, then pass `device='cuda'` when constructing video readers:
```shell
conda install -c conda-forge 'torchcodec=*=cuda130*' ffmpeg libstdcxx-ng
```
Use `cuda126*`, `cuda129*`, or `cuda130*` to match your driver. Useful
links: [TorchCodec CUDA decoding](https://meta-pytorch.org/torchcodec/stable/generated_examples/decoding/basic_cuda_example.html)
and [NVIDIA Video Codec SDK](https://developer.nvidia.com/video-codec-sdk).

## Citation

If you use face-rhythm in your research, please cite our preprint:

> Hakim et al. (2025). Spectral envelopes of facial movements predict
> intention, cortical representations, and neural prosthetic control.
> *bioRxiv*. https://doi.org/10.1101/2025.09.10.675423

BibTeX and a machine-readable `CITATION.cff` are at the root of the repo.

## Contributing

Bug reports, feature requests, and pull requests are welcome. Please open
an [issue](https://github.com/RichieHakim/face-rhythm/issues) before
submitting substantial changes.

## License

MIT — see [LICENSE](LICENSE).

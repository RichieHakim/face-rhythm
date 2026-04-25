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

## Rhythmic facial movements from video ᗢ

A Python package that turns videos of facial or other behavior into a small set of interpretable behavioral components.

**Why use face-rhythm?**
- **Unsupervised.** No labels, no model zoo — you give it a video, it gives
  you a handful of components.
- **Interpretable.** Each component is a (space × frequency × time) factor
  you can plot and read off directly.

## How to use face-rhythm

**Interactive notebooks**:

- [`demo_pipeline.ipynb`](https://github.com/RichieHakim/face-rhythm/blob/dev/notebooks/demo_pipeline.ipynb)
  — end-to-end demo on a single session. Start here.
- [`demo_set_rois_multisession.ipynb`](https://github.com/RichieHakim/face-rhythm/blob/dev/notebooks/demo_set_rois_multisession.ipynb)
  — draw and align ROIs across multiple sessions of the same subject.
- [`demo_event_alignment.ipynb`](https://github.com/RichieHakim/face-rhythm/blob/dev/notebooks/demo_event_alignment.ipynb)
  — line the extracted factors up with event timestamps and look at
  trial-averaged traces.

**Command line** for batch runs across many sessions:
```shell
python scripts/run_pipeline_basic.py --path_params params.json --directory_save /path/to/project/
```
`scripts/params_pipeline_basic.json` is a ready-to-edit template.

**Python API:** see [Quick start](#quick-start) below, or the
[full API reference](https://face-rhythm.readthedocs.io/en/latest/api.html).

## Installation

If you have any issues during installation, please open a
[GitHub issue](https://github.com/RichieHakim/face-rhythm/issues).

### 0. Requirements
- [Anaconda](https://www.anaconda.com/distribution/) or
  [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- **ffmpeg** (system library, required by torchcodec):
  - Linux: `apt install ffmpeg`
  - macOS: `brew install ffmpeg`
  - conda (any OS): `conda install -c conda-forge ffmpeg`

### 1. Create a new conda environment
```shell
conda create -n face_rhythm python=3.12 -y
conda activate face_rhythm
```
You will need to activate the environment with `conda activate face_rhythm`
each time you want to use face-rhythm.

### 2. Install face-rhythm

```shell
pip install "face-rhythm[all]"
```

This installs both video backends (`torchcodec` and `decord`) where available:

- **Linux / macOS:** torchcodec is the default. `BufferedVideoReader()` uses
  it transparently and includes a built-in workaround for torchcodec issue
  [#905](https://github.com/meta-pytorch/torchcodec/issues/905).
- **Windows:** torchcodec has no Windows wheels, so only the decord backend
  is available. Construct `BufferedVideoReader(..., backend='decord')`
  explicitly. (The torchcodec wrapper raises a helpful error pointing to
  this if you forget.)

For headless installs on servers (where you can't visually view videos),
you'll need to `pip uninstall opencv_contrib_python` and
`pip install opencv_contrib_python_headless`.

### 3. Clone the repo to get the notebooks
```shell
git clone https://github.com/RichieHakim/face-rhythm.git
```
Then open the notebooks in `face-rhythm/notebooks/`.

## Quick start

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

## Upgrading

```shell
pip install --upgrade "face-rhythm[all]"
```

To update the notebooks/scripts from a clone:
```shell
cd face-rhythm && git pull
```

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

---

## Optional: GPU-accelerated video decoding and tracking

Neither of the following is required — face-rhythm runs on CPU with the
stock `pip`-installed OpenCV. These notes are for users who want extra
throughput on a CUDA GPU.

### OpenCV built with CUDA (optical-flow speedup)

face-rhythm automatically uses OpenCV's CUDA optical-flow and CLAHE if
available. The stock `opencv-python` wheel on PyPI does **not** include
CUDA support — you need to build OpenCV from source.

Relevant CMake flags: `-DWITH_CUDA=ON`, `-DWITH_CUDNN=ON`,
`-DOPENCV_DNN_CUDA=ON`, and `-DCUDA_ARCH_BIN=<your compute capability>`.
See the upstream guide:
<https://docs.opencv.org/4.x/d2/de6/tutorial_py_setup_in_ubuntu.html>.

### NVDEC hardware video decoding

[`torchcodec`](https://github.com/pytorch/torchcodec) is the default video
decoding backend on Linux and macOS. It supports both CPU software decoding
and NVIDIA NVDEC GPU decoding. Pass `device='cuda'` (or `device='cuda:0'`)
to `BufferedVideoReader` to activate NVDEC. GPU decoding requires
torchcodec built with CUDA support and an FFmpeg build with `--enable-cuda`.

For CUDA-enabled torchcodec wheels and GPU setup instructions, see:
- torchcodec CUDA install guide: <https://pytorch.org/torchcodec/stable/>
- NVIDIA Video Codec SDK: <https://developer.nvidia.com/video-codec-sdk>

**Windows users:** torchcodec has no Windows wheels. Install with
`pip install "face-rhythm[decord]"` and use `backend='decord'` instead.
GPU decoding via decord is not currently supported.

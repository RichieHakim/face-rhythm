# -*- coding: utf-8 -*-
#
# face-rhythm documentation build configuration file.
#
# For the full list of configuration options, see
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import re
import sys
import types
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# cv2 mock shim
# ---------------------------------------------------------------------------
# autodoc's default mock (MagicMock) does not support bitwise operators, so
# class-body expressions like
#     cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
# (used in face_rhythm.point_tracking.PointTracker) raise TypeError at import
# time and break autodoc for that module and anything it transitively touches.
# We pre-install a real module in sys.modules with int constants for the
# TermCriteria flags; everything else falls back to MagicMock.

class _CV2Module(types.ModuleType):
    TERM_CRITERIA_EPS = 2
    TERM_CRITERIA_COUNT = 1
    TERM_CRITERIA_MAX_ITER = 1

    def __getattr__(self, name):  # noqa: D401 — sphinx-only shim
        attr = MagicMock()
        setattr(self, name, attr)
        return attr


sys.modules["cv2"] = _CV2Module("cv2")


# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = "face-rhythm"
author = "Rich Hakim"
copyright = f"{datetime.now():%Y}, {author}"


def _get_version() -> str:
    """Resolve the project version without importing face_rhythm.

    ``face_rhythm/__init__.py`` executes ``import torch`` at module load,
    which is not guaranteed to succeed in a docs-build environment (RTD
    container, local env without a GPU/CPU wheel installed, etc.). To
    avoid that failure mode we:

    1. Ask :mod:`importlib.metadata` for the version of the installed
       distribution. This works whenever the package has been
       ``pip install``-ed (which RTD does before running Sphinx) and does
       **not** import any of the package code.
    2. Fall back to parsing ``__version__`` out of
       ``face_rhythm/_version.py`` with a regex when the distribution is
       not installed (e.g. a bare ``sphinx-build`` in a fresh venv).
       ``_version.py`` is the canonical version source (see
       ``pyproject.toml`` ``[tool.setuptools.dynamic]``).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version("face-rhythm")
    except Exception:
        pass

    version_path = Path(__file__).resolve().parent.parent / "face_rhythm" / "_version.py"
    try:
        text = version_path.read_text(encoding="utf-8")
        match = re.search(
            r"""^__version__\s*=\s*['\"]([^'\"]+)['\"]""",
            text,
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1)
    except OSError:
        pass

    return "0.0.0+unknown"


release = _get_version()
# The short X.Y version.
version = ".".join(release.split(".")[:2])


# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",      # Parse docstrings from imported modules.
    "sphinx.ext.autosummary",  # Recursive summary tables + stub generation.
    "sphinx.ext.napoleon",     # NumPy / Google style docstrings.
    "sphinx.ext.intersphinx",  # Cross-reference external project docs.
    "sphinx.ext.viewcode",     # Add [source] links to HTML output.
    "sphinx.ext.mathjax",      # Math rendering.
    "sphinx.ext.githubpages",  # GitHub Pages support (harmless elsewhere).
    "myst_parser",             # Parse Markdown alongside reST.
    "sphinx_copybutton",       # Copy-to-clipboard button on code blocks.
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file suffixes. myst_parser picks up the .md entries.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
pygments_style = "sphinx"


# ---------------------------------------------------------------------------
# Autodoc / autosummary
# ---------------------------------------------------------------------------

# Mock every heavy or optional runtime dependency. Autodoc only needs to
# *import* the modules to read their signatures and docstrings; actual
# execution is unnecessary. Being generous here keeps the build green
# even when the RTD container cannot install GPU/CUDA wheels.
autodoc_mock_imports = [
    "torch",
    "torchvision",
    "torchaudio",
    # NOTE: 'cv2' is intentionally NOT in this list — it is pre-installed
    # above as a real module with int TermCriteria constants, because
    # MagicMock does not support bitwise OR which PointTracker uses at
    # class-definition time.
    "decord",
    "eva_decord",
    "tensorly",
    "scipy",
    "numpy",
    "matplotlib",
    "h5py",
    "tables",
    "sklearn",
    "scikit_learn",
    "scikit_image",
    "skimage",
    "opencv_contrib_python",
    "opencv_python",
    "einops",
    "natsort",
    "pandas",
    "PIL",
    "Pillow",
    "tqdm",
    "yaml",
    "pyyaml",
    "nvidia_ml_py3",
    "pynvml",
    "py_cpuinfo",
    "cpuinfo",
    "GPUtil",
    "psutil",
    "ffmpeg",
    "vqt",
    "requests",
    "ipywidgets",
    "ipympl",
    "jupyter",
    "notebook",
    "librosa",
    "torchcodec",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

autosummary_generate = True
autosummary_imported_members = False

# Napoleon — accept both styles; lean on NumPy conventions.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True


# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


# ---------------------------------------------------------------------------
# MyST-Parser
# ---------------------------------------------------------------------------

# Only pure-Python MyST extensions are enabled here so the build does
# not require optional deps (e.g. `linkify-it-py`, which is not included
# by plain `myst-parser`).
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

# `sphinx_rtd_theme` ships as part of the `[docs]` extra in pyproject.toml
# and is the canonical Read the Docs theme. We purposely do not import it
# at module load time so that a bare environment without the theme can
# still render a warning rather than crashing on config load.
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
htmlhelp_basename = "face-rhythmdoc"


# ---------------------------------------------------------------------------
# LaTeX / manpage / texinfo (kept minimal; rarely used in practice)
# ---------------------------------------------------------------------------

latex_documents = [
    ("index", "face-rhythm.tex", "face-rhythm Documentation", author, "manual"),
]

man_pages = [
    ("index", "face-rhythm", "face-rhythm Documentation", [author], 1),
]

texinfo_documents = [
    (
        "index",
        "face-rhythm",
        "face-rhythm Documentation",
        author,
        "face-rhythm",
        "Extract and decompose rhythmic facial movements from video.",
        "Miscellaneous",
    ),
]

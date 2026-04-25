"""Single source of truth for the face-rhythm package version.

Kept in its own module with NO other imports so build-time tooling
(``setuptools`` dynamic version lookup via ``attr``) and CI scripts
(``.github/scripts/increment_version.py``) can read/update it without
triggering heavy imports (``torch``, ``cv2``) from ``face_rhythm/__init__.py``.
"""

__version__ = '0.3.1'

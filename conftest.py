from pathlib import Path
import shutil
import tempfile
from typing import Union, Optional, List, Tuple
import functools
import warnings

import numpy as np

import pytest

import face_rhythm as fr

"""
WARNING: DO NOT REQUIRE ANY DEPENDENCIES FROM ANY NON-STANDARD LIBRARY
 MODULES IN THIS FILE. It is intended to be run before any other
 modules are imported.
"""


@pytest.fixture(scope='session')
def dir_data_test():
    """
    Prepares the directory containing the test data.
    Steps:
        1. Determine the path to the data directory.
        2. Create the data directory if it does not exist.
        3. Download the test data if it does not exist.
            If the data exists, check its hash.
        4. Extract the test data.
        5. Return the path to the data directory.
    """
    # dir_data_test = str(Path('data_test/').resolve().absolute())
    dir_data_test = str((Path(tempfile.gettempdir()) / 'data_test').resolve().absolute())
    print(dir_data_test)
    path_data_test_zip = download_data_test_zip(dir_data_test)
    fr.helpers.extract_zip(
        path_zip=path_data_test_zip, 
        path_extract=dir_data_test,
        verbose=True,
    )
    return dir_data_test

def download_data_test_zip(directory):
    """
    Provides the test data zip at ``<directory>/data_test.zip``.

    Resolution order:
        1. If a local checkout copy exists at ``<repo>/tests/data_test.zip``,
           copy it to ``path_save``. This avoids flaky network access in CI
           (e.g., transient HTTP 502s on raw.githubusercontent.com).
        2. Otherwise, fall back to downloading from GitHub raw.

    Args:
        directory (str):
            Directory in which to place ``data_test.zip``.

    Returns:
        (str):
            Absolute path to the resulting ``data_test.zip``.

    Raises:
        FileNotFoundError:
            If the network fallback was used and the file is still missing
            after the download attempt (e.g., silent HTTP failure).
    """
    path_save = str(Path(directory) / 'data_test.zip')

    ## 1. Prefer the local checkout copy if it exists
    path_local = Path(__file__).parent / 'tests' / 'data_test.zip'
    if path_local.exists():
        Path(path_save).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(path_local), path_save)
        return path_save

    ## 2. Fallback: download from GitHub raw
    fr.helpers.download_file(
        url=r'https://github.com/RichieHakim/face-rhythm/raw/dev/tests/data_test.zip',
        path_save=path_save,
        check_local_first=True,
        check_hash=True,
        hash_type='MD5',
        hash_hex=r'cbf9bdefd1a1bd3c66cc632098edc27c',
        mkdir=True,
        allow_overwrite=True,
        write_mode='wb',
        verbose=True,
        chunk_size=1024,
    )
    if not Path(path_save).exists():
        raise FileNotFoundError(
            f"Download of data_test.zip failed (HTTP error or network issue). "
            f"Expected at: {path_save}"
        )
    return path_save

@pytest.fixture(scope='session')
def array_hasher():
    """
    Returns a function that hashes an array.
    """
    from functools import partial
    import xxhash
    return partial(xxhash.xxh64_hexdigest, seed=0)
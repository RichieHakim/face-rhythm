"""General-purpose helpers: video I/O wrappers, path tools, image warping, downloads.

Collected utilities used across the face-rhythm package. Notable groups:

* Video readers (``VideoReaderWrapper``, ``BufferedVideoReader``) around
  ``decord`` / ``torchcodec`` with pre-fetch threads.
* Path and file helpers (``find_paths``, ``prepare_filepath_for_saving``,
  download + hash verification, zip extraction).
* Image registration helpers (``find_geometric_transformation``, remap-index
  and flow-field conversions) used by :mod:`face_rhythm.rois`.
* Parameter dictionary utilities (``fill_missing_keys_with_defaults``,
  ``flatten_dict``) and a handful of numerical / plotting / device utilities.

Some routines are adapted from Rich Hakim's ``basic_neural_processing_modules``.
"""

import multiprocessing as mp
import threading
from typing import Union
import time
import gc
import json
from pathlib import Path
import copy
import re
from typing import List, Optional, Tuple, Union, Dict, Any, Callable, MutableMapping
import os
import sys
from functools import partial
import warnings

import numpy as np
import cv2
try:
    import decord
except ImportError:
    decord = None
import torch
from tqdm.auto import tqdm
import yaml
import zipfile
import pickle

import scipy
import scipy.sparse
import scipy.signal

import tensorly as tl

def prepare_cv2_imshow():
    """
    Pre-initializes ``cv2.imshow`` to avoid kernel crashes. RH 2022

    Calling ``cv2.imshow`` after ``av`` or ``decord`` have been imported can
    crash the Python kernel. Showing a small dummy frame here primes the
    OpenCV display loop so subsequent ``cv2.imshow`` calls work safely.
    """
    import numpy as np
    import cv2
    test = np.zeros((1,300,400,3))
    for frame in test:
        cv2.putText(frame, "WELCOME TO FACE RHYTHM!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Prepping CV2", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Calling this figure allows cv2.imshow ", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "to work without crashing if this function", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "is called before importing av and decord", (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('startup', frame)
        cv2.waitKey(1000)
    cv2.destroyWindow('startup')



#############################################################################################################################################################################
################################################################################### BNPM ####################################################################################
######################################################################## EVERYTHING BELOW IS FROM THE #######################################################################
############################################################# BASIC NEURAL PROCESSING MODULES (BNPM) REPOSITORY #############################################################
####################################################### https://github.com/RichieHakim/basic_neural_processing_modules ######################################################
#############################################################################################################################################################################


#####################################################################################################################################
########################################################### PATH HELPERS ############################################################
#####################################################################################################################################

def find_paths(
    dir_outer: Union[str, List[str]],
    reMatch: str = 'filename', 
    reMatch_in_path: Optional[str] = None,
    find_files: bool = True, 
    find_folders: bool = False, 
    depth: int = 0, 
    natsorted: bool = True, 
    alg_ns: Optional[str] = None,
    verbose: bool = False,
) -> List[str]:
    """
    Searches for files and/or folders recursively in a directory using a regex
    match. 
    RH 2022-2023

    Args:
        dir_outer (Union[str, List[str]]):
            Path(s) to the directory(ies) to search. If a list of directories,
            then all directories will be searched.
        reMatch (str): 
            Regular expression to match. Each file or folder name encountered
            will be compared using ``re.search(reMatch, filename)``. If the
            output is not ``None``, the file will be included in the output.
        reMatch_in_path (Optional[str]):
            Additional regular expression to match anywhere in the upper path.
            Useful for finding files/folders in specific subdirectories. If
            ``None``, then no additional matching is done. \n
            (Default is ``None``)
        find_files (bool): 
            Whether to find files. (Default is ``True``)
        find_folders (bool): 
            Whether to find folders. (Default is ``False``)
        depth (int): 
            Maximum folder depth to search. (Default is *0*). \n
            * depth=0 means only search the outer directory. 
            * depth=2 means search the outer directory and two levels of
              subdirectories below it
        natsorted (bool): 
            Whether to sort the output using natural sorting with the natsort
            package. (Default is ``True``)
        alg_ns (str): 
            Algorithm to use for natural sorting. See ``natsort.ns`` or
            https://natsort.readthedocs.io/en/4.0.4/ns_class.html/ for options.
            Default is PATH. Other commons are INT, FLOAT, VERSION. (Default is
            ``None``)
        verbose (bool):
            Whether to print the paths found. (Default is ``False``)

    Returns:
        (List[str]): 
            paths (List[str]): 
                Paths to matched files and/or folders in the directory.
    """
    import natsort
    if alg_ns is None:
        alg_ns = natsort.ns.PATH

    def fn_match(path, reMatch, reMatch_in_path):
        # returns true if reMatch is basename and reMatch_in_path in full dirname
        if reMatch is not None:
            if re.search(reMatch, os.path.basename(path)) is None:
                return False
        if reMatch_in_path is not None:
            if re.search(reMatch_in_path, os.path.dirname(path)) is None:
                return False
        return True

    def get_paths_recursive_inner(dir_inner, depth_end, depth=0):
        paths = []
        for path in os.listdir(dir_inner):
            path = os.path.join(dir_inner, path)
            if os.path.isdir(path):
                if find_folders:
                    if fn_match(path, reMatch, reMatch_in_path):
                        print(f'Found folder: {path}') if verbose else None
                        paths.append(path)
                if depth < depth_end:
                    paths += get_paths_recursive_inner(path, depth_end, depth=depth+1)
            else:
                if find_files:
                    if fn_match(path, reMatch, reMatch_in_path):
                        print(f'Found file: {path}') if verbose else None
                        paths.append(path)
        return paths

    def fn_check_pathLike(obj):
        if isinstance(obj, (
            str,
            Path,
            os.PathLike,
            np.str_,
            bytes,
            memoryview,
            np.bytes_,
            re.Pattern,
            re.Match,
        )):
            return True
        else:
            return False            

    dir_outer = [dir_outer] if fn_check_pathLike(dir_outer) else dir_outer
    paths = list(set(sum([get_paths_recursive_inner(str(d), depth, depth=0) for d in dir_outer], start=[])))
    if natsorted:
        paths = natsort.natsorted(paths, alg=alg_ns)
    return paths


#####################################################################################################################################
########################################################### FILE HELPERS ############################################################
#####################################################################################################################################

def prepare_path(
    path: str, 
    mkdir: bool = False, 
    exist_ok: bool = True,
) -> str:
    """
    Validates a directory or file path for saving or loading. RH 2023

    Resolution rules: \n
    * If the path exists and ``exist_ok`` is ``True``, it is accepted.
    * If the path exists and ``exist_ok`` is ``False``, an error is raised.
    * If the path does not exist and refers to a file: the parent directory
      is created when ``mkdir`` is ``True``, otherwise an error is raised
      when the parent does not exist.
    * If the path does not exist and refers to a directory: the directory is
      created when ``mkdir`` is ``True``, otherwise an error is raised. \n

    Args:
        path (str):
            Path to be checked.
        mkdir (bool):
            If ``True``, creates the parent directory (or directory) if it
            does not exist. (Default is ``False``)
        exist_ok (bool):
            If ``True``, allows the path to already exist. (Default is
            ``True``)

    Returns:
        (str):
            path (str):
                Resolved absolute path.
    """
    ## check if path is valid
    try:
        path_obj = Path(path).resolve()
    except FileNotFoundError as e:
        print(f'Invalid path: {path}')
        raise e
    
    ## check if path object exists
    flag_exists = path_obj.exists()

    ## determine if path is a directory or file
    if flag_exists:
        flag_dirFileNeither = 'dir' if path_obj.is_dir() else 'file' if path_obj.is_file() else 'neither'  ## 'neither' should never happen since path.is_file() or path.is_dir() should be True if path.exists()
        assert flag_dirFileNeither != 'neither', f'Path: {path} is neither a file nor a directory.'
        assert exist_ok, f'{path} already exists and exist_ok=False.'
    else:
        flag_dirFileNeither = 'dir' if path_obj.suffix == '' else 'file'  ## rely on suffix to determine if path is a file or directory

    ## if path exists and is a file or directory
    # all good. If exist_ok=False, then this should have already been caught above.
    
    ## if path doesn't exist and is a file
    ### if parent directory exists        
    # all good
    ### if parent directory doesn't exist
    #### mkdir if mkdir=True and raise error if mkdir=False
    if not flag_exists and flag_dirFileNeither == 'file':
        if Path(path).parent.exists():
            pass ## all good
        elif mkdir:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        else:
            assert False, f'File: {path} does not exist, Parent directory: {Path(path).parent} does not exist, and mkdir=False.'
        
    ## if path doesn't exist and is a directory
    ### mkdir if mkdir=True and raise error if mkdir=False
    if not flag_exists and flag_dirFileNeither == 'dir':
        if mkdir:
            Path(path).mkdir(parents=True, exist_ok=True)
        else:
            assert False, f'{path} does not exist and mkdir=False.'

    ## if path is neither a file nor a directory
    ### raise error
    if flag_dirFileNeither == 'neither':
        assert False, f'{path} is neither a file nor a directory. This should never happen. Check this function for bugs.'

    return str(path_obj)

def prepare_filepath_for_saving(
    filepath: str, 
    mkdir: bool = False, 
    allow_overwrite: bool = True
) -> str:
    """
    Prepares a file path for saving a file. Ensures the file path is valid and has the necessary permissions. 

    Args:
        filepath (str): 
            The file path to be prepared for saving.
        mkdir (bool): 
            If set to ``True``, creates parent directory if it does not exist. (Default is ``False``)
        allow_overwrite (bool): 
            If set to ``True``, allows overwriting of existing file. (Default is ``True``)

    Returns:
        (str): 
            path (str): 
                The prepared file path for saving.
    """
    return prepare_path(filepath, mkdir=mkdir, exist_ok=allow_overwrite)
def prepare_filepath_for_loading(
    filepath: str, 
    must_exist: bool = True
) -> str:
    """
    Prepares a file path for loading a file. Ensures the file path is valid and has the necessary permissions. 

    Args:
        filepath (str): 
            The file path to be prepared for loading.
        must_exist (bool): 
            If set to ``True``, the file at the specified path must exist. (Default is ``True``)

    Returns:
        (str): 
            path (str): 
                The prepared file path for loading.
    """
    path = prepare_path(filepath, mkdir=False, exist_ok=must_exist)
    if must_exist:
        assert Path(path).is_file(), f'{path} is not a file.'
    return path
def prepare_directory_for_saving(
    directory: str, 
    mkdir: bool = False, 
    exist_ok: bool = True
) -> str:
    """
    Prepares a directory path for saving a file. This function is rarely used.

    Args:
        directory (str): 
            The directory path to be prepared for saving.
        mkdir (bool): 
            If set to ``True``, creates parent directory if it does not exist. (Default is ``False``)
        exist_ok (bool): 
            If set to ``True``, allows overwriting of existing directory. (Default is ``True``)

    Returns:
        (str): 
            path (str): 
                The prepared directory path for saving.
    """
    return prepare_path(directory, mkdir=mkdir, exist_ok=exist_ok)
def prepare_directory_for_loading(
    directory: str, 
    must_exist: bool = True
) -> str:
    """
    Prepares a directory path for loading a file. This function is rarely used.

    Args:
        directory (str): 
            The directory path to be prepared for loading.
        must_exist (bool): 
            If set to ``True``, the directory at the specified path must exist. (Default is ``True``)

    Returns:
        (str): 
            path (str): 
                The prepared directory path for loading.
    """
    path = prepare_path(directory, mkdir=False, exist_ok=must_exist)
    if must_exist:
        assert Path(path).is_dir(), f'{path} is not a directory.'
    return path


def pickle_save(
    obj: Any, 
    filepath: str, 
    mode: str = 'wb', 
    zipCompress: bool = False, 
    mkdir: bool = False, 
    allow_overwrite: bool = True,
    **kwargs_zipfile: Dict[str, Any],
) -> None:
    """
    Saves an object to a pickle file using `pickle.dump`.
    Allows for zipping of the file.

    RH 2022

    Args:
        obj (Any): 
            The object to save.
        filepath (str): 
            The path to save the object to.
        mode (str): 
            The mode to open the file in. Options are: \n
            * ``'wb'``: Write binary.
            * ``'ab'``: Append binary.
            * ``'xb'``: Exclusive write binary. Raises FileExistsError if the
              file already exists. \n
            (Default is ``'wb'``)
        zipCompress (bool): 
            If ``True``, compresses pickle file using zipfileCompressionMethod,
            which is similar to ``savez_compressed`` in numpy (with
            ``zipfile.ZIP_DEFLATED``). Useful for saving redundant and/or sparse
            arrays objects. (Default is ``False``)
        mkdir (bool): 
            If ``True``, creates parent directory if it does not exist. (Default
            is ``False``)
        allow_overwrite (bool): 
            If ``True``, allows overwriting of existing file. (Default is
            ``True``)
        kwargs_zipfile (Dict[str, Any]): 
            Keyword arguments that will be passed into `zipfile.ZipFile`.
            compression=``zipfile.ZIP_DEFLATED`` by default.
            See https://docs.python.org/3/library/zipfile.html#zipfile-objects.
            Other options for 'compression' are (input can be either int or object): \n
                * ``0``:  zipfile.ZIP_STORED (no compression)
                * ``8``:  zipfile.ZIP_DEFLATED (usual zip compression)
                * ``12``: zipfile.ZIP_BZIP2 (bzip2 compression) (usually not as
                  good as ZIP_DEFLATED)
                * ``14``: zipfile.ZIP_LZMA (lzma compression) (usually better
                  than ZIP_DEFLATED but slower)
    """
    path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)

    if len(kwargs_zipfile)==0:
        kwargs_zipfile = {
            'compression': zipfile.ZIP_DEFLATED,
        }

    if zipCompress:
        with zipfile.ZipFile(path, 'w', **kwargs_zipfile) as f:
            f.writestr('data', pickle.dumps(obj))
    else:
        with open(path, mode) as f:
            pickle.dump(obj, f)

def pickle_load(
    filepath: str, 
    zipCompressed: bool = False,
    mode: str = 'rb',
) -> Any:
    """
    Loads an object from a pickle file.
    RH 2022

    Args:
        filepath (str): 
            Path to the pickle file.
        zipCompressed (bool): 
            If ``True``, the file is assumed to be a .zip file. The function
            will first unzip the file, then load the object from the unzipped
            file. 
            (Default is ``False``)
        mode (str): 
            The mode to open the file in. (Default is ``'rb'``)

    Returns:
        (Any): 
            obj (Any): 
                The object loaded from the pickle file.
    """
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    if zipCompressed:
        with zipfile.ZipFile(path, 'r') as f:
            return pickle.loads(f.read('data'))
    else:
        with open(path, mode) as f:
            return pickle.load(f)

def json_save(
    obj: Any, 
    filepath: str, 
    indent: int = 4, 
    mode: str = 'w', 
    mkdir: bool = False, 
    allow_overwrite: bool = True,
) -> None:
    """
    Saves an object to a json file using `json.dump`.
    RH 2022

    Args:
        obj (Any): 
            The object to save.
        filepath (str): 
            The path to save the object to.
        indent (int): 
            Number of spaces for indentation in the output json file. (Default
            is *4*)
        mode (str): 
            The mode to open the file in. Options are: \n
            * ``'wb'``: Write binary.
            * ``'ab'``: Append binary.
            * ``'xb'``: Exclusive write binary. Raises FileExistsError if the
              file already exists. \n
            (Default is ``'w'``)
        mkdir (bool): 
            If ``True``, creates parent directory if it does not exist. (Default
            is ``False``)
        allow_overwrite (bool): 
            If ``True``, allows overwriting of existing file. (Default is
            ``True``)
    """
    import json
    path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)
    with open(path, mode) as f:
        json.dump(obj, f, indent=indent)

def json_load(
    filepath: str, 
    mode: str = 'r',
) -> Any:
    """
    Loads an object from a json file.
    RH 2022

    Args:
        filepath (str): 
            Path to the json file.
        mode (str): 
            The mode to open the file in. (Default is ``'r'``)

    Returns:
        (Any): 
            obj (Any): 
                The object loaded from the json file.
    """
    import json
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    with open(path, mode) as f:
        return json.load(f)


def yaml_save(
    obj: object, 
    filepath: str, 
    indent: int = 4, 
    mode: str = 'w', 
    mkdir: bool = False, 
    allow_overwrite: bool = True,
) -> None:
    """
    Saves an object to a YAML file using the ``yaml.dump`` method.
    RH 2022

    Args:
        obj (object): 
            The object to be saved.
        filepath (str): 
            Path to save the object to.
        indent (int): 
            The number of spaces for indentation in the saved YAML file.
            (Default is *4*)
        mode (str): 
            Mode to open the file in. \n
            * ``'w'``: write (default)
            * ``'wb'``: write binary
            * ``'ab'``: append binary
            * ``'xb'``: exclusive write binary. Raises ``FileExistsError`` if
              file already exists. \n
            (Default is ``'w'``)
        mkdir (bool): 
            If ``True``, creates the parent directory if it does not exist.
            (Default is ``False``)
        allow_overwrite (bool): 
            If ``True``, allows overwriting of existing files. (Default is
            ``True``)
    """
    path = prepare_filepath_for_saving(filepath, mkdir=mkdir, allow_overwrite=allow_overwrite)
    with open(path, mode) as f:
        yaml.dump(obj, f, indent=indent)

def yaml_load(
    filepath: str, 
    mode: str = 'r', 
    loader: object = yaml.FullLoader,
) -> object:
    """
    Loads a YAML file.
    RH 2022

    Args:
        filepath (str): 
            Path to the YAML file to load.
        mode (str): 
            Mode to open the file in. (Default is ``'r'``)
        loader (object): 
            The YAML loader to use. \n
            * ``yaml.FullLoader``: Loads the full YAML language. Avoids
              arbitrary code execution. (Default for PyYAML 5.1+)
            * ``yaml.SafeLoader``: Loads a subset of the YAML language, safely.
              This is recommended for loading untrusted input.
            * ``yaml.UnsafeLoader``: The original Loader code that could be
              easily exploitable by untrusted data input.
            * ``yaml.BaseLoader``: Only loads the most basic YAML. All scalars
              are loaded as strings. \n
            (Default is ``yaml.FullLoader``)

    Returns:
        (object): 
            loaded_obj (object):
                The object loaded from the YAML file.
    """
    path = prepare_filepath_for_loading(filepath, must_exist=True)
    with open(path, mode) as f:
        return yaml.load(f, Loader=loader)    
            

def download_file(
    url: Optional[str],
    path_save: str,
    check_local_first: bool = True,
    check_hash: bool = False,
    hash_type: str = 'MD5',
    hash_hex: Optional[str] = None,
    mkdir: bool = False,
    allow_overwrite: bool = True,
    write_mode: str = 'wb',
    verbose: bool = True,
    chunk_size: int = 1024,
) -> None:
    """
    Downloads a file from a URL to a local path using requests. Checks if file
    already exists locally and verifies the hash of the downloaded file against
    a provided hash if required.
    RH 2023

    Args:
        url (Optional[str]): 
            URL of the file to download. If ``None``, then no download is
            attempted. (Default is ``None``)
        path_save (str): 
            Path to save the file to.
        check_local_first (bool): 
            Whether to check if the file already exists locally. If ``True`` and
            the file exists locally, the download is skipped. If ``True`` and
            ``check_hash`` is also ``True``, the hash of the local file is
            checked. If the hash matches, the download is skipped. If the hash
            does not match, the file is downloaded. (Default is ``True``)
        check_hash (bool): 
            Whether to check the hash of the local or downloaded file against
            ``hash_hex``. (Default is ``False``)
        hash_type (str): 
            Type of hash to use. Options are: ``'MD5'``, ``'SHA1'``,
            ``'SHA256'``, ``'SHA512'``. (Default is ``'MD5'``)
        hash_hex (Optional[str]): 
            Hash to compare to, in hexadecimal format (e.g., 'a1b2c3d4e5f6...').
            Can be generated using ``hash_file()`` or ``hashlib.hexdigest()``.
            If ``check_hash`` is ``True``, ``hash_hex`` must be provided.
            (Default is ``None``)
        mkdir (bool): 
            If ``True``, creates the parent directory of ``path_save`` if it
            does not exist. (Default is ``False``)
        write_mode (str): 
            Write mode for saving the file. Options include: ``'wb'`` (write
            binary), ``'ab'`` (append binary), ``'xb'`` (write binary, fail if
            file exists). (Default is ``'wb'``)
        verbose (bool): 
            If ``True``, prints status messages. (Default is ``True``)
        chunk_size (int): 
            Size of chunks in which to download the file. (Default is *1024*)
    """
    import os
    import requests

    # Check if file already exists locally
    if check_local_first:
        if os.path.isfile(path_save):
            print(f'File already exists locally: {path_save}') if verbose else None
            # Check hash of local file
            if check_hash:
                hash_local = hash_file(path_save, type_hash=hash_type)
                if hash_local == hash_hex:
                    print('Hash of local file matches provided hash_hex.') if verbose else None
                    return True
                else:
                    print('Hash of local file does not match provided hash_hex.') if verbose else None
                    print(f'Hash of local file: {hash_local}') if verbose else None
                    print(f'Hash provided in hash_hex: {hash_hex}') if verbose else None
                    print('Downloading file...') if verbose else None
            else:
                return True
        else:
            print(f'File does not exist locally: {path_save}. Will attempt download from {url}') if verbose else None

    # Download file
    if url is None:
        print('No URL provided. No download attempted.') if verbose else None
        return None
    try:
        response = requests.get(url, stream=True)
    except requests.exceptions.RequestException as e:
        print(f'Error downloading file: {e}') if verbose else None
        return False
    # Check response
    if response.status_code != 200:
        print(f'Error downloading file. Response status code: {response.status_code}') if verbose else None
        return False
    # Create parent directory if it does not exist
    prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)
    # Download file with progress bar
    total_size = int(response.headers.get('content-length', 0))
    wrote = 0
    with open(path_save, write_mode) as f:
        with tqdm(total=total_size, disable=(verbose==False), unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for data in response.iter_content(chunk_size):
                wrote = wrote + len(data)
                f.write(data)
                pbar.update(len(data))
    if total_size != 0 and wrote != total_size:
        print("ERROR, something went wrong")
        return False
    # Check hash
    hash_local = hash_file(path_save, type_hash=hash_type)
    if check_hash:
        if hash_local == hash_hex:
            print('Hash of downloaded file matches hash_hex.') if verbose else None
            return True
        else:
            print('Hash of downloaded file does not match hash_hex.') if verbose else None
            print(f'Hash of downloaded file: {hash_local}') if verbose else None
            print(f'Hash provided in hash_hex: {hash_hex}') if verbose else None
            return False
    else:
        print(f'Hash of downloaded file: {hash_local}') if verbose else None
        return True


def hash_file(
    path: str, 
    type_hash: str = 'MD5', 
    buffer_size: int = 65536,
) -> str:
    """
    Computes the hash of a file using the specified hash type and buffer size.
    RH 2022

    Args:
        path (str):
            Path to the file to be hashed.
        type_hash (str):
            Type of hash to use. (Default is ``'MD5'``). Either \n
            * ``'MD5'``: MD5 hash algorithm.
            * ``'SHA1'``: SHA1 hash algorithm.
            * ``'SHA256'``: SHA256 hash algorithm.
            * ``'SHA512'``: SHA512 hash algorithm.
        buffer_size (int):
            Buffer size (in bytes) for reading the file. 
            65536 corresponds to 64KB. (Default is *65536*)

    Returns:
        (str): 
            hash_val (str):
                The computed hash of the file.
    """
    import hashlib

    if type_hash == 'MD5':
        hasher = hashlib.md5()
    elif type_hash == 'SHA1':
        hasher = hashlib.sha1()
    elif type_hash == 'SHA256':
        hasher = hashlib.sha256()
    elif type_hash == 'SHA512':
        hasher = hashlib.sha512()
    else:
        raise ValueError(f'{type_hash} is not a valid hash type.')

    with open(path, 'rb') as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            hasher.update(data)

    hash_val = hasher.hexdigest()
        
    return hash_val
    

def get_dir_contents(
    directory: str,
) -> Tuple[List[str], List[str]]:
    """
    Retrieves the names of the folders and files in a directory (does not
    include subdirectories).
    RH 2021

    Args:
        directory (str):
            The path to the directory.

    Returns:
        (tuple): tuple containing:
            folders (List[str]):
                A list of folder names.
            files (List[str]):
                A list of file names.
    """
    walk = os.walk(directory, followlinks=False)
    folders = []
    files = []
    for ii,level in enumerate(walk):
        folders, files = level[1:]
        if ii==0:
            break
    return folders, files


def compare_file_hashes(
    hash_dict_true: Dict[str, Tuple[str, str]],
    dir_files_test: Optional[str] = None,
    paths_files_test: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, bool], Dict[str, str]]:
    """
    Compares hashes of files in a directory or list of paths to provided hashes.
    RH 2022

    Args:
        hash_dict_true (Dict[str, Tuple[str, str]]):
            Dictionary of hashes to compare. Each entry should be in the format:
            *{'key': ('filename', 'hash')}*.
        dir_files_test (str): 
            Path to directory containing the files to compare hashes. 
            Unused if paths_files_test is not ``None``. (Optional)
        paths_files_test (List[str]): 
            List of paths to files to compare hashes. 
            dir_files_test is used if ``None``. (Optional)
        verbose (bool): 
            If ``True``, failed comparisons are printed out. (Default is ``True``)

    Returns:
        (tuple): tuple containing:
            total_result (bool):
                ``True`` if all hashes match, ``False`` otherwise.
            individual_results (Dict[str, bool]):
                Dictionary indicating whether each hash matched.
            paths_matching (Dict[str, str]):
                Dictionary of paths that matched. Each entry is in the format:
                *{'key': 'path'}*.
    """
    if paths_files_test is None:
        if dir_files_test is None:
            raise ValueError('Must provide either dir_files_test or path_files_test.')
        
        ## make a dict of {filename: path} for each file in dir_files_test
        files_test = {filename: (Path(dir_files_test).resolve() / filename).as_posix() for filename in get_dir_contents(dir_files_test)[1]} 
    else:
        files_test = {Path(path).name: path for path in paths_files_test}

    paths_matching = {}
    results_matching = {}
    for key, (filename, hash_true) in hash_dict_true.items():
        match = True
        if filename not in files_test:
            print(f'{filename} not found in test directory: {dir_files_test}.') if verbose else None
            match = False
        elif hash_true != hash_file(files_test[filename]):
            print(f'{filename} hash mismatch with {key, filename}.') if verbose else None
            match = False
        if match:
            paths_matching[key] = files_test[filename]
        results_matching[key] = match

    return all(results_matching.values()), results_matching, paths_matching


def extract_zip(
    path_zip: str,
    path_extract: Optional[str] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Extracts a zip file.
    RH 2022

    Args:
        path_zip (str): 
            Path to the zip file.
        path_extract (Optional[str]): 
            Path (directory) to extract the zip file to.
            If ``None``, extracts to the same directory as the zip file.
            (Default is ``None``)
        verbose (bool): 
            Whether to print progress. (Default is ``True``)

    Returns:
        (List[str]): 
            paths_extracted (List[str]):
                List of paths to the extracted files.
    """
    import zipfile

    if path_extract is None:
        path_extract = str(Path(path_zip).resolve().parent)
    path_extract = str(Path(path_extract).resolve().absolute())

    print(f'Extracting {path_zip} to {path_extract}.') if verbose else None

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(path_extract)
        paths_extracted = [str(Path(path_extract) / p) for p in zip_ref.namelist()]

    print('Completed zip extraction.') if verbose else None

    return paths_extracted


#####################################################################################################################################
############################################################# INDEXING ##############################################################
#####################################################################################################################################

def make_batches(
    iterable,
    batch_size=None,
    num_batches=None,
    min_batch_size=0,
    return_idx=False,
    length=None
):
    """
    Generates batches of data from an iterable. RH 2021

    Args:
        iterable (Iterable):
            Iterable to be batched.
        batch_size (Optional[int]):
            Size of each batch. If ``None``, ``batch_size`` is computed from
            ``num_batches``. (Default is ``None``)
        num_batches (Optional[int]):
            Number of batches to make. Used only when ``batch_size`` is
            ``None``. (Default is ``None``)
        min_batch_size (int):
            Minimum size of each batch. Batches smaller than this are
            skipped. (Default is ``0``)
        return_idx (bool):
            If ``True``, yields ``(batch, [start, end])`` tuples instead of
            just the batch. (Default is ``False``)
        length (Optional[int]):
            Length of the iterable. If ``None``, uses ``len(iterable)``. Useful
            when the iterable does not implement ``__len__``. (Default is
            ``None``)

    Returns:
        (Generator):
            output (Generator):
                Yields successive batches from ``iterable``. If ``return_idx``
                is ``True``, yields ``(batch, [start, end])`` tuples.
    """

    if length is None:
        l = len(iterable)
    else:
        l = length
    
    if batch_size is None:
        batch_size = np.int64(np.ceil(l / num_batches))
    
    for start in range(0, l, batch_size):
        end = min(start + batch_size, l)
        if (end-start) < min_batch_size:
            break
        else:
            if return_idx:
                yield iterable[start:end], [start, end]
            else:
                yield iterable[start:end]


def cp_to_dense(cp, weights=None):
    """
    Reconstructs a dense tensor from a CP-format list of factor matrices. RH 2022

    Args:
        cp (List[np.ndarray]):
            List of length ``n_modes`` of 2D factor matrices, each with
            shape *(len_dim, rank)*. This is the format Tensorly uses for
            its ``'cp'`` representation. Elements may be NumPy arrays or
            ``torch.Tensor`` (matching dtype).
        weights (Optional[np.ndarray]):
            Per-rank weights of length ``rank``. If ``None``, uses a vector
            of ones. (Default is ``None``)

    Returns:
        (np.ndarray):
            dense (np.ndarray):
                Reconstructed dense tensor. shape: *(len_dim_0, len_dim_1, ...)*.
    """
    rank = cp[0].shape[1]
    n_modes = len(cp)
    str_einsum = ','.join([chr(97+m)+'r' for m in range(n_modes)]) + '->' + ''.join([chr(97+m) for m in range(n_modes)])
    if weights is None:
        weights = np.ones(rank)

    ## check if numpy or torch
    if isinstance(cp[0], np.ndarray):
        einsum = np.einsum
        weights = np.array(weights).astype(cp[0].dtype)
    elif isinstance(cp[0], torch.Tensor):
        einsum = torch.einsum
        weights = torch.as_tensor(weights).type(cp[0].dtype).to(cp[0].device)

    dense = einsum(str_einsum, *[cp[m] * weights for m in range(n_modes)])
    
    return dense


#####################################################################################################################################
######################################################### CONTAINER HELPERS #########################################################
#####################################################################################################################################

class Lazy_repeat_item():
    """
    Lazy iterator-like container that always returns the same item. RH 2021

    Args:
        item (Any):
            Item to repeat on every access.
        pseudo_length (Optional[int]):
            Reported length of the container. If ``None``, the container has
            no enforced length and ``__getitem__`` always returns ``item``.
            (Default is ``None``)

    Attributes:
        item (Any):
            The repeated item.
        pseudo_length (Optional[int]):
            Stored pseudo length.
    """
    def __init__(self, item, pseudo_length=None):
        """Initializes the repeater with the item and an optional pseudo length."""
        super().__init__()
        self.item = item
        self.pseudo_length = pseudo_length

    def __getitem__(self, i):
        """
        Returns ``self.item``.

        Args:
            i (int):
                Index requested. Ignored when ``pseudo_length`` is ``None``.
                Otherwise an ``IndexError`` is raised when ``i >= pseudo_length``.

        Returns:
            (Any):
                item (Any):
                    The repeated item.
        """
        if self.pseudo_length is None:
            return self.item
        elif i < self.pseudo_length:
            return self.item
        else:
            raise IndexError('Index out of bounds')


    def __len__(self):
        return self.pseudo_length

    def __repr__(self):
        return repr(self.item)


def deep_update_dict(dictionary, key, new_val=None, new_key=None, in_place=False):
    """
    Updates a value or renames a key inside a nested dictionary. RH 2022

    Args:
        dictionary (Dict):
            Dictionary to update.
        key (List[str]):
            Hierarchical path of string keys leading to the entry to update.
            Each element corresponds to a nesting level.
        new_val (Optional[Any]):
            New value to assign. If ``None``, ``new_key`` must be provided and
            only the key is renamed. (Default is ``None``)
        new_key (Optional[str]):
            If provided, ``key[-1]`` is removed and replaced with ``new_key``
            (mapping to ``new_val`` if given, otherwise to the existing value).
            (Default is ``None``)
        in_place (bool):
            If ``True``, updates ``dictionary`` in place and returns ``None``.
            If ``False``, returns a deep-copied updated dictionary. (Default
            is ``False``)

    Returns:
        (Optional[Dict]):
            output (Optional[Dict]):
                Updated dictionary when ``in_place`` is ``False``; otherwise
                ``None``.

    Example:
        .. highlight:: python
        .. code-block:: python

            deep_update_dict(params, ['dataloader_kwargs', 'prefetch_factor'], val)
    """
    def helper_deep_update_dict(d, key):
        if type(key) is str:
            key = [key]

        assert key[0] in d, f"RH ERROR, key: '{key[0]}' is not found"

        if type(key) is list:
            if len(key) > 1:
                helper_deep_update_dict(d[key[0]], key[1:])
            elif len(key) == 1:
                val = d[key[0]] if new_val is None else new_val
                if new_key is None:
                    d[key[0]] = val
                else:
                    d[new_key] = val
                    del d[key[0]]

    if in_place:
        helper_deep_update_dict(dictionary, key)
    else:
        d = copy.deepcopy(dictionary)
        helper_deep_update_dict(d, key)
        return d


def flatten_dict(d: MutableMapping, parent_key: str = '', sep: str ='.') -> MutableMapping:
    """
    Flattens a nested dictionary into a single dictionary. RH 2022

    All keys are coerced to strings and joined by ``sep``. Adapted from
    https://stackoverflow.com/a/6027615.

    Args:
        d (MutableMapping):
            Dictionary to flatten.
        parent_key (str):
            Key prefix prepended to flattened keys. Used internally for
            recursion. (Default is ``''``)
        sep (str):
            Separator between key components. Used internally for recursion.
            (Default is ``'.'``)

    Returns:
        (Dict):
            flattened (Dict):
                Flat dictionary with paths joined by ``sep``.
    """

    items = []
    for k, v in d.items():
        new_key = str(parent_key) + str(sep) + str(k) if parent_key else str(k)
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def find_subDict_key(d: dict, s: str, max_depth: int=9999999):
    """
    Recursively searches a nested dictionary for keys matching a regex.

    Args:
        d (dict):
            Dictionary to search.
        s (str):
            Regex pattern that keys are matched against.
        max_depth (int):
            Maximum depth to descend. ``1`` searches only the top level, ``2``
            searches the first and second levels, etc. (Default is
            ``9999999``)

    Returns:
        (List[Tuple[List[str], Any]]):
            k_all (List[Tuple[List[str], Any]]):
                List of 2-tuples ``(path, value)`` where ``path`` is the list
                of string keys leading to the matched entry and ``value`` is
                the matched sub-dictionary value.
    """
    def helper_find_subDict_key(d, s, depth=999, _k_all=[]):
        """
        _k_all: 
            Used for recursion. List of keys. Set to [] on first call.
        depth: 
            Used for recursion. Decrements by 1 each call. At 0, stops
             recursion.
        """
        if depth > 0:    
            depth -= 1
            for k, v in d.items():
                if re.search(s, k):
                    yield _k_all + [k], v
                if isinstance(v, dict):
                    yield from helper_find_subDict_key(v, s, depth, _k_all + [k])
    return list(helper_find_subDict_key(d, s, depth=max_depth, _k_all=[]))


## parameter dictionary helpers ##

def fill_in_dict(
    d: Dict,
    defaults: Dict,
    verbose: bool = True,
    hierarchy: List[str] = ['dict'],
):
    """
    Fills in a dictionary in place with values from ``defaults`` for missing
    keys, recursing into nested dictionaries. RH 2023

    Args:
        d (Dict):
            Dictionary to fill in (modified in place).
        defaults (Dict):
            Dictionary of default values.
        verbose (bool):
            If ``True``, prints a message each time a default value is
            inserted. (Default is ``True``)
        hierarchy (List[str]):
            Path of keys leading to ``d``. Used internally for recursion.
            (Default is ``['dict']``)
    """
    from copy import deepcopy
    for key in defaults:
        if key not in d:
            print(f"Key '{key}' not found in params dictionary: {' > '.join([f'{str(h)}' for h in hierarchy])}. Using default value: {defaults[key]}") if verbose else None
            d.update({key: deepcopy(defaults[key])})
        elif isinstance(defaults[key], dict):
            assert isinstance(d[key], dict), f"Key '{key}' is a dict in defaults, but not in params. {' > '.join([f'{str(h)}' for h in hierarchy])}."
            fill_in_dict(d[key], defaults[key], hierarchy=hierarchy+[key])
            

def check_keys_subset(
    d,
    default_dict,
    error_on_missing_keys=True,
    hierarchy=['defaults'],
):
    """
    Verifies recursively that every key in ``d`` also appears in
    ``default_dict``. RH 2023

    Args:
        d (Dict):
            Dictionary to check.
        default_dict (Dict):
            Dictionary containing the allowed keys.
        error_on_missing_keys (bool):
            If ``True``, raises ``AssertionError`` when a key in ``d`` is
            not in ``default_dict``. If ``False``, emits a warning instead.
            (Default is ``True``)
        hierarchy (List[str]):
            Path of keys leading to ``d``. Used internally for recursion.
            (Default is ``['defaults']``)
    """
    default_keys = list(default_dict.keys())
    for key in d.keys():
        if error_on_missing_keys:
            assert key in default_keys, f"Key '{key}' not found in defaults dictionary: {' > '.join([f'{str(h)}' for h in hierarchy])}."
        else:
            if key not in default_keys:
                warnings.warn(f"Key '{key}' not found in defaults dictionary: {' > '.join([f'{str(h)}' for h in hierarchy])}.")
                continue
        if isinstance(default_dict[key], dict) and isinstance(d[key], dict):
            check_keys_subset(
                d=d[key], 
                default_dict=default_dict[key], 
                error_on_missing_keys=error_on_missing_keys,
                hierarchy=hierarchy+[key],
            )


def prepare_params(
    params,
    defaults,
    error_on_missing_keys=True,
    verbose=True,
):
    """
    Validates ``params`` against ``defaults`` and fills in missing keys.

    Performs the following: \n
    * Checks that all keys in ``params`` are also in ``defaults``.
    * Fills in any missing keys in ``params`` with values from ``defaults``.
    * Returns a deepcopy of the filled-in ``params``. \n

    Args:
        params (Dict):
            Dictionary of parameters.
        defaults (Dict):
            Dictionary of defaults.
        error_on_missing_keys (bool):
            If ``True``, raises an error when a key in ``params`` is not in
            ``defaults``. If ``False``, emits a warning instead. (Default is
            ``True``)
        verbose (bool):
            If ``True``, prints messages while filling in defaults. (Default
            is ``True``)

    Returns:
        (Dict):
            params_out (Dict):
                Validated and default-filled deepcopy of ``params``.
    """
    from copy import deepcopy
    ## Check inputs
    assert isinstance(params, dict), f"p must be a dict. Got {type(params)} instead."
    ## Make sure all the keys in p are valid
    check_keys_subset(
        d=params, 
        default_dict=defaults,
        error_on_missing_keys=error_on_missing_keys,
    )
    ## Fill in any missing keys with defaults
    params_out = deepcopy(params)
    fill_in_dict(
        d=params_out, 
        defaults=defaults, 
        verbose=verbose,
    )

    return params_out


#####################################################################################################################################
############################################################## VIDEO ################################################################
#####################################################################################################################################

## VideoReaderWrapper is only defined if decord is available.
## When using backend='torchcodec', decord is not required.
if decord is not None:
    class VideoReaderWrapper(decord.VideoReader):
        """
        Subclass of ``decord.VideoReader`` that works around a memory leak.

        Calls ``self.seek(0)`` after initialization and after every
        ``__getitem__`` so that decord releases buffered frames. Adapted from
        https://github.com/dmlc/decord/issues/208#issuecomment-1157632702.

        Attributes:
            path (str):
                Path to the video file (the first positional argument).
        """
        def __init__(self, *args, **kwargs):
            """Forwards all arguments to ``decord.VideoReader`` and resets the seek position."""
            super().__init__(*args, **kwargs)
            self.seek(0)

            self.path = args[0]

        def __getitem__(self, key):
            """Returns frames at ``key`` and resets the underlying decord seek position."""
            frames = super().__getitem__(key)
            self.seek(0)
            return frames
else:
    VideoReaderWrapper = None


def _is_torchcodec_load_error(exc: BaseException) -> bool:
    """Returns ``True`` when ``exc`` indicates a torchcodec shared-library load failure."""
    msg = str(exc)
    markers = (
        "Could not load libtorchcodec",
        "Could not load this library",
        "FFmpeg is not properly installed",
        "libtorchcodec",
        "libavutil",
        "libavcodec",
        "libavformat",
        "Library not loaded: @rpath/libav",
    )
    return any(marker in msg for marker in markers)


def _torchcodec_unavailable_message() -> str:
    """Returns the user-facing error message shown when torchcodec cannot be loaded."""
    return (
        "torchcodec could not be imported or could not load its FFmpeg-linked "
        "shared libraries. Install FFmpeg shared libraries visible to the "
        "dynamic loader, or install torchcodec and ffmpeg from conda-forge "
        "before installing face-rhythm. To use the bundled decord backend "
        "instead, construct your video reader with backend='decord':\n"
        "    BufferedVideoReader(paths_videos=..., backend='decord')"
    )


class TorchCodecVideoReader:
    """
    Video reader backed by ``torchcodec.decoders.VideoDecoder`` with a
    workaround for torchcodec issue #905.

    Provides the same ``__getitem__`` / ``__len__`` / ``get_avg_fps``
    interface as ``VideoReaderWrapper`` (decord) so it can be used as a
    drop-in replacement inside ``BufferedVideoReader``.

    Frames are returned as ``torch.Tensor`` with shape ``(H, W, C)`` and
    dtype ``uint8`` (NHWC layout), matching the output of decord's torch
    bridge.

    **Issue #905 workaround:** torchcodec's sequential access path skips
    cursor reset; after reading ``n - has_b_frames`` frames on a single
    decoder, FFmpeg's H.264 drain emits ``has_b_frames`` AVFrames with
    ``pts = INT64_MIN``, which the internal PTS filter rejects, causing
    ``EndOfFileException`` before the last frames are decoded. The fix is
    to serve only frames ``[0, n - SAFETY)`` from the primary decoder
    (which never reaches the drain), and route the trailing ``SAFETY``
    frames through a fresh decoder that takes the non-sequential seek
    branch (``avformat_seek_file`` + flush + forward-decode from keyframe)
    where ``pkt_dts`` remains valid. ``SAFETY = max(has_b_frames, 2)``.
    torchcodec's ``VideoStreamMetadata`` does not currently expose
    ``has_b_frames``, so SAFETY always defaults to 2, which matches
    ``ffprobe``-reported ``has_b_frames=2`` for H.264 AVIs and is a safe
    overestimate for ``has_b_frames=0`` files.

    The tail decoder is lazily created on first tail access and cached for
    the lifetime of this reader (one decoder recreation per video pass,
    versus one per chunk with earlier workarounds).

    Thread safety is guaranteed by an internal lock — required because
    ``BufferedVideoReader`` loads slots from background threads.

    Args:
        path_video (str):
            Path to the video file.
        device (str):
            Decode device. ``'cpu'`` for software decode,
            ``'cuda'`` or ``'cuda:0'`` for NVDEC hardware decode.
            NVDEC requires torchcodec built with CUDA support and an
            FFmpeg built with ``--enable-cuda``.
        num_ffmpeg_threads (int):
            Number of FFmpeg internal threads for decoding.
            ``0`` lets FFmpeg choose automatically (recommended).
    """
    def __init__(self, path_video: str, device: str = 'cpu', num_ffmpeg_threads: int = 0):
        """Opens ``path_video`` with torchcodec and prepares the issue-#905 tail-decoder cache."""
        import threading

        self.path = str(path_video)
        self._device = device
        self._num_ffmpeg_threads = num_ffmpeg_threads
        self._lock = threading.Lock()

        try:
            self._decoder = self._make_fresh_decoder()
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(_torchcodec_unavailable_message()) from e
        except (OSError, RuntimeError) as e:
            if _is_torchcodec_load_error(e):
                raise ImportError(_torchcodec_unavailable_message()) from e
            raise
        self._num_frames = len(self._decoder)

        ## SAFETY = max(has_b_frames, 2). torchcodec VideoStreamMetadata does
        ## not expose has_b_frames, so getattr always falls back to 2 here.
        ## 2 is correct for standard H.264 main/high AVIs (confirmed by ffprobe)
        ## and is a safe overestimate for has_b_frames=0 files.
        _has_b_frames = getattr(self._decoder.metadata, 'has_b_frames', None)
        self._tail_safety = max(_has_b_frames if _has_b_frames is not None else 2, 2)
        self._safe_last = max(self._num_frames - self._tail_safety, 0)

        ## Tail decoder: lazily created on first tail-range access; cached to
        ## avoid one-decoder-per-call cost on sequential tail reads.
        self._tail_decoder = None

    def _make_fresh_decoder(self):
        """Return a new VideoDecoder for self.path with the configured options."""
        from torchcodec.decoders import VideoDecoder
        return VideoDecoder(
            self.path,
            seek_mode='exact',
            dimension_order='NHWC',
            device=self._device,
            num_ffmpeg_threads=self._num_ffmpeg_threads,
        )

    def _read_one(self, idx: int):
        """
        Read a single frame by index, routing tail frames to a fresh decoder.

        Args:
            idx (int):
                Frame index in ``[0, len(self))``.

        Returns:
            (torch.Tensor):
                frame (torch.Tensor): Shape ``(H, W, C)``, dtype ``uint8``.
        """
        if idx < self._safe_last:
            ## Safe range: primary decoder, sequential access, no drain risk.
            return self._decoder.get_frame_at(idx).data
        ## Tail range: use a fresh decoder (takes the non-sequential seek
        ## branch inside torchcodec), sidestepping the #905 drain bug.
        if self._tail_decoder is None:
            self._tail_decoder = self._make_fresh_decoder()
        return self._tail_decoder.get_frame_at(idx).data

    def __len__(self) -> int:
        return self._num_frames

    def __getitem__(self, key):
        """
        Index frames by int, ``np.integer``, slice, list, or ``np.ndarray``.

        All multi-frame access is implemented via scalar ``get_frame_at``
        calls internally (avoids ``get_frames_at`` / ``get_frames_in_range``,
        which share the buggy sequential access path from issue #905).

        Args:
            key (int | np.integer | slice | list | np.ndarray):
                Frame index or indices.

        Returns:
            (torch.Tensor):
                frames (torch.Tensor):
                    Single frame ``(H, W, C)`` for scalar key, or batch
                    ``(N, H, W, C)`` for slice / list / ndarray key.
        """
        import torch

        with self._lock:
            if isinstance(key, (int, np.integer)):
                return self._read_one(int(key))
            elif isinstance(key, slice):
                indices = list(range(*key.indices(self._num_frames)))
                if not indices:
                    return self._empty_batch()
                return torch.stack([self._read_one(i) for i in indices])
            elif isinstance(key, (list, np.ndarray)):
                indices = [int(i) for i in key]
                if not indices:
                    return self._empty_batch()
                return torch.stack([self._read_one(i) for i in indices])
            else:
                raise TypeError(
                    f"TorchCodecVideoReader: unsupported key type {type(key)}. "
                    "Expected int, np.integer, slice, list, or np.ndarray."
                )

    def get_avg_fps(self) -> float:
        """Return the average frame rate of the video."""
        return self._decoder.metadata.average_fps

    def _empty_batch(self):
        """Return a 4D NHWC empty tensor matching the decoder's frame shape."""
        import torch
        meta = self._decoder.metadata
        return torch.empty((0, meta.height, meta.width, 3), dtype=torch.uint8)


class BufferedVideoReader:
    """
    Reads frames from one or more videos with a chunked memory buffer and
    optional background prefetching. RH 2022

    Sequential batches of frames can be read quickly because buffers are
    filled by background threads. In many cases, batches can be consumed
    without waiting for the next chunk to finish loading.

    Optimal use:

    1. Create a ``BufferedVideoReader`` object.
    2. EITHER set ``method_getitem='continuous'`` and iterate over the object
       (fastest path), OR request batches of frames sequentially (going
       backwards is slow because buffers move forward).
    3. Each batch should fit inside a single buffer slot. Slices that span
       multiple buffer slots require concatenation and are slow. With a
       buffer size of 1000 frames, ``[0:1000], [1000:2000], ...`` is fast,
       while ``[0:1700]``, ``[1700:3200]``, ``[0:990]``, ``[990:1010]`` are
       slow (too big, overlapping, backwards, or crossing slot boundaries).

    Args:
        video_readers (Optional[list]):
            List of video reader objects (``decord.VideoReader`` or
            ``TorchCodecVideoReader``). A single reader is also accepted.
            If ``None``, ``paths_videos`` must be provided. (Default is
            ``None``)
        paths_videos (Optional[list]):
            List of paths to videos. A single ``str`` is also accepted. If
            ``None``, ``video_readers`` must be provided. If both are
            supplied, ``video_readers`` wins. (Default is ``None``)
        buffer_size (int):
            Number of frames per buffer slot. Avoid indexing more than
            ``buffer_size`` frames at a time or across slot boundaries (e.g.
            across ``idx % buffer_size == 0``); these require concatenating
            buffers and are slow. (Default is ``1000``)
        prefetch (int):
            Number of buffers to prefetch ahead. ``0`` disables prefetching.
            A single buffer slot only contains frames from one video, so
            ``buffer_size <= video length`` is recommended. (Default is ``2``)
        posthold (int):
            Number of buffers to keep loaded behind the current position.
            ``0`` disables posthold. Useful when iterating backwards.
            (Default is ``1``)
        method_getitem (str):
            Indexing mode for ``__getitem__``. One of \n
            * ``'continuous'``: index across all videos as a single
              concatenated sequence; ``reader[idx_frames_slice]``.
            * ``'by_video'``: index requires a ``(idx_video, idx_frames)``
              tuple; ``reader[(idx_video, slice)]``. \n
            (Default is ``'continuous'``)
        starting_seek_position (int):
            Starting frame index for the iterator. Used only when
            ``method_getitem == 'continuous'`` and iterating. (Default is
            ``0``)
        backend (str):
            Video decoding backend. One of \n
            * ``'torchcodec'``: uses ``torchcodec.decoders.VideoDecoder``.
              Frame-accurate seeking, actively maintained, supports CPU and
              GPU (NVDEC) decode. Includes a workaround for torchcodec
              issue #905 (sequential-access drain bug near EOF in H.264
              AVIs): frames in ``[0, n - SAFETY)`` come from a persistent
              decoder; the trailing ``SAFETY = max(has_b_frames, 2)`` frames
              go through a fresh decoder cached as the tail decoder.
            * ``'decord'``: uses ``decord.VideoReader``. Well-tested
              fallback and the only backend available on Windows. Provided
              by the ``decord2`` PyPI package on Linux/macOS (with vendored
              FFmpeg 8 wheels for py3.10-3.14) and by ``eva_decord`` on
              Windows. Both are installed by face-rhythm's default
              dependencies. \n
            Only used when ``paths_videos`` is provided. (Default is
            ``'torchcodec'``)
        device (str):
            Device for video decoding when using torchcodec. ``'cpu'``
            decodes on CPU. ``'cuda'`` or ``'cuda:0'`` decodes on GPU using
            NVDEC; frames are returned as CUDA tensors. GPU decode requires
            an NVIDIA GPU, torchcodec installed with CUDA support, and
            FFmpeg built with ``--enable-cuda``. (Default is ``'cpu'``)
        decord_backend (str):
            Backend used by decord when loading frames (``'torch'``,
            ``'numpy'``, ``'mxnet'``, ...). Only used when
            ``backend='decord'``. (Default is ``'torch'``)
        decord_ctx (object):
            Context used by decord when loading frames (e.g. ``decord.cpu()``,
            ``decord.gpu()``). Only used when ``backend='decord'``. (Default
            is ``None``)
        verbose (int):
            Verbosity level. ``0`` silences output, ``1`` prints warnings,
            ``2`` prints warnings and info. (Default is ``1``)

    Attributes:
        num_frames_total (int):
            Total number of frames across all videos.
        num_videos (int):
            Number of videos being read.
        metadata (pandas.DataFrame):
            Per-video metadata (path, length, fps, frame size, channels).
        frame_rate (List[float]):
            Frame rate of each video.
        frame_height_width (List[Tuple[int, int]]):
            ``(H, W)`` of each video.
        num_channels (List[int]):
            Number of channels of each video.
        slots (List[List[Optional[torch.Tensor]]]):
            Buffer slots holding chunks of decoded frames.
        boundaries (List[List[Tuple[int, int]]]):
            Inclusive ``(start, end)`` frame index for each slot.
        lookup (pandas.DataFrame):
            Lookup table mapping continuous frame index to ``(video, slot)``.
    """
    def __init__(
        self,
        video_readers: list=None,
        paths_videos: list=None,
        buffer_size: int=1000,
        prefetch: int=2,
        posthold: int=1,
        method_getitem: str='continuous',
        starting_seek_position: int=0,
        backend: str='torchcodec',
        device: str='cpu',
        decord_backend: str='torch',
        decord_ctx=None,
        verbose: int=1,
    ):
        """Initializes the reader, opens the videos, and prepares buffer slots."""
        import pandas as pd

        self._verbose = verbose
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.posthold = posthold
        self._backend = backend
        self._device = device
        self._decord_backend = decord_backend
        self._decord_ctx = (decord.cpu(0) if decord_ctx is None else decord_ctx) if decord is not None else None

        ## Check inputs
        _single_reader_types = (TorchCodecVideoReader,)
        if decord is not None:
            _single_reader_types = (decord.VideoReader, TorchCodecVideoReader)
        if isinstance(video_readers, _single_reader_types):
            video_readers = [video_readers]
        if isinstance(paths_videos, str):
            paths_videos = [paths_videos]
        assert (video_readers is not None) or (paths_videos is not None), "Must provide either video_readers or paths_videos"

        ## If both video_readers and paths_videos are provided, use the video_readers and print a warning
        if (video_readers is not None) and (paths_videos is not None):
            print(f"FR WARNING: Both video_readers and paths_videos were provided. Using video_readers and ignoring path_videos.")
            paths_videos = None
        ## If paths are specified, create video reader objects
        if paths_videos is not None:
            print(f"FR: Loading video reader objects (backend='{self._backend}')...") if self._verbose > 1 else None
            assert isinstance(paths_videos, list), "paths_videos must be list of str"
            assert all([isinstance(p, str) for p in paths_videos]), "paths_videos must be list of str"
            if self._backend == 'torchcodec':
                print(f"FR: Video decode device: {self._device}") if self._verbose > 1 else None
                try:
                    video_readers = [TorchCodecVideoReader(path_video, device=self._device) for path_video in tqdm(paths_videos, disable=(self._verbose < 2))]
                except (ImportError, ModuleNotFoundError) as e:
                    raise ImportError(_torchcodec_unavailable_message()) from e
                except (OSError, RuntimeError) as e:
                    if _is_torchcodec_load_error(e):
                        raise ImportError(_torchcodec_unavailable_message()) from e
                    raise
            elif self._backend == 'decord':
                assert decord is not None, (
                    "FR ERROR: decord is not installed (this is unexpected — "
                    "decord2/eva_decord are face-rhythm required dependencies). "
                    "Install with: pip install decord2  (or eva_decord on Windows)"
                )
                video_readers = [VideoReaderWrapper(path_video, ctx=self._decord_ctx) for path_video in tqdm(paths_videos, disable=(self._verbose < 2))]
            else:
                raise ValueError(f"FR ERROR: Unknown video backend '{self._backend}'. Use 'torchcodec' or 'decord'.")
            self.paths_videos = paths_videos
        else:
            print(f"FR: Using provided video reader objects...") if self._verbose > 1 else None
            assert isinstance(video_readers, list), "video_readers must be a list of video reader objects"
            self.paths_videos = [v.path for v in video_readers]
        ## Assert that method_getitem is valid
        assert method_getitem in ['continuous', 'by_video'], "method_getitem must be 'continuous' or 'by_video'"
        ## Set decord bridge if using decord backend
        if self._backend == 'decord':
            decord.bridge.set_bridge(self._decord_backend)

        self.paths_videos = [str(path) for path in self.paths_videos]  ## ensure paths are str
        self.video_readers = video_readers
        self._cumulative_frame_end = np.cumsum([len(video_reader) for video_reader in self.video_readers])
        self._cumulative_frame_start = np.concatenate([[0], self._cumulative_frame_end[:-1]])
        self.num_frames_total = self._cumulative_frame_end[-1]
        self.method_getitem = method_getitem

        ## Get metadata about videos: lengths, fps, frame size, etc.
        self.metadata, self.num_frames_total, self.frame_rate, self.frame_height_width, self.num_channels = self._get_metadata(self.video_readers)
        ## Get number of videos
        self.num_videos = len(self.video_readers)

        ## Set iterator starting frame
        print(f"FR: Setting iterator starting frame to {starting_seek_position}") if self._verbose > 1 else None
        self.set_iterator_frame_idx(starting_seek_position)

        ## Initialize the buffer
        ### Make a list containing a slot for each buffer chunk
        self.slots = [[None] * np.ceil(len(d)/self.buffer_size).astype(int) for d in self.video_readers]
        ### Make a list containing the bounding indices for each buffer video chunk. Upper bound should be min(buffer_size, num_frames)
        self.boundaries = [[(i*self.buffer_size, min((i+1)*self.buffer_size, len(d))-1) for i in range(len(s))] for d, s in zip(self.video_readers, self.slots)]
        ### Make a lookup table for the buffer slot that contains each frame
        self.lookup = {
            'video': np.concatenate([np.array([ii]*len(s), dtype=int) for ii, s in enumerate(self.slots)]).tolist(),
            'slot': np.concatenate([np.arange(len(s)) for s in self.slots]).tolist(),
            'start_frame': np.concatenate([np.array([s[0] for s in b]) for b in self.boundaries]).astype(int).tolist(), 
            'end_frame': np.concatenate([np.array([s[1] for s in b]) for b in self.boundaries]).astype(int).tolist(),
        }
        self.lookup['start_frame_continuous'] = (np.array(self.lookup['start_frame']) + np.array(self._cumulative_frame_start[self.lookup['video']])).tolist()
        self.lookup = pd.DataFrame(self.lookup)
        self._start_frame_continuous = self.lookup['start_frame_continuous'].values

        ## Make a list for which slots are loaded or loading
        self.loading = []
        self.loaded = []


    def _get_metadata(self, video_readers):
        """
        Collects per-video metadata: length, fps, frame size, channels.

        Args:
            video_readers (List[object]):
                List of decoder objects (``decord.VideoReader`` or
                ``TorchCodecVideoReader``).

        Returns:
            (tuple): tuple containing:
                metadata (Dict[str, list]):
                    Per-video metadata with keys ``'paths_videos'``,
                    ``'num_frames'``, ``'frame_rate'``,
                    ``'frame_height_width'``, ``'num_channels'``.
                num_frames_total (int):
                    Total number of frames across all videos.
                frame_rate (float):
                    Median frame rate across videos.
                frame_height_width (List[int]):
                    Common ``[height, width]`` of all videos.
                num_channels (int):
                    Common channel count across videos.
        """

        ## make video metadata dataframe
        print("FR: Collecting video metadata...") if self._verbose > 1 else None
        metadata = {"paths_videos": self.paths_videos}
        num_frames, frame_rate, frame_height_width, num_channels = [], [], [], []
        for v in tqdm(video_readers, disable=(self._verbose < 2)):
            num_frames.append(int(len(v)))
            frame_rate.append(float(v.get_avg_fps()))
            frame_tmp = v[0]
            frame_height_width.append([int(n) for n in frame_tmp.shape[:2]])
            num_channels.append(int(frame_tmp.shape[2]))
        metadata["num_frames"] = num_frames
        metadata["frame_rate"] = frame_rate
        metadata["frame_height_width"] = frame_height_width
        metadata["num_channels"] = num_channels
            

        ## Assert that all videos must have at least one frame
        assert all([n > 0 for n in metadata["num_frames"]]), "FR ERROR: All videos must have at least one frame"
        ## Assert that all videos must have the same shape
        assert all([n == metadata["frame_height_width"][0] for n in metadata["frame_height_width"]]), "FR ERROR: All videos must have the same shape"
        ## Assert that all videos must have the same number of channels
        assert all([n == metadata["num_channels"][0] for n in metadata["num_channels"]]), "FR ERROR: All videos must have the same number of channels"

        ## get frame rate
        frame_rates = metadata["frame_rate"]
        ## warn if any video's frame rate is very different from others
        max_diff = float((np.max(frame_rates) - np.min(frame_rates)) / np.mean(frame_rates))
        print(f"FR WARNING: max frame rate difference is large: {max_diff*100:.2f}%") if ((max_diff > 0.1) and (self._verbose > 0)) else None
        frame_rate = float(np.median(frame_rates))

        num_frames_total = int(np.sum(metadata["num_frames"]))
        frame_height_width = metadata["frame_height_width"][0]
        num_channels = metadata["num_channels"][0]

        return metadata, num_frames_total, frame_rate, frame_height_width, num_channels


    def _load_slots(self, idx_slots: list, wait_for_load: Union[bool, list]=False):
        """
        Loads buffer slots in the background using threading.

        Args:
            idx_slots (list):
                List of ``(idx_video, idx_buffer)`` tuples identifying the
                slots to load.
            wait_for_load (Union[bool, List[bool]]):
                If ``True``, blocks until each slot is loaded before
                returning. If ``False``, returns immediately. If a list, each
                entry corresponds to the slot at the same position in
                ``idx_slots``. (Default is ``False``)
        """
        ## Check if idx_slots is a list
        if not isinstance(idx_slots, list):
            idx_slots = [idx_slots]

        ## Check if wait_for_load is a list
        if not isinstance(wait_for_load, list):
            wait_for_load = [wait_for_load] * len(idx_slots)

        print(f"FR: Loading slots {idx_slots} in the background. Waiting: {wait_for_load}") if self._verbose > 1 else None
        print(f"FR: Loaded: {self.loaded}, Loading: {self.loading}") if self._verbose > 1 else None
        thread = None
        for idx_slot, wait in zip(idx_slots, wait_for_load):
            ## Check if slot is already loaded
            (print(f"FR: Slot {idx_slot} already loaded") if (idx_slot in self.loaded) else None) if self._verbose > 1 else None
            (print(f"FR: Slot {idx_slot} already loading") if (idx_slot in self.loading) else None) if self._verbose > 1 else None
            ## If the slot is not already loaded or loading
            if (idx_slot not in self.loading) and (idx_slot not in self.loaded):
                print(f"FR: Loading slot {idx_slot}") if self._verbose > 1 else None
                ## Load the slot
                self.loading.append(idx_slot)
                thread = threading.Thread(target=self._load_slot, args=(idx_slot, thread))
                thread.start()

                ## Wait for the slot to load if wait_for_load is True
                if wait:
                    print(f"FR: Waiting for slot {idx_slot} to load") if self._verbose > 1 else None
                    thread.join()
                    print(f"FR: Slot {idx_slot} loaded") if self._verbose > 1 else None
            ## If the slot is already loading
            elif idx_slot in self.loading:
                ## Wait for the slot to load if wait_for_load is True
                if wait:
                    print(f"FR: Waiting for slot {idx_slot} to load") if self._verbose > 1 else None
                    while idx_slot in self.loading:
                        time.sleep(0.01)
                    print(f"FR: Slot {idx_slot} loaded") if self._verbose > 1 else None

    def _load_slot(self, idx_slot: tuple, blocking_thread: threading.Thread=None):
        """
        Loads a single buffer slot, optionally after another thread finishes.

        Populates ``self.slots[idx_slot[0]][idx_slot[1]]`` with decoded
        frames. If ``blocking_thread`` is provided, this call ``join()`` s
        that thread before reading, ensuring serial decode order.

        Args:
            idx_slot (Tuple[int, int]):
                ``(idx_video, idx_buffer)`` identifying the slot to load.
            blocking_thread (Optional[threading.Thread]):
                Thread to wait on before loading. (Default is ``None``)
        """
        ## Set backend of decord to PyTorch (only needed for decord)
        if self._backend == 'decord':
            decord.bridge.set_bridge(self._decord_backend)
        ## Wait for the previous slot to finish loading
        if blocking_thread is not None:
            blocking_thread.join()
        ## Load the slot
        idx_video, idx_buffer = idx_slot
        idx_frame_start, idx_frame_end = self.boundaries[idx_video][idx_buffer]
        loaded = False
        while loaded == False:
            try:
                self.slots[idx_video][idx_buffer] = self.video_readers[idx_video][idx_frame_start:idx_frame_end+1]
                loaded = True
            except Exception as e:
                print(f"FR WARNING: Failed to load slot {idx_slot}. Likely causes are: 1) File is partially corrupted, 2) You are trying to go back to a file that was recently removed from a slot.") if self._verbose > 0 else None
                print(f"    Sleeping for 1s, then will try loading again. Decord error below:") if self._verbose > 0 else None
                print(e)
                time.sleep(1)

        ## Mark the slot as loaded
        self.loaded.append(idx_slot)
        ## Remove the slot from the loading list
        self.loading.remove(idx_slot)
                
    def _delete_slots(self, idx_slots: list):
        """
        Frees buffer slots by setting their contents to ``None``.

        Args:
            idx_slots (List[Tuple[int, int]]):
                ``(idx_video, idx_buffer)`` tuples identifying the slots to
                delete.
        """
        print(f"FR: Deleting slots {idx_slots}") if self._verbose > 1 else None
        ## Find all loaded slots
        idx_loaded = [idx_slot for idx_slot in idx_slots if idx_slot in self.loaded]
        for idx_slot in idx_loaded:
            ## If the slot is loaded
            if idx_slot in self.loaded:
                ## Delete the slot
                self.slots[idx_slot[0]][idx_slot[1]] = None
                ## Remove the slot from the loaded list
                self.loaded.remove(idx_slot)
                print(f"FR: Deleted slot {idx_slot}") if self._verbose > 1 else None

    def delete_all_slots(self):
        """Frees every currently loaded slot by delegating to ``_delete_slots``."""
        print(f"FR: Deleting all slots") if self._verbose > 1 else None
        self._delete_slots(self.loaded)

    def wait_for_loading(self):
        """Blocks until every background slot-loading thread has finished."""
        print(f"FR: Waiting for all slots to load") if self._verbose > 1 else None
        while len(self.loading) > 0:
            time.sleep(0.01)
        

    
    def get_frames_from_single_video_index(self, idx: tuple):
        """
        Returns frames from a single video by ``(video, frame)`` index.

        If ``idx`` is an ``int`` or ``slice`` it is interpreted as a video
        index and a new ``BufferedVideoReader`` is constructed over the
        selected videos.

        Args:
            idx (Union[int, slice, Tuple[int, Union[int, slice]]]):
                Either ``(idx_video, idx_frames)`` to read frames from one
                video, or an ``int`` / ``slice`` to spawn a reader over a
                subset of videos.

        Returns:
            (Union[torch.Tensor, BufferedVideoReader]):
                frames (Union[torch.Tensor, BufferedVideoReader]):
                    Decoded frames with shape *(num_frames, H, W, C)* when
                    ``idx`` is a tuple, or a new reader when ``idx`` selects
                    videos.
        """
        ## if idx is an int or slice, use idx to make a new BufferedVideoReader of just those videos
        idx = slice(idx, idx+1) if isinstance(idx, int) else idx
        if isinstance(idx, slice):
            ## convert to a slice
            print(f"FR: Returning new buffered video reader(s). Videos={idx.start} to {idx.stop}.") if self._verbose > 1 else None
            return BufferedVideoReader(
                video_readers=self.video_readers[idx],
                buffer_size=self.buffer_size,
                prefetch=self.prefetch,
                method_getitem='continuous',
                starting_seek_position=0,
                backend=self._backend,
                device=self._device,
                decord_backend=self._decord_backend,
                decord_ctx=self._decord_ctx,
                verbose=self._verbose,
            )
        print(f"FR: Getting item {idx}") if self._verbose > 1 else None
        ## Assert that idx is a tuple of (int, int) or (int, slice)
        assert isinstance(idx, tuple), f"idx must be: int, tuple of (int, int), or (int, slice). Got {type(idx)}"
        assert len(idx) == 2, f"idx must be: int, tuple of (int, int), or (int, slice). Got {len(idx)} elements"
        assert isinstance(idx[0], int), f"idx[0] must be an int. Got {type(idx[0])}"
        assert isinstance(idx[1], int) or isinstance(idx[1], slice), f"idx[1] must be an int or a slice. Got {type(idx[1])}"
        ## Get the index of the video and the slice of frames
        idx_video, idx_frames = idx
        ## If idx_frames is a single integer, convert it to a slice
        idx_frames = slice(idx_frames, idx_frames+1) if isinstance(idx_frames, int) else idx_frames
        ## Bound the range of the slice
        idx_frames = slice(max(idx_frames.start, 0), min(idx_frames.stop, len(self.video_readers[idx_video])))
        ## Assert that slice is not empty
        assert idx_frames.start < idx_frames.stop, f"Slice is empty: idx:{idx}"

        ## Get the start and end indices for the slice of frames
        idx_frame_start = idx_frames.start if idx_frames.start is not None else 0
        idx_frame_end = idx_frames.stop if idx_frames.stop is not None else len(self.video_readers[idx_video])
        idx_frame_step = idx_frames.step if idx_frames.step is not None else 1

        ## Get the indices of the slots that contain the frames
        idx_slots = [(idx_video, i) for i in range(idx_frame_start // self.buffer_size, ((idx_frame_end-1) // self.buffer_size)+1)]
        print(f"FR: Slots to load: {idx_slots}") if self._verbose > 1 else None

        ## Load the prefetch slots
        idx_slot_lookuptable = np.where((self.lookup['video']==idx_slots[-1][0]) * (self.lookup['slot']==idx_slots[-1][1]))[0][0]
        if self.prefetch > 0:
            idx_slots_prefetch = [(self.lookup['video'][ii], self.lookup['slot'][ii]) for ii in range(idx_slot_lookuptable+1, idx_slot_lookuptable+self.prefetch+1) if ii < len(self.lookup)]
        else:
            idx_slots_prefetch = []
        ## Load the slots
        self._load_slots(idx_slots + idx_slots_prefetch, wait_for_load=[True]*len(idx_slots) + [False]*len(idx_slots_prefetch))
        ## Delete the slots that are no longer needed. 
        ### Find slots before the posthold to delete
        idx_slots_delete = [(self.lookup['video'][ii], self.lookup['slot'][ii]) for ii in range(idx_slot_lookuptable-self.posthold) if ii >= 0]
        ### Delete all previous slots
        self._delete_slots(idx_slots_delete)
        # ### All slots from old videos should be deleted.
        # self._delete_slots([idx_slot for idx_slot in self.loaded if idx_slot[0] < idx_video])
        # ### All slots from previous buffers should be deleted.
        # self._delete_slots([idx_slot for idx_slot in self.loaded if idx_slot[0] == idx_video and idx_slot[1] < idx_frame_start // self.buffer_size])

        ## Get the frames from the slots
        idx_frames_slots = [slice(max(idx_frame_start - self.boundaries[idx_slot[0]][idx_slot[1]][0], 0), min(idx_frame_end - self.boundaries[idx_slot[0]][idx_slot[1]][0], self.buffer_size), idx_frame_step) for idx_slot in idx_slots]
        print(f"FR: Frames within slots: {idx_frames_slots}") if self._verbose > 1 else None

        ## Get the frames. Then concatenate them along the first dimension using torch.cat
        ### Skip the concatenation if there is only one slot
        if len(idx_slots) == 1:
            frames = self.slots[idx_slots[0][0]][idx_slots[0][1]][idx_frames_slots[0]]
        else:
            print(f"FR: Warning. Slicing across multiple slots is SLOW. Consider increasing buffer size or adjusting batching method.") if self._verbose > 1 else None
            frames = torch.cat([self.slots[idx_slot[0]][idx_slot[1]][idx_frames_slot] for idx_slot, idx_frames_slot in zip(idx_slots, idx_frames_slots)], dim=0)
        
        # ## Squeeze if there is only one frame
        # frames = frames.squeeze(0) if frames.shape[0] == 1 else frames

        return frames

    def get_frames_from_continuous_index(self, idx):
        """
        Returns frames addressed by a continuous (concatenated) frame index.

        The videos are treated as one long sequence of frames; ``idx`` is the
        index of the frames within this sequence.

        Args:
            idx (Union[int, slice]):
                Frame index. If an ``int``, a single frame is returned. If a
                ``slice``, the corresponding batch of frames is returned.

        Returns:
            (torch.Tensor):
                frames (torch.Tensor):
                    Stacked frames. shape: *(num_frames, height, width, num_channels)*.
        """
        ## Assert that idx is an int or a slice
        assert isinstance(idx, (int, np.intp)) or isinstance(idx, slice), f"idx must be an int or a slice. Got {type(idx)}"
        idx = int(idx) if isinstance(idx, (np.intp)) else idx
        ## If idx is a single integer, convert it to a slice
        idx = slice(idx, idx+1) if isinstance(idx, int) else idx
        ## Assert that the slice is not empty
        assert idx.start < idx.stop, f"Slice is empty: idx:{idx}"
        ## Assert that the slice is not out of bounds
        assert idx.stop <= self.num_frames_total, f"Slice is out of bounds: idx:{idx}"
        
        ## Find the video and frame indices
        idx_video_start = np.searchsorted(self._cumulative_frame_start, idx.start, side='right') - 1
        idx_video_end = np.searchsorted(self._cumulative_frame_end, idx.stop, side='left')
        ## Get the frames using the __getitem__ method
        ### This needs to be done one video at a time
        frames = []
        for idx_video in range(idx_video_start, idx_video_end+1):
            ## Get the start and end indices for the slice of frames
            idx_frame_start = idx.start - self._cumulative_frame_start[idx_video] if idx_video == idx_video_start else 0
            idx_frame_end = idx.stop - self._cumulative_frame_start[idx_video] if idx_video == idx_video_end else len(self.video_readers[idx_video])
            ## Get the frames
            print(f"FR: Getting frames from video {idx_video} from {idx_frame_start} to {idx_frame_end}") if self._verbose > 1 else None
            frames.append(self.get_frames_from_single_video_index((idx_video, slice(idx_frame_start, idx_frame_end, idx.step))))
        ## Concatenate the frames if there are multiple videos
        frames = torch.cat(frames, dim=0) if len(frames) > 1 else frames[0]

        return frames

    def set_iterator_frame_idx(self, idx):
        """
        Sets the starting frame for the iterator.

        Args:
            idx (int):
                Frame index from which the iterator should start. Must be in
                ``'continuous'`` format, i.e. the index of the frame within
                the concatenated sequence of all videos.
        """
        self._iterator_frame = idx
        
    def __getitem__(self, idx):
        """Dispatches to ``get_frames_from_single_video_index`` or ``get_frames_from_continuous_index`` based on ``method_getitem``."""
        if self.method_getitem == 'by_video':
            return self.get_frames_from_single_video_index(idx)
        elif self.method_getitem == 'continuous':
            return self.get_frames_from_continuous_index(idx)
        else:
            raise ValueError(f"Invalid method_getitem: {self.method_getitem}")

    def __len__(self):
        """Returns the number of videos (``'by_video'``) or the total frame count (``'continuous'``)."""
        if self.method_getitem == 'by_video':
            return len(self.video_readers)
        elif self.method_getitem == 'continuous':
            return self.num_frames_total
    def __repr__(self):
        """Returns a debug string summarizing buffer size, video count, prefetch state, and verbosity."""
        if self.method_getitem == 'by_video':
            return f"BufferedVideoReader(buffer_size={self.buffer_size}, num_videos={self.num_videos}, method_getitem='{self.method_getitem}', loaded={self.loaded}, prefetch={self.prefetch}, loading={self.loading}, verbose={self._verbose})"
        elif self.method_getitem == 'continuous':
            return f"BufferedVideoReader(buffer_size={self.buffer_size}, num_videos={self.num_videos}, total_frames={self.num_frames_total}, method_getitem='{self.method_getitem}', iterator_frame={self._iterator_frame}, prefetch={self.prefetch}, loaded={self.loaded}, loading={self.loading}, verbose={self._verbose})"
    def __iter__(self):
        """
        Iterates over the contents according to ``method_getitem``.

        When ``method_getitem == 'by_video'``, yields a fresh
        ``BufferedVideoReader`` for each underlying video. When
        ``method_getitem == 'continuous'``, yields single frames pulled
        directly from buffer slots, calling
        ``get_frames_from_continuous_index`` whenever a new slot must be
        loaded.

        Returns:
            (Iterator):
                iterator (Iterator):
                    Iterator over readers (``'by_video'``) or frames
                    (``'continuous'``).
        """
        if self.method_getitem == 'by_video':
            return iter([BufferedVideoReader(
                video_readers=[self.video_readers[idx]],
                buffer_size=self.buffer_size,
                prefetch=self.prefetch,
                method_getitem='continuous',
                starting_seek_position=0,
                backend=self._backend,
                device=self._device,
                decord_backend=self._decord_backend,
                decord_ctx=self._decord_ctx,
                verbose=self._verbose,
            ) for idx in range(len(self.video_readers))])
        elif self.method_getitem == 'continuous':
            ## Initialise the buffers by loading the first frame in the sequence
            self.get_frames_from_continuous_index(self._iterator_frame)
            ## Make lazy iterator over all frames
            def lazy_iterator():
                while self._iterator_frame < self.num_frames_total:
                    ## Find slot for current frame idx
                    idx_video = np.searchsorted(self._cumulative_frame_start, self._iterator_frame, side='right') - 1
                    idx_slot_in_video = (self._iterator_frame - self._cumulative_frame_start[idx_video]) // self.buffer_size
                    idx_frame = self._iterator_frame - self._cumulative_frame_start[idx_video]
                    ## If the frame is at the beginning of a slot, then use get_frames_from_single_video_index otherwise just grab directly from the slot
                    if (self._iterator_frame in self._start_frame_continuous):
                        yield self.get_frames_from_continuous_index(self._iterator_frame)[0]
                    else:
                    ## Get the frame directly from the slot
                        yield self.slots[idx_video][idx_slot_in_video][idx_frame%self.buffer_size]
                    self._iterator_frame += 1
        return iter(lazy_iterator())



def save_gif(
    array,
    path,
    frameRate=5.0,
    loop=0,
    backend='PIL',
    kwargs_backend={},
):
    """
    Saves an array of images as an animated GIF. RH 2023

    Args:
        array (Union[np.ndarray, list]):
            3D (grayscale) or 4D (color) array of images. If dtype is
            floating, values are interpreted in *[0, 1]*; if integer, in
            *[0, 255]*.
        path (str):
            Output path for the GIF.
        frameRate (float):
            Frame rate of the GIF in frames per second. (Default is ``5.0``)
        loop (int):
            Number of loops. ``0`` loops forever, ``1`` plays once, ``2``
            plays twice, etc. (Default is ``0``)
        backend (str):
            GIF writer backend. One of \n
            * ``'imageio'``
            * ``'PIL'`` \n
            (Default is ``'PIL'``)
        kwargs_backend (dict):
            Extra keyword arguments forwarded to the chosen backend.
            (Default is ``{}``)
    """
    array = np.stack(array, axis=0) if isinstance(array, list) else array
    array = grayscale_to_rgb(array) if array.ndim == 3 else array
    if np.issubdtype(array.dtype, np.floating):
        array = (array*255).astype('uint8')
    
    kwargs_backend.update({'loop': loop} if loop != 1 else {})

    if backend == 'imageio':
        import imageio
        imageio.mimsave(
            path, 
            array, 
            format='GIF',
            duration=1000/frameRate, 
            **kwargs_backend,
        )
    elif backend == 'PIL':
        from PIL import Image
        frames = [Image.fromarray(array[i_frame]) for i_frame in range(array.shape[0])]
        frames[0].save(
            path, 
            format='GIF', 
            append_images=frames[1:], 
            save_all=True, 
            duration=1000/frameRate, 
            **kwargs_backend,
        )
    else:
        raise Exception(f'Unsupported backend {backend}')


def grayscale_to_rgb(array):
    """
    Converts a grayscale image or movie to RGB by repeating the channel. RH 2023

    Args:
        array (Union[np.ndarray, torch.Tensor, list]):
            2D image or 3D movie of grayscale frames. Lists of arrays or
            tensors are stacked first.

    Returns:
        (Union[np.ndarray, torch.Tensor]):
            rgb (Union[np.ndarray, torch.Tensor]):
                Same backend as the input with an extra trailing channel
                dimension of size 3.
    """
    import torch
    if isinstance(array, list):
        if isinstance(array[0], np.ndarray):
            array = np.stack(array, axis=0)
        elif isinstance(array[0], torch.Tensor):
            array = torch.stack(array, axis=0)
        else:
            raise Exception(f'Failed to convert list of type {type(array[0])} to array')
    if isinstance(array, np.ndarray):
        return np.stack([array, array, array], axis=-1)
    elif isinstance(array, torch.Tensor):
        return torch.stack([array, array, array], dim=-1)
    
    
########################################################################################################################################
############################################################# CONVOLUTION ##############################################################
########################################################################################################################################

class Toeplitz_convolution2d:
    """
    Convolves a 2D array with a 2D kernel via Toeplitz matrix multiplication.
    RH 2022

    Allows **sparse** ``x`` inputs (``k`` must remain dense). Ideal when
    ``x`` is very sparse (density < 0.01), ``x`` is small (shape <
    *(1000, 1000)*), ``k`` is small (shape < *(100, 100)*), and the batch
    size is large (e.g. 1000+). Generally faster than
    ``scipy.signal.convolve2d`` when convolving many arrays with the same
    kernel. Memory footprint stays low because the Toeplitz matrix is held
    as a sparse matrix.

    See https://stackoverflow.com/a/51865516 and
    https://github.com/alisaaalehi/convolution_as_multiplication for an
    illustration. See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html
    for the 1D version, and
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html
    for potential speedups.

    Args:
        x_shape (Tuple[int, int]):
            Shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Convolution mode. One of \n
            * ``'full'``
            * ``'same'``
            * ``'valid'`` \n
            See ``scipy.signal.convolve2d`` for details. (Default is ``'same'``)
        dtype (Optional[np.dtype]):
            Data type for the Toeplitz matrix. Ideally matches the dtype of
            the input array. If ``None``, the dtype of ``k`` is used.
            (Default is ``None``)

    Attributes:
        k (np.ndarray):
            Flipped copy of the kernel used internally.
        mode (str):
            Convolution mode set in ``__init__``.
        x_shape (Tuple[int, int]):
            Stored input shape.
        dtype (np.dtype):
            Data type of the Toeplitz matrix.
        so (Tuple[int, int]):
            Output array size before cropping.
        dt (scipy.sparse.csr_matrix):
            The double-block Toeplitz matrix in sparse CSR form.

    Example:
        .. highlight:: python
        .. code-block:: python

            conv = Toeplitz_convolution2d(x_shape=x.shape, k=kernel, mode='same')
            y = conv(x)
    """
    def __init__(
        self,
        x_shape,
        k,
        mode='same',
        dtype=None,
    ):
        """Builds and caches the double-block Toeplitz matrix for the given shape and kernel."""
        self.k = k = np.flipud(k.copy())
        self.mode = mode
        self.x_shape = x_shape
        self.dtype = k.dtype if dtype is None else dtype

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"

        self.so = so = size_output_array = ( (k.shape[0] + x_shape[0] -1), (k.shape[1] + x_shape[1] -1))  ## 'size out' is the size of the output array

        ## make the toeplitz matrices
        t = toeplitz_matrices = [scipy.sparse.diags(
            diagonals=np.ones((k.shape[1], x_shape[1]), dtype=self.dtype) * k_i[::-1][:,None], 
            offsets=np.arange(-k.shape[1]+1, 1), 
            shape=(so[1], x_shape[1]),
            dtype=self.dtype,
        ) for k_i in k[::-1]]  ## make the toeplitz matrices for the rows of the kernel
        tc = toeplitz_concatenated = scipy.sparse.vstack(t + [scipy.sparse.dia_matrix((t[0].shape), dtype=self.dtype)]*(x_shape[0]-1))  ## add empty matrices to the bottom of the block due to padding, then concatenate

        ## make the double block toeplitz matrix
        self.dt = double_toeplitz = scipy.sparse.hstack([self._roll_sparse(
            x=tc, 
            shift=(ii>0)*ii*(so[1])  ## shift the blocks by the size of the output array
        ) for ii in range(x_shape[0])]).tocsr()
    
    def __call__(
        self,
        x,
        batching=True,
        mode=None,
    ):
        """
        Convolves an input array (or batch of arrays) with the stored kernel.

        Args:
            x (Union[np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]):
                Input array(s) to convolve. When ``batching`` is ``False``,
                a single 2D array of shape *(self.x_shape[0],
                self.x_shape[1])*. When ``batching`` is ``True``, multiple
                2D arrays that have been flattened into row vectors with
                shape *(n_arrays, self.x_shape[0] * self.x_shape[1])*.
            batching (bool):
                If ``False``, ``x`` is a single 2D array. If ``True``, ``x``
                is a 2D array whose rows are flattened 2D inputs. (Default
                is ``True``)
            mode (Optional[str]):
                Convolution mode. One of \n
                * ``'full'``
                * ``'same'``
                * ``'valid'`` \n
                Overrides the mode set in ``__init__`` when provided.
                (Default is ``None``)

        Returns:
            (Union[np.ndarray, scipy.sparse.csr_matrix]):
                out (Union[np.ndarray, scipy.sparse.csr_matrix]):
                    When ``batching`` is ``True``, multiple convolved 2D
                    arrays flattened into row vectors with shape
                    *(n_arrays, height * width)*. When ``batching`` is
                    ``False``, a single convolved 2D array with shape
                    *(height, width)*.
        """
        if mode is None:
            mode = self.mode  ## use the mode that was set in the init if not specified
        issparse = scipy.sparse.issparse(x)
        
        if batching:
            x_v = x.T  ## transpose into column vectors
        else:
            x_v = x.reshape(-1, 1)  ## reshape 2D array into a column vector
        
        if issparse:
            x_v = x_v.tocsc()
        
        out_v = self.dt @ x_v  ## if sparse, then 'out_v' will be a csc matrix
            
        ## crop the output to the correct size
        if mode == 'full':
            p_t = 0
            p_b = self.so[0]+1
            p_l = 0
            p_r = self.so[1]+1
        if mode == 'same':
            p_t = (self.k.shape[0]-1)//2
            p_b = -(self.k.shape[0]-1)//2
            p_l = (self.k.shape[1]-1)//2
            p_r = -(self.k.shape[1]-1)//2

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        if mode == 'valid':
            p_t = (self.k.shape[0]-1)
            p_b = -(self.k.shape[0]-1)
            p_l = (self.k.shape[1]-1)
            p_r = -(self.k.shape[1]-1)

            p_b = self.x_shape[0]+1 if p_b==0 else p_b
            p_r = self.x_shape[1]+1 if p_r==0 else p_r
        
        if batching:
            idx_crop = np.zeros((self.so), dtype=np.bool_)
            idx_crop[p_t:p_b, p_l:p_r] = True
            idx_crop = idx_crop.reshape(-1)
            out = out_v[idx_crop,:].T
        else:
            if issparse:
                out = out_v.reshape((self.so)).tocsc()[p_t:p_b, p_l:p_r]
            else:
                out = out_v.reshape((self.so))[p_t:p_b, p_l:p_r]  ## reshape back into 2D array and crop
        return out
    
    def _roll_sparse(
        self,
        x,
        shift,
    ):
        """
        Shifts the row indices of a sparse matrix by ``shift`` (no wrap).

        Args:
            x (scipy.sparse.coo_matrix):
                Sparse matrix to shift.
            shift (int):
                Row offset to add to ``x.row``.

        Returns:
            (scipy.sparse.coo_matrix):
                out (scipy.sparse.coo_matrix):
                    Copy of ``x`` with ``row`` shifted by ``shift``.
        """
        out = x.copy()
        out.row += shift
        return out

def cosine_kernel_2D(center=(5,5), image_size=(11,11), width=5):
    """
    Generates a 2D radial cosine kernel. RH 2021

    Args:
        center (Tuple[int, int]):
            ``(x, y)`` peak position, zero-indexed. Set the second element
            to ``0`` to obtain a 1D kernel. (Default is ``(5, 5)``)
        image_size (Tuple[int, int]):
            ``(width, height)`` of the output kernel. Set the second
            element to ``0`` for a 1D kernel. (Default is ``(11, 11)``)
        width (float):
            Full width of one cycle of the cosine. (Default is ``5``)

    Returns:
        (np.ndarray):
            k_cos (np.ndarray):
                Cosine kernel. shape: *(image_size[0], image_size[1])*.
    """
    x, y = np.meshgrid(range(image_size[1]), range(image_size[0]))  # note dim 1:X and dim 2:Y
    dist = np.sqrt((y - int(center[1])) ** 2 + (x - int(center[0])) ** 2)
    dist_scaled = (dist/(width/2))*np.pi
    dist_scaled[np.abs(dist_scaled > np.pi)] = np.pi
    k_cos = (np.cos(dist_scaled) + 1)/2
    return k_cos


##########################################################################################################################################
############################################################# MATH FUNCTIONS #############################################################
##########################################################################################################################################

def bounded_logspace(start, stop, num,):
    """
    Logarithmically spaced values between ``start`` and ``stop`` (inclusive). RH 2022

    Args:
        start (float):
            First value in the output array.
        stop (float):
            Last value in the output array.
        num (int):
            Number of values in the output array.

    Returns:
        (np.ndarray):
            output (np.ndarray):
                Logarithmically spaced values bounded by ``start`` and
                ``stop``. shape: *(num,)*.
    """

    exp = 2  ## doesn't matter what this is, just needs to be > 1

    return exp ** np.linspace(np.log(start)/np.log(exp), np.log(stop)/np.log(exp), num, endpoint=True)

def gaussian(x=None, mu=0, sig=1, plot_pref=False):
    """
    Evaluates a normalized 1D Gaussian function on a grid. RH 2021

    Args:
        x (Optional[np.ndarray]):
            1D array of x positions. If ``None``, a default range covering
            five sigma on each side is used. (Default is ``None``)
        mu (float):
            Mean of the Gaussian. (Default is ``0``)
        sig (float):
            Standard deviation of the Gaussian. (Default is ``1``)
        plot_pref (bool):
            If ``True``, plots the Gaussian using matplotlib. (Default is
            ``False``)

    Returns:
        (np.ndarray):
            gaus (np.ndarray):
                Gaussian evaluated at each value of ``x``.
    """
    import matplotlib.pyplot as plt

    if x is None:
        x = np.linspace(-sig*5, sig*5, sig*7, endpoint=True)

    gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp((-((x-mu)/sig) **2)/2)

    if plot_pref:
        plt.figure()
        plt.plot(x , gaus)
        plt.xlabel('x')
        plt.title(f'mu={mu}, sigma={sig}')

    return gaus


##########################################################################################################################################
########################################################### SPECTRAL ANALYSIS ############################################################
##########################################################################################################################################

def torch_hilbert(x, N=None, dim=0):
    """
    Computes the analytic signal of ``x`` via a Hilbert transform. RH 2022

    Mirrors ``scipy.signal.hilbert`` but operates on ``torch.Tensor`` inputs.

    Args:
        x (torch.Tensor):
            Real-valued signal of arbitrary rank.
        N (Optional[int]):
            Number of Fourier components. If ``None``, uses ``x.shape[dim]``.
            (Default is ``None``)
        dim (int):
            Dimension along which to transform. (Default is ``0``)

    Returns:
        (torch.Tensor):
            xa (torch.Tensor):
                Complex analytic signal with the same shape as ``x``.
    """
    assert x.is_complex() == False, "x should be real"
    n = x.shape[dim] if N is None else N
    assert n >= 0, "N must be non-negative"

    xf = torch.fft.fft(input=x, n=n, dim=dim)
    m = torch.zeros(n, dtype=xf.dtype, device=xf.device)
    if n % 2 == 0: ## then even
        m[0] = m[n//2] = 1
        m[1:n//2] = 2
    else:
        m[0] = 1 ## then odd
        m[1:(n+1)//2] = 2

    if x.ndim > 1:
        ind = [np.newaxis] * x.ndim
        ind[dim] = slice(None)
        m = m[tuple(ind)]

    return torch.fft.ifft(xf * m, dim=dim)


def make_VQT_filters(    
    Fs_sample=1000,
    Q_lowF=3,
    Q_highF=20,
    F_min=10,
    F_max=400,
    n_freq_bins=55,
    win_size=501,
    symmetry='center',
    taper_asymmetric=True,
    plot_pref=False
):
    """
    Builds a bank of complex sinusoid filters for the VQT algorithm. RH 2022

    Setting ``Q_lowF == Q_highF`` produces a Constant-Q Transform (CQT)
    filter set. Differing values vary the Q factor logarithmically across
    the frequency range.

    Args:
        Fs_sample (float):
            Sampling frequency of the signal. (Default is ``1000``)
        Q_lowF (float):
            Q factor for the lowest frequency. (Default is ``3``)
        Q_highF (float):
            Q factor for the highest frequency. (Default is ``20``)
        F_min (float):
            Lowest frequency. (Default is ``10``)
        F_max (float):
            Highest frequency (inclusive). (Default is ``400``)
        n_freq_bins (int):
            Number of frequency bins. (Default is ``55``)
        win_size (int):
            Window size in samples. Must be odd. (Default is ``501``)
        symmetry (str):
            Window symmetry. One of \n
            * ``'center'``: symmetric / two-sided window.
            * ``'left'``: one-sided window, only the left half is nonzero.
            * ``'right'``: one-sided window, only the right half is nonzero. \n
            (Default is ``'center'``)
        taper_asymmetric (bool):
            If ``True`` and ``symmetry != 'center'``, the center sample of
            the window is multiplied by 0.5 to taper the discontinuity.
            (Default is ``True``)
        plot_pref (bool):
            If ``True``, plots the filters and windows. (Default is
            ``False``)

    Returns:
        (tuple): tuple containing:
            filts_complex (torch.Tensor):
                Complex sinusoid filters. shape: *(n_freq_bins, win_size)*.
            freqs (np.ndarray):
                Filter center frequencies. shape: *(n_freq_bins,)*.
            wins (torch.Tensor):
                Gaussian window for each filter. shape: *(n_freq_bins, win_size)*.
    """

    assert win_size%2==1, "RH Error: win_size should be an odd integer"
    
    ## Make frequencies. Use a geometric spacing.
    freqs = np.geomspace(
        start=F_min,
        stop=F_max,
        num=n_freq_bins,
        endpoint=True,
        dtype=np.float32,
    )

    periods = 1 / freqs
    periods_inSamples = Fs_sample * periods

    ## Make sigmas for gaussian windows. Use a geometric spacing.
    sigma_all = np.geomspace(
        start=Q_lowF,
        stop=Q_highF,
        num=n_freq_bins,
        endpoint=True,
        dtype=np.float32,
    )
    sigma_all = sigma_all * periods_inSamples / 4

    ## Make windows
    ### Make windows gaussian
    wins = torch.stack([gaussian(torch.arange(-win_size//2, win_size//2), 0, sig=sigma) for sigma in sigma_all])
    ### Make windows symmetric or asymmetric
    if symmetry=='center':
        pass
    else:
        heaviside = (torch.arange(win_size) <= win_size//2).float()
        if symmetry=='left':
            pass
        elif symmetry=='right':
            heaviside = torch.flip(heaviside, dims=[0])
        else:
            raise ValueError("symmetry must be 'center', 'left', or 'right'")
        wins *= heaviside
        ### Taper the center of the window by multiplying center sample of window by 0.5
        if taper_asymmetric:
            wins[:, win_size//2] = wins[:, win_size//2] * 0.5


    filts = torch.stack([torch.cos(torch.linspace(-np.pi, np.pi, win_size) * freq * (win_size/Fs_sample)) * win for freq, win in zip(freqs, wins)], dim=0)    
    filts_complex = torch_hilbert(filts.T, dim=0).T

    ## Normalize filters to have unit magnitude
    filts_complex = filts_complex / torch.sum(torch.abs(filts_complex), dim=1, keepdims=True)
    
    ## Plot
    if plot_pref:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(freqs)
        plt.xlabel('filter num')
        plt.ylabel('frequency (Hz)')

        plt.figure()
        plt.imshow(wins / torch.max(wins, 1, keepdims=True)[0], aspect='auto')
        plt.title('windows (gaussian)')

        plt.figure()
        plt.plot(sigma_all)
        plt.xlabel('filter num')
        plt.ylabel('window width (sigma of gaussian)')    

        plt.figure()
        plt.imshow(torch.real(filts_complex) / torch.max(torch.real(filts_complex), 1, keepdims=True)[0], aspect='auto', cmap='bwr', vmin=-1, vmax=1)
        plt.title('filters (real component)')


        worN=win_size*4
        filts_freq = np.array([scipy.signal.freqz(
            b=filt,
            fs=Fs_sample,
            worN=worN,
        )[1] for filt in filts_complex])

        filts_freq_xAxis = scipy.signal.freqz(
            b=filts_complex[0],
            worN=worN,
            fs=Fs_sample
        )[0]

        plt.figure()
        plt.plot(filts_freq_xAxis, np.abs(filts_freq.T));
        plt.xscale('log')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('magnitude')

    return filts_complex, freqs, wins

class VQT():
    """
    Variable Q Transform implemented with PyTorch. RH 2022

    Differs from librosa / nnAudio: this implementation does not iterate
    lowpass filtering. Instead it convolves a fixed set of complex filters,
    optionally returns the envelope via Hilbert transform, and downsamples.
    Gradients propagate through the transform, and computation can run on
    GPU. ``Q`` is the quality factor, roughly the number of cycles inside
    four sigma (95%) of a Gaussian window.

    Args:
        Fs_sample (float):
            Sampling frequency of the signal. (Default is ``1000``)
        Q_lowF (float):
            Q factor for the lowest frequency. (Default is ``3``)
        Q_highF (float):
            Q factor for the highest frequency. (Default is ``20``)
        F_min (float):
            Lowest frequency. (Default is ``10``)
        F_max (float):
            Highest frequency. (Default is ``400``)
        n_freq_bins (int):
            Number of frequency bins. (Default is ``55``)
        win_size (int):
            Window size in samples. Must be odd. (Default is ``501``)
        symmetry (str):
            Window symmetry passed through to ``make_VQT_filters``. One of \n
            * ``'center'``
            * ``'left'``
            * ``'right'`` \n
            (Default is ``'center'``)
        taper_asymmetric (bool):
            If ``True`` and ``symmetry != 'center'``, the center sample of
            the window is multiplied by 0.5. (Default is ``True``)
        downsample_factor (int):
            Time-downsampling factor. The input is zero-padded to be a
            multiple of this value. (Default is ``4``)
        padding (str):
            Convolution padding. ``'same'`` pads to keep output length
            equal to input length; ``'valid'`` does not pad. (Default is
            ``'valid'``)
        DEVICE_compute (str):
            Device used for computation. (Default is ``'cpu'``)
        DEVICE_return (str):
            Device on which results are returned. (Default is ``'cpu'``)
        batch_size (int):
            Number of signals processed per batch. Reduce when out of
            memory. (Default is ``1000``)
        return_complex (bool):
            If ``True``, returns the complex-valued transform; otherwise
            returns its absolute value (envelope). ``downsample_factor``
            must be ``1`` when ``True``. (Default is ``False``)
        filters (Optional[torch.Tensor]):
            Pre-built complex sinusoid filters. shape: *(n_freq_bins,
            win_size)*. If ``None``, ``make_VQT_filters`` is called.
            (Default is ``None``)
        plot_pref (bool):
            If ``True``, plots the filters. (Default is ``False``)
        progressBar (bool):
            If ``True``, displays a tqdm progress bar during ``__call__``.
            (Default is ``True``)

    Attributes:
        filters (torch.Tensor):
            Complex sinusoid filters used for convolution.
        freqs (np.ndarray):
            Filter center frequencies (only when filters were generated
            internally).
        wins (torch.Tensor):
            Gaussian windows for each filter (only when filters were
            generated internally).
        using_custom_filters (bool):
            ``True`` if filters were supplied by the caller.
    """
    def __init__(
        self,
        Fs_sample=1000,
        Q_lowF=3,
        Q_highF=20,
        F_min=10,
        F_max=400,
        n_freq_bins=55,
        win_size=501,
        symmetry='center',
        taper_asymmetric=True,
        downsample_factor=4,
        padding='valid',
        DEVICE_compute='cpu',
        DEVICE_return='cpu',
        batch_size=1000,
        return_complex=False,
        filters=None,
        plot_pref=False,
        progressBar=True,
    ):
        """Builds filters (or accepts pre-built ones) and stores all transform parameters."""
        ## Prepare filters
        if filters is not None:
            ## Use provided filters
            self.using_custom_filters = True
            self.filters = filters
        else:
            ## Make new filters
            self.using_custom_filters = False
            self.filters, self.freqs, self.wins = make_VQT_filters(
                Fs_sample=Fs_sample,
                Q_lowF=Q_lowF,
                Q_highF=Q_highF,
                F_min=F_min,
                F_max=F_max,
                n_freq_bins=n_freq_bins,
                win_size=win_size,
                symmetry=symmetry,
                taper_asymmetric=taper_asymmetric,
                plot_pref=plot_pref,
            )
        ## Gather parameters from arguments
        self.Fs_sample, self.Q_lowF, self.Q_highF, self.F_min, self.F_max, self.n_freq_bins, self.win_size, self.downsample_factor, self.padding, self.DEVICE_compute, \
            self.DEVICE_return, self.batch_size, self.return_complex, self.plot_pref, self.progressBar = \
                Fs_sample, Q_lowF, Q_highF, F_min, F_max, n_freq_bins, win_size, downsample_factor, padding, DEVICE_compute, DEVICE_return, batch_size, return_complex, plot_pref, progressBar

    def _helper_ds(self, X: torch.Tensor, ds_factor: int=4, return_complex: bool=False):
        """
        Downsamples ``X`` along the last dimension via average pooling.

        Args:
            X (torch.Tensor):
                Input tensor (real or complex).
            ds_factor (int):
                Downsampling factor. ``1`` returns ``X`` unchanged.
                (Default is ``4``)
            return_complex (bool):
                If ``True``, treats ``X`` as complex and pools the real and
                imaginary parts separately. (Default is ``False``)

        Returns:
            (torch.Tensor):
                X_ds (torch.Tensor):
                    Downsampled tensor.
        """
        if ds_factor == 1:
            return X
        elif return_complex == False:
            return torch.nn.functional.avg_pool1d(X, kernel_size=[int(ds_factor)], stride=ds_factor, ceil_mode=True)
        elif return_complex == True:
            ## Unfortunately, torch.nn.functional.avg_pool1d does not support complex numbers. So we have to split it up.
            ### Split X, shape: (batch_size, n_freq_bins, n_samples) into real and imaginary parts, shape: (batch_size, n_freq_bins, n_samples, 2)
            Y = torch.view_as_real(X)
            ### Downsample each part separately, then stack them and make them complex again.
            Z = torch.view_as_complex(torch.stack([torch.nn.functional.avg_pool1d(y, kernel_size=[int(ds_factor)], stride=ds_factor, ceil_mode=True) for y in [Y[...,0], Y[...,1]]], dim=-1))
            return Z

    def _helper_conv(self, arr, filters, take_abs, DEVICE):
        """
        Convolves a batch of signals with complex filters using two ``conv1d`` calls.

        Args:
            arr (torch.Tensor):
                Real input signals. shape: *(batch, n_samples)*.
            filters (torch.Tensor):
                Complex filter bank. shape: *(n_freq_bins, win_size)*.
            take_abs (bool):
                If ``True``, returns the magnitude of the result.
            DEVICE (str):
                Device used for the convolution.

        Returns:
            (torch.Tensor):
                out (torch.Tensor):
                    Filtered output, complex if ``take_abs`` is ``False``.
        """
        out = torch.complex(
            torch.nn.functional.conv1d(input=arr.to(DEVICE)[:,None,:], weight=torch.real(filters.T).to(DEVICE).T[:,None,:], padding=self.padding),
            torch.nn.functional.conv1d(input=arr.to(DEVICE)[:,None,:], weight=-torch.imag(filters.T).to(DEVICE).T[:,None,:], padding=self.padding)
        )
        if take_abs:
            return torch.abs(out)
        else:
            return out

    def __call__(self, X):
        """
        Computes the variable-Q spectrogram of ``X``.

        Args:
            X (torch.Tensor):
                Input signal. shape: *(n_channels, n_samples)*.

        Returns:
            (tuple): tuple containing:
                specs (torch.Tensor):
                    Spectrogram. shape: *(n_channels, n_freq_bins, n_samples_ds)*.
                x_axis (torch.Tensor):
                    New x-axis for the spectrogram in samples. Divide by
                    ``self.Fs_sample`` to get time.
                freqs (np.ndarray):
                    Frequencies corresponding to the spectrogram bins.
        """
        if type(X) is not torch.Tensor:
            X = torch.as_tensor(X, dtype=torch.float32, device=self.DEVICE_compute)

        if X.ndim==1:
            X = X[None,:]

        ## Make iterator for batches
        batches = make_batches(X, batch_size=self.batch_size, length=X.shape[0])

        ## Make spectrograms
        specs = [self._helper_ds(
            X=self._helper_conv(
                arr=arr, 
                filters=self.filters, 
                take_abs=(self.return_complex==False),
                DEVICE=self.DEVICE_compute
                ), 
            ds_factor=self.downsample_factor,
            return_complex=self.return_complex,
            ).to(self.DEVICE_return) for arr in tqdm(batches, disable=(self.progressBar==False), leave=True, total=int(np.ceil(X.shape[0]/self.batch_size)))]
        specs = torch.cat(specs, dim=0)

        ## Make x_axis
        x_axis = torch.nn.functional.avg_pool1d(
            torch.nn.functional.conv1d(
                input=torch.arange(0, X.shape[-1], dtype=torch.float32)[None,None,:], 
                weight=torch.ones(1,1,self.filters.shape[-1], dtype=torch.float32) / self.filters.shape[-1], 
                padding=self.padding
            ),
            kernel_size=[int(self.downsample_factor)], 
            stride=self.downsample_factor, ceil_mode=True,
        ).squeeze()
        
        return specs, x_axis, self.freqs

    def __repr__(self):
        """Returns a debug string summarizing the VQT configuration."""
        if self.using_custom_filters:
            return f"VQT with custom filters"
        else:
            return f"VQT object with parameters: Fs_sample={self.Fs_sample}, Q_lowF={self.Q_lowF}, Q_highF={self.Q_highF}, F_min={self.F_min}, F_max={self.F_max}, n_freq_bins={self.n_freq_bins}, win_size={self.win_size}, downsample_factor={self.downsample_factor}, DEVICE_compute={self.DEVICE_compute}, DEVICE_return={self.DEVICE_return}, batch_size={self.batch_size}, return_complex={self.return_complex}, plot_pref={self.plot_pref}"


def generate_multiphasic_sinewave(
    n_samples: int = 10000,
    n_periods: float = 1.0,
    n_waves: int = 3,
    return_x: bool = False,
    return_phases: bool = False,
):
    """
    Generates ``n_waves`` cosine waves with evenly spaced phase offsets. RH 2024

    Args:
        n_samples (int):
            Number of samples per wave. (Default is ``10000``)
        n_periods (float):
            Number of full periods spanned by ``n_samples``. (Default is
            ``1.0``)
        n_waves (int):
            Number of phase-shifted sine waves to return. (Default is ``3``)
        return_x (bool):
            If ``True``, also returns the x positions. (Default is ``False``)
        return_phases (bool):
            If ``True``, also returns the per-wave phase arrays. (Default is
            ``False``)

    Returns:
        (Union[np.ndarray, tuple]):
            output (Union[np.ndarray, tuple]):
                Combination depending on ``return_x`` / ``return_phases``: \n
                * ``waves`` (np.ndarray): generated cosine waves.
                * ``x`` (np.ndarray): x positions, if ``return_x``.
                * ``phases`` (np.ndarray): per-wave phases, if
                  ``return_phases``.
    """
    x = np.linspace(0, n_periods * np.pi*2, n_samples)

    phases = np.stack([
        x - ii * np.pi * (2 / n_waves) for ii in range(n_waves)
    ])
    waves = np.cos(phases)
    
    if return_x and return_phases:
        return waves, x, phases
    elif return_x:
        return waves, x
    elif return_phases:
        return waves, phases
    else:
        return waves
    

############################################################################################################################################################################
#################################################################### TORCH HELPERS #########################################################################################
############################################################################################################################################################################


def set_device(
    use_GPU: bool = True, 
    device_num: int = 0, 
    verbose: bool = True
) -> str:
    """
    Sets the device for PyTorch. If a GPU is available and **use_GPU** is
    ``True``, it will be set as the device. Otherwise, the CPU will be set as
    the device. 
    RH 2022

    Args:
        use_GPU (bool): 
            Determines if the GPU should be utilized: \n
            * ``True``: the function will attempt to use the GPU if a GPU is
              not available.
            * ``False``: the function will use the CPU. \n
            (Default is ``True``)
        device_num (int): 
            Specifies the index of the GPU to use. (Default is ``0``)
        verbose (bool): 
            Determines whether to print the device information. \n
            * ``True``: the function will print out the device information.
            \n
            (Default is ``True``)

    Returns:
        (str): 
            device (str): 
                A string specifying the device, either *"cpu"* or
                *"cuda:<device_num>"*.
    """
    if use_GPU:
        print(f'devices available: {[torch.cuda.get_device_properties(ii) for ii in range(torch.cuda.device_count())]}') if verbose else None
        device = f"cuda:{device_num}" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("no GPU available. Using CPU.") if verbose else None
        else:
            print(f"Using device: '{device}': {torch.cuda.get_device_properties(device_num)}") if verbose else None
    else:
        device = "cpu"
        print(f"device: '{device}'") if verbose else None

    return device


def tensorly_cp_to_device(cp, device='cpu'):
    """
    Moves the factors and weights of a tensorly CP object to ``device``. RH 2024

    Args:
        cp (object):
            Tensorly ``CP`` tensor (``tensorly.cp_tensor.CP``).
        device (str):
            Target device for ``cp.factors`` and ``cp.weights``. (Default
            is ``'cpu'``)

    Returns:
        (object):
            cp (object):
                Same CP object with all factors and weights on ``device``.
    """
    for ii in range(len(cp.factors)):
        cp.factors[ii] = cp.factors[ii].to(device)
    cp.weights = cp.weights.to(device)
    return cp


############################################################################################################################################################################
################################################################## PLOTTING HELPERS ########################################################################################
############################################################################################################################################################################

def simple_cmap(
    colors=[
        [1,0,0],
        [1,0.6,0],
        [0.9,0.9,0],
        [0.6,1,0],
        [0,1,0],
        [0,1,0.6],
        [0,0.8,0.8],
        [0,0.6,1],
        [0,0,1],
        [0.6,0,1],
        [0.8,0,0.8],
        [1,0,0.6],
    ],
    under=[0,0,0],
    over=[0.5,0.5,0.5],
    bad=[0.9,0.9,0.9],
    name='none'):
    """
    Builds a ``LinearSegmentedColormap`` from a sequence of RGB values.

    Adapted from
    https://gist.github.com/ahwillia/3e022cdd1fe82627cbf1f2e9e2ad80a7e.

    Args:
        colors (list):
            Sequence of RGB triples (or matplotlib color strings) defining
            the colormap stops.
        under (list):
            RGB color used for values below the colormap range. (Default is
            ``[0, 0, 0]``)
        over (list):
            RGB color used for values above the colormap range. (Default is
            ``[0.5, 0.5, 0.5]``)
        bad (list):
            RGB color used for masked / NaN values. (Default is
            ``[0.9, 0.9, 0.9]``)
        name (str):
            Colormap name. (Default is ``'none'``)

    Returns:
        (matplotlib.colors.LinearSegmentedColormap):
            cmap (matplotlib.colors.LinearSegmentedColormap):
                Resulting linear-segmented colormap.

    Example:
        .. highlight:: python
        .. code-block:: python

            cmap = simple_cmap([(1, 1, 1), (1, 0, 0)])  # white to red
            cmap = simple_cmap(['w', 'r'])               # white to red
            cmap = simple_cmap(['r', 'b', 'r'])          # red to blue to red
    """
    from matplotlib.colors import LinearSegmentedColormap, colorConverter

    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')

    # convert colors to rgb
    colors = [colorConverter.to_rgb(c) for c in colors]

    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    cmap = LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})
                                   
    cmap.set_bad(bad)
    cmap.set_over(over)
    cmap.set_under(under)

    return cmap


class Cmap_conjunctive:
    """
    Combines multiple colormaps by multiplying their per-channel outputs. RH 2022

    Args:
        cmaps (list):
            List of ``matplotlib.colors.LinearSegmentedColormap`` objects to
            combine.
        dtype_out (np.dtype):
            Data type of the returned color array. (Default is ``int``)
        normalize (bool):
            If ``True``, normalizes each input column to ``[0, 1]`` before
            applying the colormaps. (Default is ``False``)
        normalization_range (list):
            ``[lo, hi]`` to which the output is rescaled. (Default is
            ``[0, 255]``)
        name (str):
            Name of the resulting colormap. (Default is ``'cmap_conjunctive'``)

    Attributes:
        cmaps (list):
            Stored input colormaps.
        n_cmaps (int):
            Number of input colormaps.
        fn_conj_cmap (Callable):
            Function that maps an input array of shape *(n_samples, n_cmaps)*
            to the elementwise product of each colormap's output.
    """
    def __init__(
        self,
        cmaps,
        dtype_out=int,
        normalize=False,
        normalization_range=[0,255],
        name='cmap_conjunctive',
    ):
        """Validates the colormaps and stores normalization parameters."""
        import matplotlib

        ## Check inputs
        assert isinstance(cmaps, list), 'cmaps must be a list.'
        assert all([isinstance(cmap, matplotlib.colors.LinearSegmentedColormap) for cmap in cmaps]), 'All elements of cmaps must be matplotlib.colors.LinearSegmentedColormap objects.'

        self.cmaps = cmaps
        self.dtype_out = dtype_out
        self.name = name
        self.normalize = normalize
        self.normalization_range = normalization_range

        self.n_cmaps = len(self.cmaps)

        self.fn_conj_cmap = lambda x: np.prod(np.stack([cmap(x_i) for cmap,x_i in zip(self.cmaps, x.T)], axis=0), axis=0)

    def __call__(self, x):
        """
        Applies the conjunctive colormap to the input array.

        Args:
            x (np.ndarray):
                Input values. shape: *(n_samples, n_cmaps)*. When
                ``self.normalize`` is ``True``, each column is independently
                rescaled to ``[0, 1]``.

        Returns:
            (np.ndarray):
                colors (np.ndarray):
                    RGBA values rescaled to ``self.normalization_range``.
                    shape: *(n_samples, 4)*, dtype: ``self.dtype_out``.
        """
        assert isinstance(x, np.ndarray), 'x must be a numpy array of shape (n_samples, n_cmaps).'

        ## Make array 2D
        if x.ndim == 1:
            x = x[None,:]
        assert x.shape[1] == self.n_cmaps, 'x.shape[1] must match the number of cmaps.'

        ## Normalize x
        if self.normalize:
            assert x.shape[1] > 1, 'x must have more than one row to normalize.'
            x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True))

        ## Get colors
        colors = self.fn_conj_cmap(x)
        colors = (colors * (self.normalization_range[1] - self.normalization_range[0]) + self.normalization_range[0]).astype(self.dtype_out)

        return colors


class Colorwheel:
    """
    2D colorwheel colormap (angle + magnitude) for cyclic data. RH 2024

    Useful for visualizing complex/polar values, optical flow, and other
    cyclic data.

    Args:
        rotation (float):
            Rotation of the colorwheel in radians. (Default is ``0.0``)
        saturation (float):
            Color saturation in *[0, 1]*. (Default is ``1.0``)
        center (int):
            Color value at the center of the wheel. (Default is ``0``)
        radius (int):
            Maximum color value at the rim of the wheel. (Default is ``255``)
        dtype (np.dtype):
            Output dtype of the color array. (Default is ``np.uint8``)
        bit_depth (int):
            Number of samples used to discretize the wheel:
            ``2 ** bit_depth``. (Default is ``16``)
        exponent (float):
            Exponent applied to each base color wave to sharpen transitions.
            (Default is ``10``)
        normalize (bool):
            If ``True``, normalizes the per-angle color sum to ``1`` so that
            color intensity is uniform around the wheel. (Default is
            ``True``)
        colors (List[Union[List, Tuple]]):
            Sequence of base RGB triples used to build the rainbow.
            (Default is a 12-color rainbow)

    Attributes:
        fn_interp (Callable):
            Interpolator that maps an angle (radians) to per-base-color
            weights along the wheel.
        colors (np.ndarray):
            Array of base colors. shape: *(n_colors, 3)*.
    """
    def __init__(
        self,
        rotation: float = 0.0,
        saturation: float = 1.0,
        center: int = 0,
        radius: int = 255,
        dtype: np.dtype = np.uint8,
        bit_depth: int = 16,
        exponent: float = 10,
        normalize: bool = True,
        colors: List[Union[List, Tuple]] = [
            [1  , 0  , 0  ], 
            [1  , 0.5, 0  ], 
            [1  , 1  , 0  ], 
            [0.5, 1  , 0  ], 
            [0  , 1  , 0  ],  
            [0  , 1  , 0.5],  
            [0  , 1  , 1  ], 
            [0  , 0.5, 1  ], 
            [0  , 0  , 1  ],
            [0.5, 0  , 1  ],
            [1  , 0  , 1  ], 
            [1  , 0  , 0.5], 
        ],        
    ):
        """Builds the angle-to-color interpolator from the rainbow base waves."""
        import scipy.interpolate
        import scipy.special

        self.rotation = rotation
        self.saturation = saturation
        self.center = center
        self.radius = radius
        self.dtype = dtype
        self.bit_depth = bit_depth
        self.exponent = exponent
        self.normalize = normalize
        self.colors = np.array(colors)

        # Make a rainbow colorwheel
        # Create 3 single cosine waves centered at 0, 120, and 240 degrees spanning 120 degrees each
        import scipy.signal
        waves, x = generate_multiphasic_sinewave(
            n_samples=int(2**bit_depth),
            n_periods=1 + 2 / int(2**bit_depth),
            n_waves=len(colors),
            return_x=True,
        )

        waves = ((waves + 1).astype(np.float64) / 2) ** exponent
        
        waves = (waves - waves.min()) / (waves.max() - waves.min())

        if normalize:
            if waves.shape[0] == 1:
                waves = np.ones_like(waves)
            else:
                waves = waves / np.sum(waves, axis=0, keepdims=True)

        waves = (waves * (radius - (1-saturation) * radius) + (1-saturation) * radius)
        waves = np.roll(waves, int(rotation * 2**bit_depth / (2*np.pi)), axis=1)

        # Create interpolation function
        self.fn_interp = scipy.interpolate.interp1d(
            x=x,
            y=waves,
            kind='linear',
            axis=1,
            bounds_error=False,
            fill_value='extrapolate',
        )

    def __call__(
        self,
        angles: Union[np.ndarray, List[Union[float, int]], Tuple[Union[float, int]], float, int],
        magnitudes: Optional[Union[np.ndarray, List[Union[float, int]], Tuple[Union[float, int]], float, int]] = None,
        normalize_magnitudes: bool = True,
    ) -> np.ndarray:
        """
        Returns RGB colors for the given angles and (optional) magnitudes. RH 2024

        Args:
            angles (Union[np.ndarray, List[Union[float, int]], Tuple[Union[float, int]], float, int]):
                Angles in radians. shape: *(n_samples,)*.
            magnitudes (Optional[Union[np.ndarray, List[Union[float, int]], Tuple[Union[float, int]], float, int]]):
                Magnitudes for each angle. shape: *(n_samples,)*. If
                ``None``, treated as ones. (Default is ``None``)
            normalize_magnitudes (bool):
                If ``True``, applies min-max normalization to ``magnitudes``
                before use. (Default is ``True``)

        Returns:
            (np.ndarray):
                rgb (np.ndarray):
                    RGB values. shape: *(n_samples, 3)*, dtype:
                    ``self.dtype``.
        """
        # Check inputs
        def check_input(arg):
            if isinstance(arg, (float, int)):
                arg = np.array([angles])
            elif isinstance(arg, (list, tuple)):
                arg = np.array(arg)
            elif not isinstance(angles, np.ndarray):
                raise ValueError("angles and magnitudes must be a numpy array, list, or tuple of ints or floats.")
            return arg

        angles = check_input(angles)
        magnitudes = check_input(magnitudes) if magnitudes is not None else None

        # Normalize the magnitudes
        if magnitudes is not None:
            if normalize_magnitudes:
                magnitudes = (magnitudes - np.min(magnitudes)) / (np.max(magnitudes) - np.min(magnitudes))
            magnitudes = np.clip(magnitudes, 0, 1)
        else:
            magnitudes = np.ones_like(angles)

        # Get the saturated color by interpolating the colorwheel
        sample_colors = self.fn_interp(angles % (2*np.pi))
        
        # Clip the colors
        sample_colors = np.clip(sample_colors, 0, self.radius)

        # Project to RGB
        rgb = self.colors.T @ sample_colors


        # Apply the saturation
        rgb = rgb * magnitudes[None, :] + (1 - magnitudes)[None, :] * self.center

        # Convert to dtype
        ## Clip the values to the dtype range
        if rgb.dtype != self.dtype:
            vmax, vmin = np.iinfo(self.dtype).max, np.iinfo(self.dtype).min
            rgb = np.clip(rgb, vmin, vmax)

        rgb = rgb.astype(self.dtype)

        return rgb.T

    def plot_colorwheel(self, n_samples: int = 100000):
        """
        Renders the colorwheel as a 2D image and displays it. RH 2024

        Args:
            n_samples (int):
                Approximate number of samples used to build the wheel; the
                actual grid is ``ceil(sqrt(n_samples))`` per side. (Default
                is ``100000``)
        """
        import matplotlib.pyplot as plt
        l = int(np.ceil(n_samples**0.5))
        grid = np.meshgrid(np.linspace(-1, 1, l), np.linspace(-1, 1, l), indexing='xy')
        grid = grid[0] + 1j*grid[1]
        grid = grid.reshape(-1)
        angles = np.angle(grid)
        magnitudes = np.abs(grid)

        mask = magnitudes > 1
        magnitudes = np.clip(magnitudes, 0, 1)
        colors = self(angles, magnitudes)
        colors = np.clip(colors, 0, 255)

        im = np.zeros((l, l, 3), dtype=self.dtype)
        im[tuple(np.meshgrid(range(l), range(l), indexing='ij'))] = colors.reshape(im.shape[:2] + (3,))

        fig, axs = plt.subplots(2, 1, figsize=(5, 10))
        axs[0].imshow(im)
        x = np.linspace(0, 2*np.pi, l)
        [axs[1].plot(x, v, color=c) for v, c in zip(self(x).T, self.colors)]
        axs[1].set_ylabel('Channel magnitude')
        axs[1].set_xlabel('Phase (rads)')

    def __repr__(self) -> str:
        """Returns a debug string with the colorwheel construction parameters."""
        return (f"ColorwheelColormap(rotation={self.rotation}, "
                f"saturation={self.saturation}, center={self.center}, "
                f"radius={self.radius}, dtype={self.dtype}, "
                f"bit_depth={self.bit_depth}, exponent={self.exponent}, "
                f"normalize={self.normalize})")


##########################################################################################################################################
############################################################ IMAGE PROCESSING ############################################################
##########################################################################################################################################


def clahe(im, grid_size=50, clipLimit=0, normalize=True):
    """
    Applies Contrast Limited Adaptive Histogram Equalization to an image. RH 2022

    Args:
        im (np.ndarray):
            Input image.
        grid_size (int):
            Tile grid size passed to ``cv2.createCLAHE``. (Default is ``50``)
        clipLimit (int):
            Contrast clip limit passed to ``cv2.createCLAHE``. (Default is
            ``0``)
        normalize (bool):
            If ``True``, normalizes the input to span the full 16-bit range
            before applying CLAHE. (Default is ``True``)

    Returns:
        (np.ndarray):
            im_c (np.ndarray):
                CLAHE-enhanced image. dtype: *uint16*.
    """
    import cv2
    im_tu = (im / im.max())*(2**16) if normalize else im
    im_tu = im_tu/10
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid_size, grid_size))
    im_c = clahe.apply(im_tu.astype(np.uint16))
    return im_c


def add_text_to_images(images, text, position=(10,10), font_size=1, color=(255,255,255), line_width=1, font=None, show=False, frameRate=30):
    """
    Overlays multi-line text onto each frame using ``cv2.putText``. RH 2022

    Args:
        images (np.ndarray):
            Frames of video or images. shape: *(n_frames, H, W, C)*.
        text (List[List[str]]):
            Text per frame. Outer list has one element per frame; each
            inner list holds the lines of text drawn on that frame.
        position (Tuple[int, int]):
            ``(x, y)`` position of the top-left corner of the text. (Default
            is ``(10, 10)``)
        font_size (int):
            Font scale passed to ``cv2.putText``. (Default is ``1``)
        color (Tuple[int, int, int]):
            ``(R, G, B)`` text color. (Default is ``(255, 255, 255)``)
        line_width (int):
            Line thickness passed to ``cv2.putText``. (Default is ``1``)
        font (Optional[int]):
            OpenCV font constant. If ``None``, uses
            ``cv2.FONT_HERSHEY_SIMPLEX``. (Default is ``None``)
        show (bool):
            If ``True``, displays each annotated frame using
            ``cv2.imshow``. (Default is ``False``)
        frameRate (float):
            Display frame rate when ``show`` is ``True``. (Default is ``30``)

    Returns:
        (np.ndarray):
            images_with_text (np.ndarray):
                Frames of video or images with text overlays applied.
    """
    import cv2
    import copy
    
    if font is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
    
    images_cp = copy.deepcopy(images)
    for ii, im in enumerate(images_cp):
        im = im[:,:,None] if im.ndim==2 else im
        images_cp[ii] = im
        
    for i_f, frame in enumerate(images_cp):
        for i_t, t in enumerate(text[i_f]):
            cv2.putText(frame, t, [position[0] , position[1] + i_t*font_size*30], font, font_size, color, line_width)
            
            if show:
                cv2.imshow('add_text_to_images', frame)
                cv2.waitKey(int(1000/frameRate))
    
        for ii, im in enumerate(images_cp):
            im = im[:,:,0] if images[ii].ndim==2 else im
            images_cp[ii] = im

    if show:
        cv2.destroyWindow('add_text_to_images')
    return images_cp


def mask_image_border(
    im: np.ndarray, 
    border_outer: Optional[Union[int, Tuple[int, int, int, int]]] = None, 
    border_inner: Optional[int] = None, 
    mask_value: float = 0,
) -> np.ndarray:
    """
    Masks an image within specified outer and inner borders. RH 2022

    Args:
        im (np.ndarray):
            Input image of shape: *(height, width)* or *(height, width,
            channels)*.
        border_outer (Union[int, tuple[int, int, int, int], None]):
            Number of pixels along the border to mask. If ``None``, the border
            is not masked. If an int is provided, all borders are equally
            masked. If a tuple of ints is provided, borders are masked in the
            order: *(top, bottom, left, right)*. (Default is ``None``)
        border_inner (int, Optional):
            Number of pixels in the center to mask. Will be a square with side
            length equal to this value. (Default is ``None``)
        mask_value (float):
            Value to replace the masked pixels with. (Default is *0*)

    Returns:
        (np.ndarray):
            im_out (np.ndarray):
                Masked output image.
    """

    ## Find the center of the image
    height, width = im.shape[:2]
    center_y = cy = int(np.floor(height/2))
    center_x = cx = int(np.floor(width/2))

    ## Mask the center
    if border_inner is not None:
        ## make edge_lengths
        center_edge_length = cel = int(np.ceil(border_inner/2)) if border_inner is not None else 0
        im[cy-cel:cy+cel, cx-cel:cx+cel] = mask_value
    ## Mask the border
    if border_outer is not None:
        ## make edge_lengths
        if isinstance(border_outer, int):
            border_outer = (border_outer, border_outer, border_outer, border_outer)
        
        im[:border_outer[0], :] = mask_value
        im[-border_outer[1]:, :] = mask_value
        im[:, :border_outer[2]] = mask_value
        im[:, -border_outer[3]:] = mask_value

    return im


def find_geometric_transformation(
    im_template: np.ndarray, 
    im_moving: np.ndarray,
    warp_mode: str = 'euclidean',
    n_iter: int = 5000,
    termination_eps: float = 1e-10,
    mask: Optional[np.ndarray] = None,
    gaussFiltSize: int = 1
) -> np.ndarray:
    """
    Estimates the geometric transformation between two images via ECC. RH 2022

    Wraps ``cv2.findTransformECC``.

    Args:
        im_template (np.ndarray):
            Template image. dtype: *uint8* or *float32*.
        im_moving (np.ndarray):
            Moving image. dtype: *uint8* or *float32*.
        warp_mode (str):
            Motion model. One of \n
            * ``'translation'``: 2x3 warpMatrix; only translation is
              estimated.
            * ``'euclidean'``: 2x3 warpMatrix; rigid (rotation + translation).
            * ``'affine'``: 2x3 warpMatrix; six parameters.
            * ``'homography'``: 3x3 warpMatrix; eight parameters. \n
            (Default is ``'euclidean'``)
        n_iter (int):
            Maximum number of iterations. (Default is ``5000``)
        termination_eps (float):
            Threshold on the per-iteration increment of the correlation
            coefficient. (Default is ``1e-10``)
        mask (Optional[np.ndarray]):
            Binary mask. Pixels where mask is zero are ignored. If
            ``None``, no mask is used. (Default is ``None``)
        gaussFiltSize (int):
            Gaussian filter size. ``0`` disables filtering. (Default is
            ``1``)

    Returns:
        (np.ndarray):
            warp_matrix (np.ndarray):
                Estimated warp matrix. Apply with ``cv2.warpAffine`` or
                ``cv2.warpPerspective``.
    """
    LUT_modes = {
        'translation': cv2.MOTION_TRANSLATION,
        'euclidean': cv2.MOTION_EUCLIDEAN,
        'affine': cv2.MOTION_AFFINE,
        'homography': cv2.MOTION_HOMOGRAPHY,
    }
    assert warp_mode in LUT_modes.keys(), f"warp_mode must be one of {LUT_modes.keys()}. Got {warp_mode}"
    warp_mode = LUT_modes[warp_mode]
    if warp_mode in [cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE]:
        shape_eye = (2, 3)
    elif warp_mode == cv2.MOTION_HOMOGRAPHY:
        shape_eye = (3, 3)
    else:
        raise ValueError(f"warp_mode {warp_mode} not recognized (should not happen)")
    warp_matrix = np.eye(*shape_eye, dtype=np.float32)

    ## assert that the inputs are numpy arrays of dtype np.uint8
    assert isinstance(im_template, np.ndarray) and (im_template.dtype == np.uint8 or im_template.dtype == np.float32), f"im_template must be a numpy array of dtype np.uint8 or np.float32. Got {type(im_template)} of dtype {im_template.dtype}"
    assert isinstance(im_moving, np.ndarray) and (im_moving.dtype == np.uint8 or im_moving.dtype == np.float32), f"im_moving must be a numpy array of dtype np.uint8 or np.float32. Got {type(im_moving)} of dtype {im_moving.dtype}"
    ## cast mask to bool then to uint8
    if mask is not None:
        assert isinstance(mask, np.ndarray), f"mask must be a numpy array. Got {type(mask)}"
        if np.issubdtype(mask.dtype, np.bool_) or np.issubdtype(mask.dtype, np.uint8):
            pass
        else:
            mask = (mask != 0).astype(np.uint8)
    
    ## make gaussFiltSize odd
    gaussFiltSize = int(np.ceil(gaussFiltSize))
    gaussFiltSize = gaussFiltSize + (gaussFiltSize % 2 == 0)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        n_iter,
        termination_eps,
    )
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(
        templateImage=im_template, 
        inputImage=im_moving, 
        warpMatrix=warp_matrix,
        motionType=warp_mode, 
        criteria=criteria, 
        inputMask=mask, 
        gaussFiltSize=gaussFiltSize
    )
    return warp_matrix

def apply_warp_transform(
    im_in: np.ndarray,
    warp_matrix: np.ndarray,
    interpolation_method: int = cv2.INTER_LINEAR, 
    borderMode: int = cv2.BORDER_CONSTANT, 
    borderValue: int = 0
) -> np.ndarray:
    """
    Applies a warp transform to an image. RH 2022

    Wraps ``cv2.warpAffine`` (for 2x3 matrices) and ``cv2.warpPerspective``
    (for 3x3 matrices).

    Args:
        im_in (np.ndarray): 
            Input image with any dimensions.
        warp_matrix (np.ndarray): 
            Warp matrix. Shape should be *(2, 3)* for affine transformations,
            and *(3, 3)* for homography. See ``cv2.findTransformECC`` for more
            info.
        interpolation_method (int): 
            Interpolation method. See ``cv2.warpAffine`` for more info. (Default
            is ``cv2.INTER_LINEAR``)
        borderMode (int): 
            Border mode. Determines how to handle pixels from outside the image
            boundaries. See ``cv2.warpAffine`` for more info. (Default is
            ``cv2.BORDER_CONSTANT``)
        borderValue (int): 
            Value to use for border pixels if borderMode is set to
            ``cv2.BORDER_CONSTANT``. (Default is *0*)

    Returns:
        (np.ndarray): 
            im_out (np.ndarray): 
                Transformed output image with the same dimensions as the input
                image.
    """
    if warp_matrix.shape == (2, 3):
        im_out = cv2.warpAffine(
            src=im_in,
            M=warp_matrix,
            dsize=(im_in.shape[1], im_in.shape[0]),
            dst=copy.copy(im_in),
            flags=interpolation_method + cv2.WARP_INVERSE_MAP,
            borderMode=borderMode,
            borderValue=borderValue
        )
        
    elif warp_matrix.shape == (3, 3):
        im_out = cv2.warpPerspective(
            src=im_in,
            M=warp_matrix,
            dsize=(im_in.shape[1], im_in.shape[0]), 
            dst=copy.copy(im_in), 
            flags=interpolation_method + cv2.WARP_INVERSE_MAP, 
            borderMode=borderMode, 
            borderValue=borderValue
        )

    else:
        raise ValueError(f"warp_matrix.shape {warp_matrix.shape} not recognized. Must be (2, 3) or (3, 3)")
    
    return im_out


def warp_matrix_to_remappingIdx(
    warp_matrix: Union[np.ndarray, torch.Tensor], 
    x: int, 
    y: int
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts a warp matrix (2x3 or 3x3) into a 2D remapping index field. RH 2023

    Args:
        warp_matrix (Union[np.ndarray, torch.Tensor]):
            Warp matrix of shape *(2, 3)* for affine transformations, or
            *(3, 3)* for homography.
        x (int):
            Width of the output remapping field.
        y (int):
            Height of the output remapping field.

    Returns:
        (Union[np.ndarray, torch.Tensor]):
            remapIdx (Union[np.ndarray, torch.Tensor]):
                Remapping indices. shape: *(y, x, 2)*. The last axis stores
                the pixel coordinate ``(x, y)`` to sample from.
    """
    assert warp_matrix.shape in [(2, 3), (3, 3)], f"warp_matrix.shape {warp_matrix.shape} not recognized. Must be (2, 3) or (3, 3)"
    assert isinstance(x, int) and isinstance(y, int), f"x and y must be integers"
    assert x > 0 and y > 0, f"x and y must be positive"

    if isinstance(warp_matrix, torch.Tensor):
        stack, meshgrid, arange, hstack, ones, float32, array = torch.stack, torch.meshgrid, torch.arange, torch.hstack, torch.ones, torch.float32, torch.as_tensor
        stack_partial = lambda x: stack(x, dim=0)
    elif isinstance(warp_matrix, np.ndarray):
        stack, meshgrid, arange, hstack, ones, float32, array = np.stack, np.meshgrid, np.arange, np.hstack, np.ones, np.float32, np.array
        stack_partial = lambda x: stack(x, axis=0)
    else:
        raise ValueError(f"warp_matrix must be a torch.Tensor or np.ndarray")

    # create the grid
    mesh = stack_partial(meshgrid(arange(x, dtype=float32), arange(y, dtype=float32)))
    mesh_coords = hstack((mesh.reshape(2,-1).T, ones((x*y, 1), dtype=float32)))
    
    # warp the grid
    mesh_coords_warped = (mesh_coords @ warp_matrix.T)
    mesh_coords_warped = mesh_coords_warped[:, :2] / mesh_coords_warped[:, 2:3] if warp_matrix.shape == (3, 3) else mesh_coords_warped  ## if homography, divide by z
    
    # reshape the warped grid
    remapIdx = mesh_coords_warped.T.reshape(2, y, x)

    # permute the axes to (x, y, 2)
    remapIdx = remapIdx.permute(1, 2, 0) if isinstance(warp_matrix, torch.Tensor) else remapIdx.transpose(1, 2, 0)

    return remapIdx


def remap_images(
    images: Union[np.ndarray, torch.Tensor],
    remappingIdx: Union[np.ndarray, torch.Tensor],
    backend: str = "torch",
    interpolation_method: str = 'linear',
    border_mode: str = 'constant',
    border_value: float = 0,
    device: str = 'cpu',
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies remapping indices to a set of images. Remapping indices, similar to
    flow fields, describe the index of the pixel to sample from rather than the
    displacement of each pixel. RH 2023

    Args:
        images (Union[np.ndarray, torch.Tensor]): 
            The images to be warped. Shapes can be *(N, C, H, W)*, *(C, H, W)*,
            or *(H, W)*.
        remappingIdx (Union[np.ndarray, torch.Tensor]): 
            The remapping indices, describing the index of the pixel to sample
            from. Shape is *(H, W, 2)*.
        backend (str): 
            The backend to use. Can be either ``'torch'`` or ``'cv2'``. (Default
            is ``'torch'``)
        interpolation_method (str): 
            The interpolation method to use. Options are ``'linear'``,
            ``'nearest'``, ``'cubic'``, and ``'lanczos'``. Refer to `cv2.remap`
            or `torch.nn.functional.grid_sample` for more details. (Default is
            ``'linear'``)
        border_mode (str): 
            The border mode to use. Options include ``'constant'``,
            ``'reflect'``, ``'replicate'``, and ``'wrap'``. Refer to `cv2.remap`
            for more details. (Default is ``'constant'``)
        border_value (float): 
            The border value to use. Refer to `cv2.remap` for more details.
            (Default is ``0``)
        device (str):
            The device to use for computations. Commonly either ``'cpu'`` or
            ``'gpu'``. (Default is ``'cpu'``)

    Returns:
        (Union[np.ndarray, torch.Tensor]):
            warped_images (Union[np.ndarray, torch.Tensor]):
                The warped images. The shape will be the same as the input
                images, which can be *(N, C, H, W)*, *(C, H, W)*, or *(H, W)*.
    """
    # Check inputs
    assert isinstance(images, (np.ndarray, torch.Tensor)), f"images must be a np.ndarray or torch.Tensor"
    assert isinstance(remappingIdx, (np.ndarray, torch.Tensor)), f"remappingIdx must be a np.ndarray or torch.Tensor"
    if images.ndim == 2:
        images = images[None, None, :, :]
    elif images.ndim == 3:
        images = images[None, :, :, :]
    elif images.ndim != 4:
        raise ValueError(f"images must be a 2D, 3D, or 4D array. Got shape {images.shape}")
    assert remappingIdx.ndim == 3, f"remappingIdx must be a 3D array of shape (H, W, 2). Got shape {remappingIdx.shape}"
    assert images.shape[-2] == remappingIdx.shape[0], f"images H ({images.shape[-2]}) must match remappingIdx H ({remappingIdx.shape[0]})"
    assert images.shape[-1] == remappingIdx.shape[1], f"images W ({images.shape[-1]}) must match remappingIdx W ({remappingIdx.shape[1]})"

    # Check backend
    if backend not in ["torch", "cv2"]:
        raise ValueError("Invalid backend. Supported backends are 'torch' and 'cv2'.")
    if backend == 'torch':
        if isinstance(images, np.ndarray):
            images = torch.as_tensor(images, device=device, dtype=torch.float32)
        elif isinstance(images, torch.Tensor):
            images = images.to(device=device).type(torch.float32)
        if isinstance(remappingIdx, np.ndarray):
            remappingIdx = torch.as_tensor(remappingIdx, device=device, dtype=torch.float32)
        elif isinstance(remappingIdx, torch.Tensor):
            remappingIdx = remappingIdx.to(device=device).type(torch.float32)
        interpolation = {
            'linear': 'bilinear',
            'nearest': 'nearest',
            'cubic': 'bicubic',
            'lanczos': 'lanczos',
        }[interpolation_method]
        border = {
            'constant': 'zeros',
            'reflect': 'reflection',
            'replicate': 'replication',
            'wrap': 'circular',
        }[border_mode]
        ## Convert remappingIdx to normalized grid
        normgrid = cv2RemappingIdx_to_pytorchFlowField(remappingIdx)

        # Apply remappingIdx
        warped_images = torch.nn.functional.grid_sample(
            images, 
            normgrid[None,...],
            mode=interpolation, 
            padding_mode=border, 
            align_corners=True,  ## align_corners=True is the default in cv2.remap. See documentation for details.
        )

    elif backend == 'cv2':
        assert isinstance(images, np.ndarray), f"images must be a np.ndarray when using backend='cv2'"
        assert isinstance(remappingIdx, np.ndarray), f"remappingIdx must be a np.ndarray when using backend='cv2'"
        ## convert to float32 if not uint8
        images = images.astype(np.float32) if images.dtype != np.uint8 else images
        remappingIdx = remappingIdx.astype(np.float32) if remappingIdx.dtype != np.uint8 else remappingIdx

        interpolation = {
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'cubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
        }[interpolation_method]
        borderMode = {
            'constant': cv2.BORDER_CONSTANT,
            'reflect': cv2.BORDER_REFLECT,
            'replicate': cv2.BORDER_REPLICATE,
            'wrap': cv2.BORDER_WRAP,
        }[border_mode]

        # Apply remappingIdx
        def remap(ims):
            out = np.stack([cv2.remap(
                im,
                remappingIdx[..., 0], 
                remappingIdx[..., 1], 
                interpolation=interpolation, 
                borderMode=borderMode, 
                borderValue=border_value,
            ) for im in ims], axis=0)
            return out
        warped_images = np.stack([remap(im) for im in images], axis=0)

    return warped_images.squeeze()


def invert_remappingIdx(
    remappingIdx: np.ndarray, 
    method: str = 'linear', 
    fill_value: Optional[float] = np.nan
) -> np.ndarray:
    """
    Inverts a remapping index field.

    Requires the assumption that the remapping index field is invertible or bijective/one-to-one and non-occluding.
    Defined 'remap_AB' as a remapping index field that warps image A onto image B, then 'remap_BA' is the remapping index field that warps image B onto image A. This function computes 'remap_BA' given 'remap_AB'.

    RH 2023

    Args:
        remappingIdx (np.ndarray): 
            An array of shape *(H, W, 2)* representing the remap field.
        method (str):
            Interpolation method to use. See ``scipy.interpolate.griddata``. Options are:
            \n
            * ``'linear'``
            * ``'nearest'``
            * ``'cubic'`` \n
            (Default is ``'linear'``)
        fill_value (Optional[float]):
            Value used to fill points outside the convex hull. 
            (Default is ``np.nan``)

    Returns:
        (np.ndarray): 
                An array of shape *(H, W, 2)* representing the inverse remap field.
    """
    H, W, _ = remappingIdx.shape
    
    # Create the meshgrid of the original image
    grid = np.mgrid[:H, :W][::-1].transpose(1,2,0).reshape(-1, 2)
    
    # Flatten the original meshgrid and remappingIdx
    remapIdx_flat = remappingIdx.reshape(-1, 2)
    
    # Interpolate the inverse mapping using griddata
    map_BA = scipy.interpolate.griddata(
        points=remapIdx_flat, 
        values=grid, 
        xi=grid, 
        method=method,
        fill_value=fill_value,
    ).reshape(H,W,2)
    
    return map_BA

def invert_warp_matrix(
    warp_matrix: np.ndarray
) -> np.ndarray:
    """
    Inverts a provided warp matrix for the transformation A->B to compute the
    warp matrix for B->A.
    RH 2023

    Args:
        warp_matrix (np.ndarray): 
            A 2x3 or 3x3 array representing the warp matrix. Shape: *(2, 3)* or
            *(3, 3)*.

    Returns:
        (np.ndarray): 
            inverted_warp_matrix (np.ndarray):
                The inverted warp matrix. Shape: same as input.
    """
    if warp_matrix.shape == (2, 3):
        # Convert 2x3 affine warp matrix to 3x3 by appending [0, 0, 1] as the last row
        warp_matrix_3x3 = np.vstack((warp_matrix, np.array([0, 0, 1])))
    elif warp_matrix.shape == (3, 3):
        warp_matrix_3x3 = warp_matrix
    else:
        raise ValueError("Input warp_matrix must be of shape (2, 3) or (3, 3)")

    # Compute the inverse of the 3x3 warp matrix
    inverted_warp_matrix_3x3 = np.linalg.inv(warp_matrix_3x3)

    if warp_matrix.shape == (2, 3):
        # Convert the inverted 3x3 warp matrix back to 2x3 by removing the last row
        inverted_warp_matrix = inverted_warp_matrix_3x3[:2, :]
    else:
        inverted_warp_matrix = inverted_warp_matrix_3x3

    return inverted_warp_matrix


def compose_remappingIdx(
    remap_AB: np.ndarray,
    remap_BC: np.ndarray,
    method: str = 'linear',
    fill_value: Optional[float] = np.nan,
    bounds_error: bool = False,
) -> np.ndarray:
    """
    Composes two remapping index fields using scipy.interpolate.interpn.
    
    This function computes 'remap_AC' from 'remap_AB' and 'remap_BC', where
    'remap_AB' is a remapping index field that warps image A onto image B, and
    'remap_BC' is a remapping index field that warps image B onto image C.
    
    RH 2023

    Args:
        remap_AB (np.ndarray): 
            An array of shape *(H, W, 2)* representing the remap field from
            image A to image B.
        remap_BC (np.ndarray): 
            An array of shape *(H, W, 2)* representing the remap field from
            image B to image C.
        method (str): 
            Interpolation method to use. Either \n
            * ``'linear'``: Use linear interpolation (default).
            * ``'nearest'``: Use nearest interpolation.
            * ``'cubic'``: Use cubic interpolation.
        fill_value (Optional[float]): 
            The value used for points outside the interpolation domain. (Default
            is ``np.nan``)
        bounds_error (bool):
            If ``True``, a ValueError is raised when interpolated values are
            requested outside of the domain of the input data. (Default is
            ``False``)
    
    Returns:
        (np.ndarray): 
            remap_AC (np.ndarray): 
                An array of shape *(H, W, 2)* representing the remap field from
                image A to image C.
    """
    # Get the shape of the remap fields
    H, W, _ = remap_AB.shape
    
    # Combine the x and y components of remap_AB into a complex number
    # This is done to simplify the interpolation process
    AB_complex = remap_AB[:,:,0] + remap_AB[:,:,1]*1j

    # Perform the interpolation using interpn
    AC = scipy.interpolate.interpn(
        (np.arange(H), np.arange(W)), 
        AB_complex, 
        remap_BC.reshape(-1, 2)[:, ::-1], 
        method=method, 
        bounds_error=bounds_error, 
        fill_value=fill_value
    ).reshape(H, W)

    # Split the real and imaginary parts of the interpolated result to get the x and y components
    remap_AC = np.stack((AC.real, AC.imag), axis=-1)

    return remap_AC


def compose_transform_matrices(
    matrix_AB: np.ndarray, 
    matrix_BC: np.ndarray,
) -> np.ndarray:
    """
    Composes two transformation matrices to create a transformation from one
    image to another. RH 2023
    
    This function is used to combine two transformation matrices, 'matrix_AB'
    and 'matrix_BC'. 'matrix_AB' represents a transformation that warps an image
    A onto an image B. 'matrix_BC' represents a transformation that warps image
    B onto image C. The result is 'matrix_AC', a transformation matrix that
    would warp image A directly onto image C.
    
    Args:
        matrix_AB (np.ndarray): 
            A transformation matrix from image A to image B. The array can have
            the shape *(2, 3)* or *(3, 3)*.
        matrix_BC (np.ndarray): 
            A transformation matrix from image B to image C. The array can have
            the shape *(2, 3)* or *(3, 3)*.

    Returns:
        (np.ndarray): 
            matrix_AC (np.ndarray):
                A composed transformation matrix from image A to image C. The
                array has the shape *(2, 3)* or *(3, 3)*.

    Raises:
        AssertionError: 
            If the input matrices do not have the shape *(2, 3)* or *(3, 3)*.

    Example:
        .. highlight:: python
        .. code-block:: python

            # Define the transformation matrices
            matrix_AB = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            matrix_BC = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            # Compose the transformation matrices
            matrix_AC = compose_transform_matrices(matrix_AB, matrix_BC)
    """
    assert matrix_AB.shape in [(2, 3), (3, 3)], "Matrix AB must be of shape (2, 3) or (3, 3)."
    assert matrix_BC.shape in [(2, 3), (3, 3)], "Matrix BC must be of shape (2, 3) or (3, 3)."

    # If the input matrices are (2, 3), extend them to (3, 3) by adding a row [0, 0, 1]
    if matrix_AB.shape == (2, 3):
        matrix_AB = np.vstack((matrix_AB, [0, 0, 1]))
    if matrix_BC.shape == (2, 3):
        matrix_BC = np.vstack((matrix_BC, [0, 0, 1]))

    # Compute the product of the extended matrices
    matrix_AC = matrix_AB @ matrix_BC

    # If the resulting matrix is (3, 3) and has the last row [0, 0, 1], convert it back to a (2, 3) matrix
    if (matrix_AC.shape == (3, 3)) and np.allclose(matrix_AC[2], [0, 0, 1]):
        matrix_AC = matrix_AC[:2, :]

    return matrix_AC


def _make_idx_grid(
    im: Union[np.ndarray, object],
) -> Union[np.ndarray, object]:
    """
    Helper function to make a grid of indices for an image. Used in
    ``flowField_to_remappingIdx`` and ``remappingIdx_to_flowField``.

    Args:
        im (Union[np.ndarray, object]): 
            An image represented as a numpy ndarray or torch Tensor.

    Returns:
        (Union[np.ndarray, object]):
            idx_grid (Union[np.ndarray, object]):
                Index grid for the given image.
    """
    if isinstance(im, torch.Tensor):
        stack, meshgrid, arange = partial(torch.stack, dim=-1), partial(torch.meshgrid, indexing='xy'), partial(torch.arange, device=im.device, dtype=im.dtype)
    elif isinstance(im, np.ndarray):
        stack, meshgrid, arange = partial(np.stack, axis=-1), partial(np.meshgrid, indexing='xy'), partial(np.arange, dtype=im.dtype)
    return stack(meshgrid(arange(im.shape[1]), arange(im.shape[0]))) # (H, W, 2). Last dimension is (x, y).
def flowField_to_remappingIdx(
    ff: Union[np.ndarray, object],
) -> Union[np.ndarray, object]:
    """
    Converts a flow field into a remapping index by adding the pixel grid. RH 2023

    **WARNING**: Strictly speaking, a flow field (displacement) and a
    remapping index (interpolation mapping) are different concepts; this
    helper performs the obvious sum and is correct under the standard
    convention.

    Args:
        ff (Union[np.ndarray, torch.Tensor]):
            Flow field describing the displacement of each pixel. shape:
            *(H, W, 2)*. Last dimension is *(x, y)*.

    Returns:
        (Union[np.ndarray, torch.Tensor]):
            ri (Union[np.ndarray, torch.Tensor]):
                Remapping index of source pixel coordinates. shape: *(H, W, 2)*.
    """
    ri = ff + _make_idx_grid(ff)
    return ri
def remappingIdx_to_flowField(
    ri: Union[np.ndarray, object],
) -> Union[np.ndarray, object]:
    """
    Converts a remapping index into a flow field by subtracting the pixel grid. RH 2023

    **WARNING**: Strictly speaking, a remapping index (interpolation
    mapping) and a flow field (displacement) are different concepts; this
    helper performs the obvious subtraction.

    Args:
        ri (Union[np.ndarray, torch.Tensor]):
            Remapping index. shape: *(H, W, 2)*. Last dimension is
            *(x, y)*.

    Returns:
        (Union[np.ndarray, torch.Tensor]):
            ff (Union[np.ndarray, torch.Tensor]):
                Flow field. shape: *(H, W, 2)*.
    """
    ff = ri - _make_idx_grid(ri)
    return ff
def cv2RemappingIdx_to_pytorchFlowField(
    ri: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Converts remapping indices from the OpenCV format to the PyTorch format. In
    the OpenCV format, the displacement is in pixels relative to the top left
    pixel of the image. In the PyTorch format, the displacement is in pixels
    relative to the center of the image. RH 2023

    Args:
        ri (Union[np.ndarray, torch.Tensor]): 
            Remapping indices. Each pixel describes the index of the pixel in
            the original image that should be mapped to the new pixel. Shape:
            *(H, W, 2)*. The last dimension is (x, y).
        
    Returns:
        (Union[np.ndarray, torch.Tensor]): 
            normgrid (Union[np.ndarray, torch.Tensor]): 
                "Flow field", in the PyTorch format. Technically not a flow
                field, since it doesn't describe displacement. Rather, it is a
                remapping index relative to the center of the image. Shape: *(H,
                W, 2)*. The last dimension is (x, y).
    """
    assert isinstance(ri, torch.Tensor), f"ri must be a torch.Tensor. Got {type(ri)}"
    im_shape = torch.flipud(torch.as_tensor(ri.shape[:2], dtype=torch.float32, device=ri.device))  ## (W, H)
    normgrid = ((ri / (im_shape[None, None, :] - 1)) - 0.5) * 2  ## PyTorch's grid_sample expects grid values in [-1, 1] because it's a relative offset from the center pixel. CV2's remap expects grid values in [0, 1] because it's an absolute offset from the top-left pixel.
    ## note also that pytorch's grid_sample expects align_corners=True to correspond to cv2's default behavior.
    return normgrid

def remap_points(
    points: np.ndarray, 
    remappingIdx: np.ndarray,
    interpolation: str = 'linear',
    fill_value: float = None,
) -> np.ndarray:
    """
    Remaps a set of 2D points through an index map produced for image warping.

    Args:
        points (np.ndarray):
            Array of points to be remapped. shape: *(n_points, 2)*, dtype:
            *floating*. Each row is an ``(x, y)`` coordinate within the
            image.
        remappingIdx (np.ndarray):
            Index map describing the warp. shape: *(height, width, 2)*,
            dtype: *floating*.
        interpolation (str):
            Interpolation method passed to
            ``scipy.interpolate.RegularGridInterpolator``. One of \n
            * ``'linear'``
            * ``'nearest'``
            * ``'slinear'``
            * ``'cubic'``
            * ``'quintic'``
            * ``'pchip'`` \n
            (Default is ``'linear'``)
        fill_value (Optional[float]):
            Value used to fill points outside the convex hull. If ``None``,
            values outside the convex hull are extrapolated. (Default is
            ``None``)

    Returns:
        (np.ndarray):
            points_remap (np.ndarray):
                Remapped points. shape: *(n_points, 2)*.
    """
    ### Assert points is a 2D numpy.ndarray of shape (n_points, 2) and that all points are within the image and that points are float
    assert isinstance(points, np.ndarray), 'points must be a numpy.ndarray'
    assert points.ndim == 2, 'points must be a 2D numpy.ndarray'
    assert points.shape[1] == 2, 'points must be of shape (n_points, 2)'
    assert np.issubdtype(points.dtype, np.floating), 'points must be a float subtype'

    assert isinstance(remappingIdx, np.ndarray), 'remappingIdx must be a numpy.ndarray'
    assert remappingIdx.ndim == 3, 'remappingIdx must be a 3D numpy.ndarray'
    assert remappingIdx.shape[2] == 2, 'remappingIdx must be of shape (height, width, 2)'
    assert np.issubdtype(remappingIdx.dtype, np.floating), 'remappingIdx must be a float subtype'

    ## Make grid of indices for image remapping
    dims = remappingIdx.shape
    x_arange, y_arange = np.arange(0., dims[1]).astype(np.float32), np.arange(0., dims[0]).astype(np.float32)

    ## Use RegularGridInterpolator to remap points
    warper = scipy.interpolate.RegularGridInterpolator(
        points=(y_arange, x_arange),
        values=remappingIdx,
        method=interpolation,
        bounds_error=False,
        fill_value=fill_value,
    )
    points_remap = warper(xi=(points[:, 1], points[:, 0]))

    return points_remap



##########################################################################################################################################
########################################################### RESOURCE TRACKING ############################################################
##########################################################################################################################################


import datetime
from threading import Timer
from pathlib import Path

import psutil

class _Device_Checker_Base():
    """
    Base class for periodic resource utilization tracking.

    Subclasses must implement ``check_utilization`` returning a dict of
    measurement values keyed by metric name.

    Args:
        verbose (int):
            Verbosity level. ``0`` silences output, ``1`` prints basic
            statements and warnings. (Default is ``1``)

    Attributes:
        log (Dict[str, Dict[int, Any]]):
            Per-metric history populated by ``log_utilization``.
        fn_timer (_RepeatTimer):
            Background repeating timer created by ``track_utilization``.
    """
    def __init__(self, verbose=1):
        """Stores the verbosity level for the checker."""
        self._verbose = int(verbose)

    def log_utilization(self, path_save=None):
        """
        Records one utilization snapshot, optionally appending to a CSV file.

        Args:
            path_save (Optional[str]):
                If provided, appends the snapshot as a row to a ``.csv``
                file at this path (creating it with a header if needed).
                (Default is ``None``)

        Returns:
            (Dict[str, Dict[int, Any]]):
                log (Dict[str, Dict[int, Any]]):
                    Updated history of all collected metrics.
        """
        info_changing = self.check_utilization()
        
        if not hasattr(self, 'log'):
            self.log = {}
            self._iter_log = 0

            ## Populate with keys
            for key in info_changing.keys():
                self.log[key] = {}
            print(f'Created self.log with keys: {self.log.keys()}') if self._verbose > 0 else None
        else:
            assert hasattr(self, '_iter_log'), 'self.log exists but self._iter_log does not'
            self._iter_log += 1

        ## Populate with values
        for key in info_changing.keys():
            self.log[key][self._iter_log] = info_changing[key]

        ## Save
        if path_save is not None:
            assert path_save.endswith('.csv'), 'path_save must be a .csv file'
            ## Check if file exists
            if not Path(path_save).exists():
                ## Make a .csv file with header
                with open(path_save, 'w') as f:
                    f.write(','.join(self.log.keys()) + '\n')
                ## Append to file
                with open(path_save, 'a') as f:
                    f.write(','.join([str(info_changing[key]) for key in self.log.keys()]) + '\n')
            ## Append to file
            else:
                with open(path_save, 'a') as f:
                    f.write(','.join([str(info_changing[key]) for key in self.log.keys()]) + '\n')

        return self.log
    
    
    def track_utilization(
        self,
        interval=0.2,
        path_save=None,
    ):
        """
        Begins periodic logging on a background thread.

        Replaces any existing tracker and stores the new one as
        ``self.fn_timer``. Each call to ``log_utilization`` happens every
        ``interval`` seconds.

        Args:
            interval (float):
                Polling interval in seconds. ``0.2`` is the minimum useful
                value. (Default is ``0.2``)
            path_save (Optional[str]):
                ``.csv`` path to which each snapshot is appended. ``None``
                disables saving. (Default is ``None``)
        """
        self.stop_tracking()
        ## Make a background thread that runs self.log_utilization() every interval seconds
        def log_utilization_thread():
            self.log_utilization(path_save=path_save)

        self.fn_timer = _RepeatTimer(interval, log_utilization_thread)
        self.fn_timer.start()
        
    def stop_tracking(self):
        """Cancels the background ``fn_timer`` thread (if one is running)."""
        if hasattr(self, 'fn_timer'):
            self.fn_timer.cancel()

    def __del__(self):
        """Stops the background tracker before the checker is garbage-collected."""
        self.stop_tracking()


class NVIDIA_Device_Checker(_Device_Checker_Base):
    """
    Resource utilization checker for an NVIDIA GPU.

    Requires the ``nvidia-ml-py3`` package.

    Args:
        device_index (Optional[int]):
            Index of the GPU to monitor. If ``None`` and only one device is
            present, that device is used; otherwise an error is raised.
            (Default is ``None``)
        verbose (int):
            Verbosity level passed to the base class. (Default is ``1``)

    Attributes:
        info_static (Dict[str, Any]):
            Static device info (name, index, total memory, power limit).
        handle (object):
            ``nvidia_smi`` device handle for the monitored GPU.
    """
    def __init__(self, device_index=None, verbose=1):
        """Initializes NVML, looks up device handles, and captures static device info."""
        try:
            import nvidia_smi
        except ImportError:
            raise ImportError('nvidia_smi package not found. Install with "pip install nvidia-ml-py3"')
        self.nvidia_smi = nvidia_smi
        super().__init__(verbose=verbose)
        
        ## Initialize
        nvidia_smi.nvmlInit()  ## This is needed to get device info

        ## Get device handles
        self._handles_allDevices = self.get_device_handles()
        n_device = len(self._handles_allDevices)
        if n_device == 1:
            self.handle = self._handles_allDevices[0]
            self.device_index = 0
            print(f'Found one device. Setting self.device_index to 0.') if self._verbose > 0 else None
        else:
            assert isinstance(device_index, int), 'Device index must be specified since multiple devices were found'
            assert device_index < n_device, f'Device index specified is greater tban the number of devices found: {n_device}'  
        
        ## Get static info
        self.info_static = {}
        self.info_static['device_name']  = nvidia_smi.nvmlDeviceGetName(self.handle)
        self.info_static['device_index'] = nvidia_smi.nvmlDeviceGetIndex(self.handle)
        self.info_static['memory_total'] = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle).total
        self.info_static['power_limit']  = nvidia_smi.nvmlDeviceGetPowerManagementLimit(self.handle)
    
    def get_device_handles(self):
        """Returns one ``nvmlDeviceGetHandleByIndex`` handle per GPU detected by NVML."""
        nvidia_smi = self.nvidia_smi
        return [nvidia_smi.nvmlDeviceGetHandleByIndex(i_device) for i_device in range(nvidia_smi.nvmlDeviceGetCount())]

    def check_utilization(self):
        """
        Returns a snapshot of the current GPU utilization metrics.

        Returns:
            (Dict[str, Any]):
                info_changing (Dict[str, Any]):
                    Includes ``time``, ``memory_free``, ``memory_used``,
                    ``memory_used_percentage``, ``power_used``,
                    ``power_used_percentage``, ``processor_used_percentage``,
                    ``temperature``, and ``fan_speed``.
        """
        nvidia_smi = self.nvidia_smi
        h = self.handle
        info_mem = nvidia_smi.nvmlDeviceGetMemoryInfo(h)

        info_changing = {}
        
        info_changing['time'] = datetime.datetime.now()
        
        info_changing['memory_free'] = info_mem.free
        info_changing['memory_used'] = info_mem.used
        info_changing['memory_used_percentage'] = 100 * info_mem.used / info_mem.total

        info_changing['power_used'] = nvidia_smi.nvmlDeviceGetPowerUsage(h)
        info_changing['power_used_percentage'] = 100* info_changing['power_used'] / nvidia_smi.nvmlDeviceGetPowerManagementLimit(h)

        info_changing['processor_used_percentage'] = nvidia_smi.nvmlDeviceGetUtilizationRates(h).gpu

        info_changing['temperature'] = nvidia_smi.nvmlDeviceGetTemperature(h, nvidia_smi.NVML_TEMPERATURE_GPU)

        info_changing['fan_speed'] = nvidia_smi.nvmlDeviceGetFanSpeed(h)

        return info_changing
    
    def __del__(self):
        """Shuts down NVML and cancels any background tracker before destruction."""
        nvidia_smi = self.nvidia_smi
        nvidia_smi.nvmlShutdown()  ## This stops the ability to get device info
        super().__del__()


class CPU_Device_Checker(_Device_Checker_Base):
    """
    Resource utilization checker for the host CPU and disk.

    Args:
        verbose (int):
            Verbosity level passed to the base class. (Default is ``1``)

    Attributes:
        info_static (Dict[str, Any]):
            Static info (CPU count, frequency, total RAM, total disk).
    """
    def __init__(self, verbose=1):
        """Captures static CPU, memory, and disk info."""
        super().__init__(verbose=verbose)

        self.info_static = {}
        
        self.info_static['cpu_count'] = psutil.cpu_count()
        self.info_static['cpu_freq'] = psutil.cpu_freq()
        
        self.info_static['memory_total'] = psutil.virtual_memory().total
        
        self.info_static['disk_total'] = psutil.disk_usage('/').total

    def check_utilization(self):
        """
        Returns a snapshot of CPU, memory, network, and disk utilization.

        Returns:
            (Dict[str, Any]):
                info_changing (Dict[str, Any]):
                    Per-snapshot metrics including memory, network I/O, disk
                    free/used, disk read/write throughput, and overall +
                    per-core CPU usage percentages.
        """
        info_changing = {}
        
        info_changing['time'] = datetime.datetime.now()
        
        ## log cpu utilization (per cpu), memory utilization, network utilization, disk utilization, etc
        info_changing['memory_used_percentage'] = psutil.virtual_memory().percent
        info_changing['memory_used'] = psutil.virtual_memory().used
        info_changing['memory_free'] = psutil.virtual_memory().free
        info_changing['memory_available'] = psutil.virtual_memory().available
        info_changing['memory_active'] = psutil.virtual_memory().active
        info_changing['memory_inactive'] = psutil.virtual_memory().inactive
        info_changing['memory_buffers'] = psutil.virtual_memory().buffers
        info_changing['memory_cached'] = psutil.virtual_memory().cached
        info_changing['memory_shared'] = psutil.virtual_memory().shared
        ## Get network info: current bytes sent and received
        info_changing['network_sent'] = psutil.net_io_counters().bytes_sent
        info_changing['network_received'] = psutil.net_io_counters().bytes_recv
        ## Get disk info: free space and used space and percentage
        info_changing['disk_free'] = psutil.disk_usage('/').free
        info_changing['disk_used'] = psutil.disk_usage('/').used
        info_changing['disk_used_percentage'] = psutil.disk_usage('/').percent
        ## Get disk read/write info
        info_changing['disk_read'] = psutil.disk_io_counters().read_bytes
        info_changing['disk_write'] = psutil.disk_io_counters().write_bytes
        ## Get processor info: current processor utilization (overall and per core)
        info_changing['processor_used_percentage'] = psutil.cpu_percent()
        for i_core, val in enumerate(psutil.cpu_percent(percpu=True)):
            info_changing[f'cpu_{i_core}'] = val

        return info_changing


class _RepeatTimer(Timer):
    """Subclass of ``threading.Timer`` that re-fires its callback every ``interval`` seconds."""
    def run(self):
        """Calls ``self.function`` repeatedly until ``self.finished`` is set."""
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)


##########################################################################################################################################
########################################################### TESTING ######################################################################
##########################################################################################################################################

class Equivalence_checker():
    """
    Class for checking if all items are equivalent or allclose (almost equal) in
    two complex data structures. Can check nested lists, dicts, and other data
    structures. Can also optionally assert (raise errors) if all items are not
    equivalent. 
    RH 2023

    Attributes:
        _kwargs_allclose (Optional[dict]): 
            Keyword arguments for the `numpy.allclose` function.
        _assert_mode (bool):
            Whether to raise an assertion error if items are not close.

    Args:
        kwargs_allclose (Optional[dict]): 
            Keyword arguments for the `numpy.allclose` function. (Default is
            ``{'rtol': 1e-7, 'equal_nan': True}``)
        assert_mode (bool): 
            Whether to raise an assertion error if items are not close.
        verbose (bool):
            How much information to print out:
                * ``False`` / ``0``: No information printed out.
                * ``True`` / ``1``: Mismatched items only.
                * ``2``: All items printed out.
    """
    def __init__(
        self,
        kwargs_allclose: Optional[dict] = {'rtol': 1e-7, 'equal_nan': True},
        assert_mode=False,
        verbose=False,
    ) -> None:
        """Stores comparison settings for later use by ``__call__``."""
        self._kwargs_allclose = kwargs_allclose
        self._assert_mode = assert_mode
        self._verbose = verbose
        
    def _checker(
        self, 
        test: Any,
        true: Any, 
        path: Optional[List[str]] = None,
    ) -> bool:
        """
        Compares the test and true values using numpy's allclose function.

        Args:
            test (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                Test value to compare.
            true (Union[dict, list, tuple, set, np.ndarray, int, float, complex,
            str, bool, None]): 
                True value to compare.
            path (Optional[List[str]]): 
                The path of the data structure that is currently being compared.
                (Default is ``None``)

        Returns:
            (bool): 
                result (bool): 
                    Returns True if all elements in test and true are close.
                    Otherwise, returns False.
        """
        try:
            ## If the dtype is a kind of string (or byte string) or object, then allclose will raise an error. In this case, just check if the values are equal.
            if np.issubdtype(test.dtype, np.str_) or np.issubdtype(test.dtype, np.bytes_) or test.dtype == np.object_:
                out = bool(np.all(test == true))
                print(f"Equivalence check {'passed' if out else 'failed'}. Path: {path}.") if self._verbose > 1 else None
            else:
                out = np.allclose(test, true, **self._kwargs_allclose)
            print(f"Equivalence check passed. Path: {path}") if self._verbose > 1 else None
        except Exception as e:
            out = None  ## This is not False because sometimes allclose will raise an error if the arrays have a weird dtype among other reasons.
            warnings.warn(f"WARNING. Equivalence check failed. Path: {path}. Error: {e}") if self._verbose else None
            
        if out == False:
            if self._assert_mode:
                raise AssertionError(f"Equivalence check failed. Path: {path}.")
            if self._verbose:
                ## Come up with a way to describe the difference between the two values. Something like the following:
                ### IF the arrays are numeric, then calculate the relative difference
                dtypes_numeric = (np.number, np.bool_, np.integer, np.floating, np.complexfloating)
                if any([np.issubdtype(test.dtype, dtype) and np.issubdtype(true.dtype, dtype) for dtype in dtypes_numeric]):
                    ## numpy 1.25+ removed the `-` operator on bool arrays; cast first.
                    test_n = test.astype(np.int8) if test.dtype == bool else test
                    true_n = true.astype(np.int8) if true.dtype == bool else true
                    diff = np.abs(test_n - true_n)
                    r_diff = diff / np.abs(true_n)
                    r_diff_mean, r_diff_max, any_nan = np.nanmean(r_diff), np.nanmax(r_diff), np.any(np.isnan(r_diff))
                    print(f"Equivalence check failed. Path: {path}. Relative difference: mean={r_diff_mean}, max={r_diff_max}, any_nan={any_nan}") if self._verbose > 0 else None
                else:
                    print(f"Equivalence check failed. Path: {path}. Value is non-numerical.") if self._verbose > 0 else None
        return out

    def __call__(
        self,
        test: Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None],
        true: Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None],
        path: Optional[List[str]] = None,
    ) -> Dict[str, Tuple[bool, str]]:
        """
        Recursively compares ``test`` to ``true`` and returns the result.

        Handles dictionaries, iterables, ``np.ndarray``, scalars, strings,
        numbers, booleans, and ``None``.

        Args:
            test (Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None]):
                Test value to compare.
            true (Union[dict, list, tuple, set, np.ndarray, int, float, complex, str, bool, None]):
                Reference value.
            path (Optional[List[str]]):
                Hierarchical path within the data structure currently being
                compared (used in recursion). (Default is ``None``)

        Returns:
            (Union[Tuple[Optional[bool], str], Dict[str, Any]]):
                result (Union[Tuple[Optional[bool], str], Dict[str, Any]]):
                    For leaf values, a ``(passed, reason)`` tuple. For
                    container values, a nested dictionary keyed by the
                    container's keys / indices.
        """
        if path is None:
            path = ['']

        if len(path) > 0:
            if path[-1].startswith('_'):
                return (None, 'excluded from testing')

        ## NP.NDARRAY
        if isinstance(true, np.ndarray):
            r = self._checker(test, true, path)
            result = (r, 'equivalence')
        ## NP.SCALAR
        elif np.isscalar(true):
            if isinstance(test, (int, float, complex, np.number)):
                r = self._checker(np.array(test), np.array(true), path)
                result = (r, 'equivalence')
            else:
                result = (test == true, 'equivalence')
        ## NUMBER
        elif isinstance(true, (int, float, complex)):
            r = self._checker(test, true, path)
            result = (result, 'equivalence')
        ## DICT
        elif isinstance(true, dict):
            result = {}
            for key in true:
                if key not in test:
                    result[str(key)] = (False, 'key not found')
                    print(f"Equivalence check failed. Path: {path}. Key {key} not found.") if self._verbose > 0 else None
                else:
                    result[str(key)] = self.__call__(test[key], true[key], path=path + [str(key)])
        ## ITERATABLE
        elif isinstance(true, (list, tuple, set)):
            if len(true) != len(test):
                result = (False, 'length_mismatch')
                print(f"Equivalence check failed. Path: {path}. Length mismatch.") if self._verbose > 0 else None
            else:
                result = {}
                for idx, (i, j) in enumerate(zip(test, true)):
                    result[str(idx)] = self.__call__(i, j, path=path + [str(idx)])
        ## STRING
        elif isinstance(true, str):
            result = (test == true, 'equivalence')
            print(f"Equivalence check {'passed' if result[0] else 'failed'}. Path: {path}.") if self._verbose > 0 else None
        ## BOOL
        elif isinstance(true, bool):
            result = (test == true, 'equivalence')
            print(f"Equivalence check {'passed' if result[0] else 'failed'}. Path: {path}.") if self._verbose > 0 else None
        ## NONE
        elif true is None:
            result = (test is None, 'equivalence')
            print(f"Equivalence check {'passed' if result[0] else 'failed'}. Path: {path}.") if self._verbose > 0 else None
        ## N/A
        else:
            result = (None, 'not tested')
            print(f"Equivalence check not performed. Path: {path}.") if self._verbose > 0 else None

        return result


##########################################################################################################################################
######################################################### SIMILARITY #####################################################################
##########################################################################################################################################


def order_cp_factors_by_EVR(
    tensor_dense: Union[np.ndarray, torch.Tensor],
    cp_factors: Union[list, object],
    cp_weights: Optional[Union[np.ndarray, torch.Tensor]] = None,
    orthogonalizable_EVR: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sorts CP factors by descending explained variance ratio. RH 2024

    Args:
        tensor_dense (Union[np.ndarray, torch.Tensor]):
            Dense tensor to be reconstructed.
        cp_factors (Union[list, object]):
            CP factors. Either a list of 2D factor matrices of shape
            *(n_samples, rank)* or a ``tensorly.CPTensor`` object.
        cp_weights (Optional[Union[np.ndarray, torch.Tensor]]):
            Per-rank weights of length *(rank,)*. (Default is ``None``)
        orthogonalizable_EVR (bool):
            If ``True``, optimizes each factor's scaling to maximize EVR by
            OLS-orthogonalizing the dense tensor against each factor.
            (Default is ``True``)

    Returns:
        (tuple): tuple containing:
            order (np.ndarray):
                Indices that sort the factors by descending EVR.
            evrs (np.ndarray):
                Sorted explained variance ratios.
    """
    if isinstance(cp_factors, list):
        # If cp_factors is a list, use it directly
        factors = cp_factors
        if cp_weights is not None:
            # Apply weights to each factor if cp_weights is provided
            factors = [factors[ii] * cp_weights[ii] for ii in range(len(factors))]
    elif isinstance(cp_factors, tl.cp_tensor.CPTensor):
        # If cp_factors is a CPTensor, extract factors and apply weights
        factors = [f * cp_factors.weights[None, :] for f in cp_factors.factors]
    else:
        raise ValueError('tensor_CP must be a list of factors or a tensorly CPTensor object')
    
    if orthogonalizable_EVR:
        # Flatten tensor_dense and remove its mean
        tensor_dense = tensor_dense.reshape(-1)
        tensor_dense -= tensor_dense.mean()
        tensor_dense_var = tensor_dense.var()  # Compute variance of tensor_dense

    # Determine the rank from the shape of the first factor
    rank = factors[0].shape[1]
    evrs = []  # Initialize list to store explained variance ratios (EVRs)

    for ii in range(rank):
        # Extract the ii-th component from each factor and reshape to be a column vector
        f = [f[:, ii][:, None] for f in factors]
        if orthogonalizable_EVR:
            # Convert CP components back to a dense tensor and flatten
            v2 = cp_to_dense(f).reshape(-1)
            v2 = v2 - v2.mean()  # Remove mean of v2
            # Calculate orthogonal component of tensor_dense with respect to v2
            v1_orth = tensor_dense - ((tensor_dense * v2).sum() / (v2 * v2).sum() )*v2
            # Compute EVR as the fraction of variance explained by v2
            evr = 1 - (v1_orth.var() / tensor_dense_var)
        else:
            # Compute EVR using a predefined function for the non-orthogonalizable case
            evr = cp_reconstruction_EVR(tensor_dense, f)
        evrs.append(evr)  # Append the computed EVR to the list

    # Sort the EVRs in descending order and return the sorted order and EVRs
    order = np.argsort(evrs)[::-1]
    return order, np.array(evrs)[order]


def cp_reconstruction_EVR(tensor_dense, tensor_CP):
    """
    Explained variance ratio of a CP-reconstructed tensor. RH 2023

    Args:
        tensor_dense (Union[np.ndarray, torch.Tensor]):
            Dense reference tensor. shape: *(n_samples, n_features)*.
        tensor_CP (Union[list, object]):
            CP tensor. Either a list of 2D factor matrices of shape
            *(n_samples, rank)* or a ``tensorly.CPTensor`` object.

    Returns:
        (Union[float, torch.Tensor]):
            ev (Union[float, torch.Tensor]):
                Explained variance ratio
                ``1 - var(tensor_dense - tensor_rec) / var(tensor_dense)``.
    """
    tensor_rec = None
    try:
        import tensorly as tl
        if isinstance(tensor_CP, tl.cp_tensor.CPTensor):
            tensor_rec = tl.cp_to_tensor(tensor_CP)
    except ImportError as e:
        raise ImportError('tensorly not installed. Please install tensorly or provide a list of factors as the tensor_CP argument.')
    
    if tensor_rec is None:
        assert isinstance(tensor_CP, list), 'tensor_CP must be a list of factors'
        assert all([isinstance(f, (np.ndarray, torch.Tensor)) for f in tensor_CP]), 'tensor_CP must be a list of factors'
        tensor_rec = cp_to_dense(tensor_CP)

    if isinstance(tensor_dense, torch.Tensor):
        var = torch.var
    elif isinstance(tensor_dense, np.ndarray):
        var = np.var
    ev = 1 - (var(tensor_dense - tensor_rec) / var(tensor_dense))
    return ev


#####################################################################################################################################
######################################################### OTHER #####################################################################
#####################################################################################################################################

import torch
from typing import Union

def rolling_mean(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Computes the running mean along ``dim`` using Welford's update. RH 2025

    Args:
        tensor (torch.Tensor):
            Input tensor.
        dim (int):
            Dimension along which the running mean is accumulated.

    Returns:
        (torch.Tensor):
            mean (torch.Tensor):
                Final mean across ``dim`` (last accumulated value).
    """
    # Ensure the dimension is non-negative and valid.
    if dim < 0:
        dim += tensor.dim()
    if dim < 0 or dim >= tensor.dim():
        raise ValueError(f"Invalid dimension {dim} for tensor with {tensor.dim()} dimensions.")
    
    # Unbind the tensor along the given dimension to get a list of slices.
    dims_permute = list(range(tensor.dim()))
    ## remove dim from the list
    dims_permute.remove(dim)
    dims_permute = [dim] + dims_permute
        
    # Initialize an empty list to store the running means.
    current_mean = None
    
    # Iterate through each slice along the given dimension.
    # Use a counter starting at 1 since we divide by the count.
    for i, slice in enumerate(tensor.permute(dims_permute)):
        if current_mean is None:
            # For the first element, the running mean is the element itself.
            current_mean = slice
        else:
            # Update the running mean using:
            current_mean = current_mean + (slice - current_mean) / (i + 1)
    
    # Stack the list of running means back into a tensor along the specified dimension.
    return current_mean


## Video helpers — ported from bnpm.video (bnpm 0.7.1, RH 2021/2024).
## Used by the demo_event_alignment notebook for tiled trial playback.
## bnpm is a personal-utility repo; these are the only pieces face-rhythm needs.


def play_video_cv2(
    array=None,
    path_video=None,
    frameRate=30,
    path_save=None,
    show=True,
    fourcc_code='MJPG',
    text=None,
    kwargs_text={},
):
    """
    Plays or saves a video using OpenCV. RH 2021/2024

    Args:
        array (Optional[np.ndarray]):
            3D ``(frames, H, W)`` or 4D ``(frames, H, W, channels)`` ``uint8``
            array. Values are clipped to ``[0, 255]``. If ``None``,
            ``path_video`` must be supplied and ``decord`` is used to read
            it. (Default is ``None``)
        path_video (Optional[Union[str, pathlib.Path]]):
            Path to a video file. Used only when ``array`` is ``None``.
            (Default is ``None``)
        frameRate (float):
            Playback / output frame rate in Hz. (Default is ``30``)
        path_save (Optional[Union[str, pathlib.Path]]):
            Destination path for the saved video. ``None`` disables saving.
            (Default is ``None``)
        show (bool):
            If ``True``, displays the video in a ``cv2`` window. (Default
            is ``True``)
        fourcc_code (str):
            FourCC codec string passed to ``cv2.VideoWriter_fourcc``.
            (Default is ``'MJPG'``)
        text (Optional[Union[str, List[str]]]):
            Text overlay. If a list, element ``i`` is drawn on frame ``i``.
            (Default is ``None``)
        kwargs_text (dict):
            Keyword arguments forwarded to ``cv2.putText``. (Default is
            ``{}``)
    """
    wait_frames = max(int((1 / frameRate) * 1000), 1)
    if path_save is not None:
        size = tuple((np.flip(array.shape[1:3])))
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        print(f'saving to file {path_save}')
        writer = cv2.VideoWriter(path_save, fourcc, frameRate, size)

    if kwargs_text is None:
        kwargs_text = {
            'org': (5, 15),
            'fontFace': 1,
            'fontScale': 1,
            'color': (255, 255, 255),
            'thickness': 1,
        }

    if array is not None:
        array[array < 0] = 0
        array[array > 255] = 255
        if array.dtype != 'uint8':
            array = array.astype('uint8')
        movie = array
        if array.ndim == 4:
            flag_convert_to_gray = True
        elif array.ndim == 3:
            flag_convert_to_gray = False
        else:
            raise Exception('Unsupported number of channels, check array shape')
    else:
        try:
            import decord
        except ImportError as e:
            _pkg = "eva_decord" if sys.platform.startswith("win") else "decord2"
            raise ImportError(
                "decord is required when array=None. "
                f"Install with: pip install {_pkg}"
            ) from e
        movie = decord.VideoReader(path_video)
        flag_convert_to_gray = False

    for i_frame, frame in enumerate(tqdm(movie)):
        if array is None:
            frame = frame.asnumpy()

        if array is not None:
            if flag_convert_to_gray:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if text is not None:
            text_frame = text[i_frame] if isinstance(text, list) else text
            frame = cv2.putText(frame, text_frame, **kwargs_text)

        if show:
            cv2.imshow('handle', np.uint8(frame))
            cv2.waitKey(wait_frames)
        if path_save is not None:
            writer.write(np.uint8(frame))

    if path_save is not None:
        writer.release()
        print('Video saved')
    if show:
        cv2.destroyWindow('handle')


def make_tiled_video_array(
    videos: List[np.ndarray],
    shape: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
):
    """
    Tiles a list of videos into a single grid video array. RH 2021/2024

    Videos are placed top-to-bottom and then left-to-right.

    Args:
        videos (List[np.ndarray]):
            List of video arrays with shape *(frames, H, W, channels)* or
            *(frames, H, W)*. All videos must share the same dtype.
        shape (Optional[Tuple[int, int]]):
            Grid layout ``(n_rows, n_cols)``. If ``None``, uses the smallest
            square grid that fits all videos. (Default is ``None``)
        verbose (bool):
            If ``True``, prints progress messages. (Default is ``True``)

    Returns:
        (np.ndarray):
            video_array (np.ndarray):
                Tiled video. shape: *(max_frames, total_H, total_W,
                channels)*.
    """
    assert isinstance(videos, list), f"videos must be a list. Got {type(videos)}"
    assert all(isinstance(v, np.ndarray) for v in videos), \
        f"All elements of videos must be numpy arrays. Got {[type(v) for v in videos]}"
    assert all(v.dtype == videos[0].dtype for v in videos), \
        f"All videos must have the same dtype. Got {[v.dtype for v in videos]}"
    if shape is not None:
        assert isinstance(shape, tuple), f"shape must be a tuple. Got {type(shape)}"
        assert len(shape) == 2, f"shape must be a 2-tuple. Got {len(shape)} elements"
        assert all(isinstance(val, int) for val in shape), \
            f"shape must contain only integers. Got {shape}"

    n_videos = len(videos)

    if shape is None:
        n_height = int(np.floor(np.sqrt(n_videos)))
        n_width = int(np.ceil(np.sqrt(n_videos)))
        shape = (n_height, n_width)
    assert shape[0] * shape[1] >= n_videos, \
        f"shape[0] * shape[1] must be >= number of videos. Got {shape[0] * shape[1]} < {n_videos}"
    if verbose:
        print(f"Making video array with shape: {shape} videos")

    for ii, video in enumerate(videos):
        assert video.ndim in [3, 4], \
            f"videos[{ii}] must be 3D or 4D. Got {video.ndim} dimensions"
        if video.ndim == 3:
            videos[ii] = video[..., None]

    n_frames, heights, widths, channels = (
        np.array([video.shape[ii] for video in videos]) for ii in range(4)
    )
    max_frames = max(n_frames)
    if verbose:
        print(f"Max video length and total frames in output video array: {max_frames}")

    dtype = videos[0].dtype
    if verbose:
        print(f"Video dtype: {dtype}")

    idx_videoArray_videos = np.array(
        [(ii % shape[0], ii // shape[0]) for ii in range(n_videos)]
    )
    if verbose:
        print(f"Video array indices: {idx_videoArray_videos}")

    widths_col = np.array([
        np.max(widths[idx_videoArray_videos[:, 1] == ii])
        for ii in np.unique(idx_videoArray_videos[:, 1])
    ])
    heights_row = np.array([
        np.max(heights[idx_videoArray_videos[:, 0] == ii])
        for ii in np.unique(idx_videoArray_videos[:, 0])
    ])

    final_shape = (max_frames, np.sum(heights_row), np.sum(widths_col), channels[0])
    video_array = np.zeros(final_shape, dtype)
    if verbose:
        print(f"Final output array shape: {video_array.shape}")

    idx_tops_rows = np.cumsum(np.concatenate(([0], heights_row)))[:-1]
    idx_tops_cols = np.cumsum(np.concatenate(([0], widths_col)))[:-1]
    if verbose:
        print(f"idx_tops_rows: {idx_tops_rows}, idx_tops_cols: {idx_tops_cols}")

    for ii, video in enumerate(videos):
        idx_top = idx_tops_rows[idx_videoArray_videos[ii, 0]]
        idx_left = idx_tops_cols[idx_videoArray_videos[ii, 1]]
        video_array[
            :video.shape[0],
            idx_top:idx_top + video.shape[1],
            idx_left:idx_left + video.shape[2],
            :,
        ] = video

    return video_array

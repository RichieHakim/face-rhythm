import numpy as np

def prepare_cv2_imshow():
    """
    This function is necessary because cv2.imshow() 
     can crash the kernel if called after importing 
     av and decord.
    RH 2022
    """
    import numpy as np
    import cv2
    test = np.zeros((1,300,400,3))
    for frame in test:
        cv2.putText(frame, "Prepping CV2", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, "Calling this figure allows cv2.imshow ", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "to work without crashing if this function", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "is called before importing av and decord", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('startup', frame)
        cv2.waitKey(100)
    cv2.destroyWindow('startup')


def get_system_versions(verbose=False):
    """
    Checks the versions of various important softwares.
    Prints those versions
    RH 2022

    Args:
        verbose (bool): 
            Whether to print the versions

    Returns:
        versions (dict):
            Dictionary of versions
    """
    ## Operating system and version
    import platform
    operating_system = str(platform.system()) + ': ' + str(platform.release()) + ', ' + str(platform.version()) + ', ' + str(platform.machine()) + ', node: ' + str(platform.node()) 
    print(f'Operating System: {operating_system}') if verbose else None

    ## Conda Environment
    import os
    conda_env = os.environ['CONDA_DEFAULT_ENV']
    print(f'Conda Environment: {conda_env}') if verbose else None

    ## Python
    import sys
    python_version = sys.version.split(' ')[0]
    print(f'Python Version: {python_version}') if verbose else None

    ## GCC
    import subprocess
    gcc_version = subprocess.check_output(['gcc', '--version']).decode('utf-8').split('\n')[0].split(' ')[-1]
    print(f'GCC Version: {gcc_version}') if verbose else None
    
    ## PyTorch
    import torch
    torch_version = str(torch.__version__)
    print(f'PyTorch Version: {torch_version}') if verbose else None

    ## Numpy
    import numpy
    numpy_version = numpy.__version__
    print(f'Numpy Version: {numpy_version}') if verbose else None

    ## OpenCV
    import cv2
    opencv_version = cv2.__version__
    print(f'OpenCV Version: {opencv_version}') if verbose else None
    # print(cv2.getBuildInformation())

    ## face-rhythm
    import face_rhythm
    faceRhythm_version = face_rhythm.__version__
    print(f'face-rhythm Version: {faceRhythm_version}') if verbose else None

    versions = {
        'face-rhythm_version': faceRhythm_version,
        'operating_system': operating_system,
        'conda_env': conda_env,
        'python_version': python_version,
        'gcc_version': gcc_version,
        'torch_version': torch_version,
        'numpy_version': numpy_version,
        'opencv_version': opencv_version,
    }

    return versions



###########################################################
######################## FROM BNPM ########################
###########################################################


###########################
###### PATH HELPERS #######
###########################

def find_paths(
    dir_outer, 
    reMatch='filename', 
    find_files=True, 
    find_folders=False, 
    depth=0, 
    natsorted=True, 
    alg_ns=None, 
):
    """
    Search for files and/or folders recursively in a directory.
    RH 2022

    Args:
        dir_outer (str):
            Path to directory to search
        reMatch (str):
            Regular expression to match
            Each path name encountered will be compared using
             re.search(reMatch, filename). If the output is not None,
             the file will be included in the output.
        find_files (bool):
            Whether to find files
        find_folders (bool):
            Whether to find folders
        depth (int):
            Maximum folder depth to search.
            depth=0 means only search the outer directory.
            depth=2 means search the outer directory and two levels
             of subdirectories below it.
        natsorted (bool):
            Whether to sort the output using natural sorting
             with the natsort package.
        alg_ns (str):
            Algorithm to use for natural sorting.
            See natsort.ns or
             https://natsort.readthedocs.io/en/4.0.4/ns_class.html
             for options.
            Default is PATH.
            Other commons are INT, FLOAT, VERSION.

    Returns:
        paths (List of str):
            Paths to matched files and/or folders in the directory
    """
    import re
    import os
    import natsort
    if alg_ns is None:
        alg_ns = natsort.ns.PATH

    def get_paths_recursive_inner(dir_inner, depth_end, depth=0):
        paths = []
        for path in os.listdir(dir_inner):
            path = os.path.join(dir_inner, path)
            if os.path.isdir(path):
                if find_folders:
                    if re.search(reMatch, path) is not None:
                        paths.append(path)
                if depth < depth_end:
                    paths += get_paths_recursive_inner(path, depth_end, depth=depth+1)
            else:
                if find_files:
                    if re.search(reMatch, path) is not None:
                        paths.append(path)
        return paths

    paths = get_paths_recursive_inner(dir_outer, depth, depth=0)
    if natsorted:
        paths = natsort.natsorted(paths, alg=alg_ns)
    return paths
        

###########################
######## INDEXING #########
########################### 

def make_batches(
    iterable, 
    batch_size=None, 
    num_batches=None, 
    min_batch_size=0, 
    return_idx=False, 
    length=None
):
    """
    Make batches of data or any other iterable.
    RH 2021

    Args:
        iterable (iterable):
            iterable to be batched
        batch_size (int):
            size of each batch
            if None, then batch_size based on num_batches
        num_batches (int):
            number of batches to make
        min_batch_size (int):
            minimum size of each batch
        return_idx (bool):
            whether to return the indices of the batches.
            output will be [start, end] idx
        length (int):
            length of the iterable.
            if None, then length is len(iterable)
            This is useful if you want to make batches of 
             something that doesn't have a __len__ method.
    
    Returns:
        output (iterable):
            batches of iterable
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

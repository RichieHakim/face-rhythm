import multiprocessing as mp
import threading
from typing import Union
import time
import gc
import json
from pathlib import Path
import copy
import re

import numpy as np
import decord
import torch
from tqdm import tqdm

import scipy
import scipy.sparse
import scipy.signal

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
        cv2.putText(frame, "WELCOME TO FACE RHYTHM!", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Prepping CV2", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Calling this figure allows cv2.imshow ", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "to work without crashing if this function", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "is called before importing av and decord", (10,190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('startup', frame)
        cv2.waitKey(1000)
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


#####################################################################################################################################
########################################################### FILE HELPERS ############################################################
#####################################################################################################################################

def prepare_filepath_for_saving(path, mkdir=False, allow_overwrite=True):
    """
    Checks if a file path is valid.
    RH 2022

    Args:
        path (str):
            Path to check.
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True) if mkdir else None
    assert allow_overwrite or not Path(path).exists(), f'{path} already exists.'
    assert Path(path).parent.exists(), f'{Path(path).parent} does not exist.'
    assert Path(path).parent.is_dir(), f'{Path(path).parent} is not a directory.'


def json_save(obj, path_save, indent=4, mode='w', mkdir=False, allow_overwrite=True):
    """
    Saves an object to a json file.
    Uses json.dump.
    RH 2022

    Args:
        obj (object):
            Object to save.
        path_save (str):
            Path to save object to.
        mode (str):
            Mode to open file in.
            Can be:
                'wb' (write binary)
                'ab' (append binary)
                'xb' (exclusive write binary. Raises FileExistsError if file already exists.)
        mkdir (bool):
            If True, creates parent directory if it does not exist.
        allow_overwrite (bool):
            If True, allows overwriting of existing file.        
    """
    prepare_filepath_for_saving(path_save, mkdir=mkdir, allow_overwrite=allow_overwrite)
    with open(path_save, mode) as f:
        json.dump(obj, f, indent=indent)

def json_load(filename, mode='r'):
    """
    Loads a json file.
    RH 2022

    Args:
        filename (str):
            Path to pickle file.
        mode (str):
            Mode to open file in.

    Returns:
        obj (object):
            Object loaded from pickle file.
    """
    with open(filename, mode) as f:
        return json.load(f)
        

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


#####################################################################################################################################
######################################################### CONTAINER HELPERS #########################################################
#####################################################################################################################################

def deep_update_dict(dictionary, key, new_val=None, new_key=None, in_place=False):
    """
    Updates a dictionary with a new value.
    RH 2022

    Args:
        dictionary (Dict):
            dictionary to update
        key (list of str):
            Key to update
            List elements should be strings.
            Each element should be a hierarchical
             level of the dictionary.
            DEMO:
                deep_update_dict(params, ['dataloader_kwargs', 'prefetch_factor'], val)
        new_val (any):
            If not None, the value to update with this
            If None, then new_key must be specified and will only
             be used to update the key.
        new_key (str):
            If not None, the key will be updated with this key.
             [key[-1]] will be deleted and replaced with new_key.
            If None, then [key[-1]] will be updated with new_val.
             
        in_place (bool):
            whether to update in place

    Returns:
        output (Dict):
            updated dictionary
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

def find_subDict_key(d: dict, s: str, max_depth: int=9999999):
    """
    Recursively search for a sub-dictionary that contains the given string.
    Yield the result.

    Args:
        d (dict):
            dictionary to search
        s (str):
            string of the key to search for using regex
        max_depth (int):
            maximum depth to search.
            1 means only search the keys in the top level of
             the dictionary. 2 means the first and second level.

    Returns:
        k_all (list of tuples):
            List of 2-tuples: (list of tuples containing:
             list of strings of keys to sub-dictionary, value
             of sub-dictionary)
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


#####################################################################################################################################
############################################################## VIDEO ################################################################
#####################################################################################################################################

class VideoReaderWrapper(decord.VideoReader):
    """
    Used to fix a memory leak bug in decord.VideoReader
    Taken from here.
    https://github.com/dmlc/decord/issues/208#issuecomment-1157632702
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seek(0)
        
        self.path = args[0]

    def __getitem__(self, key):
        frames = super().__getitem__(key)
        self.seek(0)
        return frames


class BufferedVideoReader:
    """
    A video reader that loads chunks of frames into a memory buffer
     in background processes so that sequential batches of frames
     can be accessed quickly.
    In many cases, allows for reading videos in batches without
     waiting for loading of the next batch to finish.
    Uses threading to read frames in the background.

    Optimal use case:
    1. Create a _BufferedVideoReader object
    EITHER 2A. Set method_getitem to 'continuous' and iterate over the
        object. This will read frames continuously in the
        background. This is the fastest way to read frames.
    OR 2B. Call batches of frames sequentially. Going backwards is
        slow. Buffers move forward.
    3. Each batch should be within a buffer. There should be no
        batch slices that overlap multiple buffers. Eg. if the
        buffer size is 1000 frames, then the following is fast:
        [0:1000], [1000:2000], [2000:3000], etc.
        But the following are slow:
        [0:1700],  [1700:3200],   [0:990],         [990:1010], etc.
        ^too big,  ^2x overlaps,  ^went backward,  ^1x overlap

    RH 2022
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
        decord_backend: str='torch',
        decord_ctx=None,
        verbose: int=1,
    ):
        """
        video_readers (list of decord.VideoReader): 
            list of decord.VideoReader objects.
            Can also be single decord.VideoReader object.
            If None, then paths_videos must be provided.
        paths_videos (list of str):
            list of paths to videos.
            Can also be single str.
            If None, then video_readers must be provided.
            If both paths_videos and video_readers are provided, 
             then video_readers will be used.
        buffer_size (int):
            Number of frames per buffer slot.
            When indexing this object, try to not index more than
             buffer_size frames at a time, and try to not index
             across buffer slots (eg. across idx%buffer_size==0).
             These require concatenating buffers, which is slow.
        prefetch (int):
            Number of buffers to prefetch.
            If 0, then no prefetching.
            Note that a single buffer slot can only contain frames
             from a single video. Best to keep 
             buffer_size <= video length.
        posthold (int):
            Number of buffers to hold after a new buffer is loaded.
            If 0, then no posthold.
            This is useful if you want to go backwards in the video.
        method_getitem (str):
            Method to use for __getitem__.
            'continuous' - read frames continuously across videos.
                Index behaves like videos are concatenated:
                - reader[idx] where idx: slice=idx_frames
            'by_video' - index must specify video index and frame 
                index:
                - reader[idx] where idx: tuple=(int: idx_video, slice: idx_frames)
        starting_seek_position (int):
            Starting frame index to start iterator from.
            Only used when method_getitem=='continuous' and
             using the iterator method.
        decord_backend (str):
            Backend to use for decord when loading frames.
            See decord documentation for options.
            ('torch', 'numpy', 'mxnet', ...)
        decord_ctx (decord.Context):
            Context to use for decord when loading frames.
            See decord documentation for options.
            (decord.cpu(), decord.gpu(), ...)
        verbose (int):
            Verbosity level.
            0: no output
            1: output warnings
            2: output warnings and info
        """
        import pandas as pd

        self._verbose = verbose
        self.buffer_size = buffer_size
        self.prefetch = prefetch
        self.posthold = posthold
        self._decord_backend = decord_backend
        self._decord_ctx = decord.cpu(0) if decord_ctx is None else decord_ctx

        ## Check inputs
        if isinstance(video_readers, decord.VideoReader):
            video_readers = [video_readers]
        if isinstance(paths_videos, str):
            paths_videos = [paths_videos]
        assert (video_readers is not None) or (paths_videos is not None), "Must provide either video_readers or paths_videos"

        ## If both video_readers and paths_videos are provided, use the video_readers and print a warning
        if (video_readers is not None) and (paths_videos is not None):
            print(f"FR WARNING: Both video_readers and paths_videos were provided. Using video_readers and ignoring path_videos.")
            paths_videos = None
        ## If paths are specified, import them as decord.VideoReader objects
        if paths_videos is not None:
            print(f"FR: Loading lazy video reader objects...") if self._verbose > 1 else None
            assert isinstance(paths_videos, list), "paths_videos must be list of str"
            assert all([isinstance(p, str) for p in paths_videos]), "paths_videos must be list of str"
            video_readers = [VideoReaderWrapper(path_video, ctx=self._decord_ctx) for path_video in tqdm(paths_videos, disable=(self._verbose < 2))]
            self.paths_videos = paths_videos
        else:
            print(f"FR: Using provided video reader objects...") if self._verbose > 1 else None
            assert isinstance(video_readers, list), "video_readers must be list of decord.VideoReader objects"
            self.paths_videos = [v.path for v in video_readers]
            assert all([isinstance(v, decord.VideoReader) for v in video_readers]), "video_readers must be list of decord.VideoReader objects"
        ## Assert that method_getitem is valid
        assert method_getitem in ['continuous', 'by_video'], "method_getitem must be 'continuous' or 'by_video'"
        ## Check if backend is valid by trying to set it here (only works fully when used in the _load_frames method)
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
        Get metadata about videos: lengths, fps, frame size, 
         num_channels, etc.

        Args:
            video_readers (list of decord.VideoReader):
                List of decord.VideoReader objects

        Returns:
            metadata (list of dict):
                Dictionary containing metadata for each video.
                Contains: 'num_frames', 'frame_rate',
                 'frame_height_width', 'num_channels'
            num_frames_total (int):
                Total number of frames across all videos.
            frame_rate (float):
                Frame rate of videos.
            frame_height_width (tuple of int):
                Height and width of frames.
            num_channels (int):
                Number of channels.
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
        Load slots in the background using threading.

        Args:
            idx_slots (list): 
                List of tuples containing the indices of the slots to load.
                Each tuple should be of the form (idx_video, idx_buffer).
            wait_for_load (bool or list):
                If True, wait for the slots to load before returning.
                If False, return immediately.
                If True wait for each slot to load before returning.
                If a list of bools, each bool corresponds to a slot in
                 idx_slots.
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
        Load a single slot.
        self.slots[idx_slot[0]][idx_slot[1]] will be populated
         with the loaded data.
        Allows for a blocking_thread argument to be passed in,
         which will force this new thread to wait until the
         blocking_thread is finished (join()) before loading.
        
        Args:
            idx_slot (tuple):
                Tuple containing the indices of the slot to load.
                Should be of the form (idx_video, idx_buffer).
            blocking_thread (threading.Thread):
                Thread to wait for before loading.
        """
        ## Set backend of decord to PyTorch
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
        Delete slots from memory.
        Sets self.slots[idx_slot[0]][idx_slot[1]] to None.

        Args:
            idx_slots (list):
                List of tuples containing the indices of the 
                 slots to delete.
                Each tuple should be of the form (idx_video, idx_buffer).
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
        """
        Delete all slots from memory.
        Uses the _delete_slots() method.
        """
        print(f"FR: Deleting all slots") if self._verbose > 1 else None
        self._delete_slots(self.loaded)

    def wait_for_loading(self):
        """
        Wait for all slots to finish loading.
        """
        print(f"FR: Waiting for all slots to load") if self._verbose > 1 else None
        while len(self.loading) > 0:
            time.sleep(0.01)
        

    
    def get_frames_from_single_video_index(self, idx: tuple):
        """
        Get a slice of frames by specifying the video number and 
         the frame number.

        Args:
            idx (tuple or int):
            A tuple containing the index of the video and a slice for the frames.
            (idx_video: int, idx_frames: slice)
            If idx is an int or slice, it is assumed to be the index of the video, and
             a new BufferedVideoReader(s) will be created with just those videos.

        Returns:
            frames (torch.Tensor):
                A tensor of shape (num_frames, height, width, num_channels)
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
                decord_backend='torch',
                decord_ctx=None,
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
        Get a batch of frames from a continuous index.
        Here the videos are treated as one long sequence of frames,
         and the index is the index of the frames in this sequence.

        Args:
            idx (int or slice):
                The index of the frames to get. If an int, a single frame is returned.
                If a slice, a batch of frames is returned.

        Returns:
            frames (torch.Tensor):
                A tensor of shape (num_frames, height, width, num_channels)
        """
        ## Assert that idx is an int or a slice
        assert isinstance(idx, int) or isinstance(idx, slice), f"idx must be an int or a slice. Got {type(idx)}"
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
        Set the starting frame for the iterator.
        Index should be in 'continuous' format.

        Args:
            idx (int):
                The index of the frame to start the iterator from.
                Should be in 'continuous' format where the index
                 is the index of the frame in the entire sequence 
                 of frames.
        """
        self._iterator_frame = idx
        
    def __getitem__(self, idx):
        if self.method_getitem == 'by_video':
            return self.get_frames_from_single_video_index(idx)
        elif self.method_getitem == 'continuous':
            return self.get_frames_from_continuous_index(idx)
        else:
            raise ValueError(f"Invalid method_getitem: {self.method_getitem}")

    def __len__(self): 
        if self.method_getitem == 'by_video':
            return len(self.video_readers)
        elif self.method_getitem == 'continuous':
            return self.num_frames_total
    def __repr__(self): 
        if self.method_getitem == 'by_video':
            return f"BufferedVideoReader(buffer_size={self.buffer_size}, num_videos={self.num_videos}, method_getitem='{self.method_getitem}', loaded={self.loaded}, prefetch={self.prefetch}, loading={self.loading}, verbose={self._verbose})"    
        elif self.method_getitem == 'continuous':
            return f"BufferedVideoReader(buffer_size={self.buffer_size}, num_videos={self.num_videos}, total_frames={self.num_frames_total}, method_getitem='{self.method_getitem}', iterator_frame={self._iterator_frame}, prefetch={self.prefetch}, loaded={self.loaded}, loading={self.loading}, verbose={self._verbose})"
    def __iter__(self): 
        """
        If method_getitem is 'by_video':
            Iterate over BufferedVideoReaders for each video.
        If method_getitem is 'continuous':
            Iterate over the frames in the video.
            Makes a generator that yields single frames directly from
            the buffer slots.
            If it is the initial frame, or the first frame of a slot,
            then self.get_frames_from_continuous_index is called to
            load the next slots into the buffer.
        """
        if self.method_getitem == 'by_video':
            return iter([BufferedVideoReader(
                video_readers=[self.video_readers[idx]],
                buffer_size=self.buffer_size,
                prefetch=self.prefetch,
                method_getitem='continuous',
                starting_seek_position=0,
                decord_backend='torch',
                decord_ctx=None,
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


########################################################################################################################################
############################################################# CONVOLUTION ##############################################################
########################################################################################################################################

class Toeplitz_convolution2d:
    """
    Convolve a 2D array with a 2D kernel using the Toeplitz matrix 
     multiplication method.
    Allows for SPARSE 'x' inputs. 'k' should remain dense.
    Ideal when 'x' is very sparse (density<0.01), 'x' is small
     (shape <(1000,1000)), 'k' is small (shape <(100,100)), and
     the batch size is large (e.g. 1000+).
    Generally faster than scipy.signal.convolve2d when convolving mutliple
     arrays with the same kernel. Maintains low memory footprint by
     storing the toeplitz matrix as a sparse matrix.

    See: https://stackoverflow.com/a/51865516 and https://github.com/alisaaalehi/convolution_as_multiplication
     for a nice illustration.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.convolution_matrix.html 
     for 1D version.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.matmul_toeplitz.html#scipy.linalg.matmul_toeplitz 
     for potential ways to make this implementation faster.

    Test with: tests.test_toeplitz_convolution2d()
    RH 2022
    """
    def __init__(
        self,
        x_shape,
        k,
        mode='same',
        dtype=None,
    ):
        """
        Initialize the convolution object.
        Makes the Toeplitz matrix and stores it.

        Args:
            x_shape (tuple):
                The shape of the 2D array to be convolved.
            k (np.ndarray):
                2D kernel to convolve with
            mode (str):
                'full', 'same' or 'valid'
                see scipy.signal.convolve2d for details
            dtype (np.dtype):
                The data type to use for the Toeplitz matrix.
                Ideally, this matches the data type of the input array.
                If None, then the data type of the kernel is used.
        """
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
        Convolve the input array with the kernel.

        Args:
            x (np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix):
                Input array(s) (i.e. image(s)) to convolve with the kernel
                If batching==False: Single 2D array to convolve with the kernel.
                    shape: (self.x_shape[0], self.x_shape[1])
                    type: np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
                If batching==True: Multiple 2D arrays that have been flattened
                 into row vectors (with order='C').
                    shape: (n_arrays, self.x_shape[0]*self.x_shape[1])
                    type: np.ndarray or scipy.sparse.csc_matrix or scipy.sparse.csr_matrix
            batching (bool):
                If False, x is a single 2D array.
                If True, x is a 2D array where each row is a flattened 2D array.
            mode (str):
                'full', 'same' or 'valid'
                see scipy.signal.convolve2d for details
                Overrides the mode set in __init__.

        Returns:
            out (np.ndarray or scipy.sparse.csr_matrix):
                If batching==True: Multiple convolved 2D arrays that have been flattened
                 into row vectors (with order='C').
                    shape: (n_arrays, height*width)
                    type: np.ndarray or scipy.sparse.csc_matrix
                If batching==False: Single convolved 2D array of shape (height, width)
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
            idx_crop = np.zeros((self.so), dtype=np.bool8)
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
        Roll columns of a sparse matrix.
        """
        out = x.copy()
        out.row += shift
        return out

def cosine_kernel_2D(center=(5,5), image_size=(11,11), width=5):
    """
    Generate a 2D cosine kernel
    RH 2021
    
    Args:
        center (tuple):  
            The mean position (X, Y) - where high value expected. 0-indexed. Make second value 0 to make 1D
        image_size (tuple): 
            The total image size (width, height). Make second value 0 to make 1D
        width (scalar): 
            The full width of one cycle of the cosine
    
    Return:
        k_cos (np.ndarray): 
            2D or 1D array of the cosine kernel
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
    Like np.logspace, but with a defined start and
     stop.
    RH 2022
    
    Args:
        start (float):
            First value in output array
        stop (float):
            Last value in output array
        num (int):
            Number of values in output array
            
    Returns:
        output (np.ndarray):
            Array of values
    """

    exp = 2  ## doesn't matter what this is, just needs to be > 1

    return exp ** np.linspace(np.log(start)/np.log(exp), np.log(stop)/np.log(exp), num, endpoint=True)

def gaussian(x=None, mu=0, sig=1, plot_pref=False):
    '''
    A gaussian function (normalized similarly to scipy's function)
    RH 2021
    
    Args:
        x (np.ndarray): 1-D array of the x-axis of the kernel
        mu (float): center position on x-axis
        sig (float): standard deviation (sigma) of gaussian
        plot_pref (boolean): True/False or 1/0. Whether you'd like the kernel plotted
        
    Returns:
        gaus (np.ndarray): gaussian function (normalized) of x
        params_gaus (dict): dictionary containing the input params
    '''
    import matplotlib.pyplot as plt

    if x is None:
        x = np.linspace(-sig*5, sig*5, sig*7, endpoint=True)

    gaus = 1/(np.sqrt(2*np.pi)*sig)*np.exp((-((x-mu)/sig) **2)/2)

    if plot_pref:
        plt.figure()
        plt.plot(x , gaus)
        plt.xlabel('x')
        plt.title(f'$\mu$={mu}, $\sigma$={sig}')

    return gaus


##########################################################################################################################################
########################################################### SPECTRAL ANALYSIS ############################################################
##########################################################################################################################################

def torch_hilbert(x, N=None, dim=0):
    """
    Computes the analytic signal using the Hilbert transform.
    Based on scipy.signal.hilbert
    RH 2022
    
    Args:
        x (nd tensor):
            Signal data. Must be real.
        N (int):
            Number of Fourier components to use.
            If None, then N = x.shape[dim]
        dim (int):
            Dimension along which to do the transformation.
    
    Returns:
        xa (nd tensor):
            Analytic signal of input x along dim
    """
    assert x.is_complex() == False, "x should be real"
    n = x.shape[dim] if N is None else N
    assert n >= 0, "N must be non-negative"

    xf = torch.fft.fft(input=x, n=n, dim=dim)
    m = torch.zeros(n, dtype=xf.dtype, device=xf.device)
    if n % 2: ## then even
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
    plot_pref=False
):
    """
    Creates a set of filters for use in the VQT algorithm.

    Set Q_lowF and Q_highF to be the same value for a 
     Constant Q Transform (CQT) filter set.
    Varying these values will varying the Q factor 
     logarithmically across the frequency range.

    RH 2022

    Args:
        Fs_sample (float):
            Sampling frequency of the signal.
        Q_lowF (float):
            Q factor to use for the lowest frequency.
        Q_highF (float):
            Q factor to use for the highest frequency.
        F_min (float):
            Lowest frequency to use.
        F_max (float):
            Highest frequency to use (inclusive).
        n_freq_bins (int):
            Number of frequency bins to use.
        win_size (int):
            Size of the window to use, in samples.
        plot_pref (bool):
            Whether to plot the filters.

    Returns:
        filters (Torch ndarray):
            Array of complex sinusoid filters.
            shape: (n_freq_bins, win_size)
        freqs (Torch array):
            Array of frequencies corresponding to the filters.
        wins (Torch ndarray):
            Array of window functions (gaussians)
             corresponding to each filter.
            shape: (n_freq_bins, win_size)
    """
    import matplotlib.pyplot as plt

    assert win_size%2==1, "RH Error: win_size should be an odd integer"
    
    freqs = bounded_logspace(
        start=F_min,
        stop=F_max,
        num=n_freq_bins,
    )

    periods = 1 / freqs
    periods_inSamples = Fs_sample * periods

    sigma_all = bounded_logspace(
        start=Q_lowF,
        stop=Q_highF,
        num=n_freq_bins,
    )
    sigma_all = sigma_all * periods_inSamples / 4

    wins = torch.stack([gaussian(torch.arange(-win_size//2, win_size//2), 0, sig=sigma) for sigma in sigma_all])

    filts = torch.stack([torch.cos(torch.linspace(-np.pi, np.pi, win_size) * freq * (win_size/Fs_sample)) * win for freq, win in zip(freqs, wins)], dim=0)    
    filts_complex = torch_hilbert(filts.T, dim=0).T
    
    if plot_pref:
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
        plt.imshow(filts / torch.max(filts, 1, keepdims=True)[0], aspect='auto', cmap='bwr', vmin=-1, vmax=1)
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
    def __init__(
        self,
        Fs_sample=1000,
        Q_lowF=3,
        Q_highF=20,
        F_min=10,
        F_max=400,
        n_freq_bins=55,
        win_size=501,
        downsample_factor=4,
        DEVICE_compute='cpu',
        DEVICE_return='cpu',
        batch_size=1000,
        return_complex=False,
        filters=None,
        plot_pref=False,
        progressBar=True,
    ):
        """
        Variable Q Transform.
        Class for applying the variable Q transform to signals.

        This function works differently than the VQT from 
         librosa or nnAudio. This one does not use iterative
         lowpass filtering. Instead, it uses a fixed set of 
         filters, and a Hilbert transform to compute the analytic
         signal. It can then take the envelope and downsample.
        
        Uses Pytorch for GPU acceleration, and allows gradients
         to pass through.

        Q: quality factor; roughly corresponds to the number 
         of cycles in a filter. Here, Q is the number of cycles
         within 4 sigma (95%) of a gaussian window.

        RH 2022

        Args:
            Fs_sample (float):
                Sampling frequency of the signal.
            Q_lowF (float):
                Q factor to use for the lowest frequency.
            Q_highF (float):
                Q factor to use for the highest frequency.
            F_min (float):
                Lowest frequency to use.
            F_max (float):
                Highest frequency to use.
            n_freq_bins (int):
                Number of frequency bins to use.
            win_size (int):
                Size of the window to use, in samples.
            downsample_factor (int):
                Factor to downsample the signal by.
                If the length of the input signal is not
                 divisible by downsample_factor, the signal
                 will be zero-padded at the end so that it is.
            DEVICE_compute (str):
                Device to use for computation.
            DEVICE_return (str):
                Device to use for returning the results.
            batch_size (int):
                Number of signals to process at once.
                Use a smaller batch size if you run out of memory.
            return_complex (bool):
                Whether to return the complex version of 
                 the transform. If False, then returns the
                 absolute value (envelope) of the transform.
                downsample_factor must be 1 if this is True.
            filters (Torch tensor):
                Filters to use. If None, will make new filters.
                Should be complex sinusoids.
                shape: (n_freq_bins, win_size)
            plot_pref (bool):
                Whether to plot the filters.
            progressBar (bool):
                Whether to show a progress bar.
        """
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
                plot_pref=plot_pref,
            )
        ## Gather parameters from arguments
        self.Fs_sample, self.Q_lowF, self.Q_highF, self.F_min, self.F_max, self.n_freq_bins, self.win_size, self.downsample_factor, self.DEVICE_compute, \
            self.DEVICE_return, self.batch_size, self.return_complex, self.plot_pref, self.progressBar = \
                Fs_sample, Q_lowF, Q_highF, F_min, F_max, n_freq_bins, win_size, downsample_factor, DEVICE_compute, DEVICE_return, batch_size, return_complex, plot_pref, progressBar

    def _helper_ds(self, X: torch.Tensor, ds_factor: int=4, return_complex: bool=False):
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
        out = torch.complex(
            torch.nn.functional.conv1d(arr.to(DEVICE)[:,None,:],  torch.real(filters.T).to(DEVICE).T[:,None,:], padding='same'),
            torch.nn.functional.conv1d(arr.to(DEVICE)[:,None,:], -torch.imag(filters.T).to(DEVICE).T[:,None,:], padding='same')
        )
        if take_abs:
            return torch.abs(out)
        else:
            return out

    def __call__(self, X):
        """
        Forward pass of VQT.

        Args:
            X (Torch tensor):
                Input signal.
                shape: (n_channels, n_samples)

        Returns:
            Spectrogram (Torch tensor):
                Spectrogram of the input signal.
                shape: (n_channels, n_samples_ds, n_freq_bins)
            x_axis (Torch tensor):
                New x-axis for the spectrogram in units of samples.
                Get units of time by dividing by self.Fs_sample.
            self.freqs (Torch tensor):
                Frequencies of the spectrogram.
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
            torch.arange(0, X.shape[-1])[None,:], 
            kernel_size=[int(self.downsample_factor)], 
            stride=self.downsample_factor, ceil_mode=True
            ).squeeze()
        
        return specs, x_axis, self.freqs

    def __repr__(self):
        if self.using_custom_filters:
            return f"VQT with custom filters"
        else:
            return f"VQT object with parameters: Fs_sample={self.Fs_sample}, Q_lowF={self.Q_lowF}, Q_highF={self.Q_highF}, F_min={self.F_min}, F_max={self.F_max}, n_freq_bins={self.n_freq_bins}, win_size={self.win_size}, downsample_factor={self.downsample_factor}, DEVICE_compute={self.DEVICE_compute}, DEVICE_return={self.DEVICE_return}, batch_size={self.batch_size}, return_complex={self.return_complex}, plot_pref={self.plot_pref}"
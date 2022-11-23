import multiprocessing as mp
import threading
from typing import Union
import time

import numpy as np
import decord
import torch
from tqdm import tqdm

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


###########################
######### VIDEO ###########
###########################

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
    2. Call batches of frames sequentially. Going backwards is
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
        starting_video_index: int=None,
        starting_frame_index: int=None,
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
        starting_video_index (int):
            Index of video to load in initialization.
            If None, no video is loaded into buffer.
        starting_frame_index (int):
            Index of frame to load in initialization.
            If None, no frames are loaded into buffer.
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
        self.starting_video_index = starting_video_index
        self.starting_frame_index = starting_frame_index

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
            video_readers = [VideoReaderWrapper(path_video, ctx=decord.cpu(0), num_threads=mp.cpu_count()) for path_video in tqdm(paths_videos, disable=(self._verbose < 2))]
            self.paths_videos = paths_videos
        else:
            print(f"FR: Using provided video reader objects...") if self._verbose > 1 else None
            assert isinstance(video_readers, list), "video_readers must be list of decord.VideoReader objects"
            assert all([isinstance(v, decord.VideoReader) for v in video_readers]), "video_readers must be list of decord.VideoReader objects"


        self.video_readers = video_readers

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
        self.lookup = pd.DataFrame(self.lookup)

        ## Make a list for which slots are loaded or loading
        self.loading = []
        self.loaded = []

        if self.starting_video_index is not None:
            ## Load the starting video and frame
            idx_video = 0
            idx_frame_start = 0
            idx_frame_end = self.buffer_size * (self.prefetch + 1)
            idx_slots = [(idx_video, i) for i in range(idx_frame_start // self.buffer_size, ((idx_frame_end-1) // self.buffer_size)+1)]

            ## Load first prefetch slots
            self._load_slots(idx_slots, wait_for_load=[[True]+[False]*(self.prefetch-1)])
        else:
            print(f"FR: No starting video index provided. No video loaded into buffer.") if (self._verbose > 1) else None

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

        print(f"FR: Loading slots {idx_slots} in the background.") if self._verbose > 1 else None
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
        decord.bridge.set_bridge('torch')
        ## Wait for the previous slot to finish loading
        if blocking_thread is not None:
            blocking_thread.join()
        ## Load the slot
        idx_video, idx_buffer = idx_slot
        idx_frame_start, idx_frame_end = self.boundaries[idx_video][idx_buffer]
        self.slots[idx_video][idx_buffer] = self.video_readers[idx_video][idx_frame_start:idx_frame_end+1]
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
        for idx_slot in idx_slots:
            ## If the slot is loaded
            if idx_slot in self.loaded:
                print(f"FR: Deleting slot {idx_slot}") if self._verbose > 1 else None
                ## Delete the slot
                self.slots[idx_slot[0]][idx_slot[1]] = None
                ## Remove the slot from the loaded list
                self.loaded.remove(idx_slot)

    
    def __getitem__(self, idx: tuple):
        """
        Get a slice of frames by specifying the video number and 
         the frame number.

        Args:
            idx (tuple):
            A tuple containing the index of the video and a slice for the frames.
            (idx_video: int, idx_frames: slice)
            If there is only one video, idx_video can be omitted and
             idx can simply be a slice.

        Returns:
            frames (torch.Tensor):
                A tensor of shape (num_frames, height, width, num_channels)
        """
        print(f"FR: Getting item {idx}") if self._verbose > 1 else None
        ## If there is only one video, the video number is not needed
        idx = (0, idx) if (len(self.video_readers) == 1) else idx
        ## Get the index of the video and the slice of frames
        idx_video, idx_frames = idx
        ## If idx_frames is a single integer, convert it to a slice
        idx_frames = slice(idx_frames, idx_frames+1) if isinstance(idx_frames, int) else idx_frames
        ## Bound the range of the slice
        idx_frames = slice(max(idx_frames.start, 0), min(idx_frames.stop, len(self.video_readers[idx_video])))
        ## Assert that slice is not empty
        assert idx_frames.start < idx_frames.stop, "Slice is empty."

        ## Get the start and end indices for the slice of frames
        idx_frame_start = idx_frames.start if idx_frames.start is not None else 0
        idx_frame_end = idx_frames.stop if idx_frames.stop is not None else len(self.video_readers[idx_video])
        idx_frame_step = idx_frames.step if idx_frames.step is not None else 1

        ## Get the indices of the slots that contain the frames
        idx_slots = [(idx_video, i) for i in range(idx_frame_start // self.buffer_size, ((idx_frame_end-1) // self.buffer_size)+1)]
        print(f"FR: Slots to load: {idx_slots}") if self._verbose > 1 else None

        ## Load the prefetch slots
        if self.prefetch > 0:
            idx_slot_lookuptable = np.where((self.lookup['video']==idx_slots[-1][0]) * (self.lookup['slot']==idx_slots[-1][1]))[0][0]
            idx_slots_prefetch = [(self.lookup['video'][ii], self.lookup['slot'][ii]) for ii in range(idx_slot_lookuptable+1, idx_slot_lookuptable+self.prefetch+1) if ii < len(self.lookup)]
        else:
            idx_slots_prefetch = []
        # idx_slot_next = (idx_slots[-1][0], idx_slots[-1][1]+1) if (idx_slots[-1][1] < len(self.slots[idx_slots[-1][0]])-1) else (idx_slots[-1][0]+1, 0)
        ## Load the slots
        self._load_slots(idx_slots + idx_slots_prefetch, wait_for_load=[True]*len(idx_slots) + [False])
        ## Delete the slots that are no longer needed. 
        ### All slots from old videos should be deleted.
        self._delete_slots([idx_slot for idx_slot in self.loaded if idx_slot[0] < idx_video])
        ### All slots from previous buffers should be deleted.
        self._delete_slots([idx_slot for idx_slot in self.loaded if idx_slot[0] == idx_video and idx_slot[1] < idx_frame_start // self.buffer_size])

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

    def __len__(self): return len(self.video_readers)
    def __repr__(self): return f"BufferedVideoReader(buffer_size={self.buffer_size}, loaded={self.loaded}, loading={self.loading}, verbose={self._verbose})"
    def __iter__(self): return iter([_BufferedVideoReader_singleVideo(self, idx_video) for idx_video in range(len(self.video_readers))])
    
class _BufferedVideoReader_singleVideo:
    def __init__(
        self,
        bvr: BufferedVideoReader,
        idx_video: int,
    ):
        self.bvr = bvr
        self.idx_video = idx_video

    def __getitem__(self, idx: slice):
        return self.bvr[self.idx_video, idx]

    def __len__(self): return len(self.bvr.video_readers[self.idx_video])
    def __iter__(self): return iter([self.bvr[self.idx_video, idx] for idx in range(len(self))])
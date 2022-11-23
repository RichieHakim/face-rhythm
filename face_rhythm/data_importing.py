from pathlib import Path
from typing import Union
from typing import List
import multiprocessing as mp
import threading
import time

import numpy as np
from tqdm import tqdm
import decord
import torch

from .util import FR_Module

## Define Dataset class as a subclass of utils.FR_Module
class Dataset_videos(FR_Module):
    def __init__(
        self,
        paths_videos: Union[str, List[str]]=None,
        contiguous: bool=False,
        frame_rate_clamp: float=None,
        verbose: Union[bool, int]=1,
    ):
        ## Imports
        super().__init__()

        ## Set variables
        self.contiguous = bool(contiguous)
        self.verbose = int(verbose)


        ## Assert that paths_videos is either a list of strings or a string
        assert paths_videos is not None, "FR ERROR: paths_videos must be specified as a list of strings to paths of videos"
        if isinstance(paths_videos, list): 
            assert all([isinstance(path, str) for path in paths_videos]), "FR ERROR: paths_videos must be a list of strings to paths of videos"  
        else:
            assert isinstance(paths_videos, str), "FR ERROR: paths_videos must be a list of strings to paths of videos"
        
        ## If paths_videos is a string, convert it to a list of strings
        self.paths_videos = paths_videos if isinstance(paths_videos, list) else [paths_videos]

        ## Assert that all paths_videos exist
        exists_paths_videos = [bool(Path(path).exists()) for path in self.paths_videos]
        assert all(exists_paths_videos), f"FR ERROR: paths_videos must exist. The following paths do not exist: {[path for path, exists in zip(self.paths_videos, exists_paths_videos) if not exists]}"


        ## Load videos
        print("FR: Loading lazy video reader objects...") if self.verbose > 1 else None
        self.videos = [_VideoReaderWrapper(path_video, ctx=decord.cpu(0), num_threads=mp.cpu_count()) for path_video in tqdm(self.paths_videos, disable=(self.verbose < 2))]


        ## make video metadata dataframe
        print("FR: Collecting video metadata...") if self.verbose > 1 else None
        self.metadata = {"paths_videos": self.paths_videos}
        self.num_frames, self.frame_rate, self.frame_height_width, self.num_channels = [], [], [], []
        for v in tqdm(self.videos):
            self.num_frames.append(int(len(v)))
            self.frame_rate.append(float(v.get_avg_fps()))
            frame_tmp = v[0].asnumpy()
            self.frame_height_width.append([int(n) for n in frame_tmp.shape[:2]])
            self.num_channels.append(int(frame_tmp.shape[2]))
        self.metadata["num_frames"] = self.num_frames
        self.metadata["frame_rate"] = self.frame_rate
        self.metadata["frame_height_width"] = self.frame_height_width
        self.metadata["num_channels"] = self.num_channels
            

        ## Assert that all videos must have at least one frame
        assert all([n > 0 for n in self.metadata["num_frames"]]), "FR ERROR: All videos must have at least one frame"
        ## Assert that all videos must have the same shape
        assert all([n == self.metadata["frame_height_width"][0] for n in self.metadata["frame_height_width"]]), "FR ERROR: All videos must have the same shape"
        ## Assert that all videos must have the same number of channels
        assert all([n == self.metadata["num_channels"][0] for n in self.metadata["num_channels"]]), "FR ERROR: All videos must have the same number of channels"

        ## set frame rate
        if frame_rate_clamp is None:
            frame_rates = self.metadata["frame_rate"]
            ## warn if any video's frame rate is very different from others
            max_diff = float((np.max(frame_rates) - np.min(frame_rates)) / np.mean(frame_rates))
            print(f"FR WARNING: max frame rate difference is large: {max_diff*100:.2f}%") if ((max_diff > 0.1) and (self.verbose > 0)) else None
            self.frame_rate = float(np.median(frame_rates))
        else:
            self.frame_rate = float(frame_rate_clamp)

        self.num_frames_total = int(np.sum(self.metadata["num_frames"]))
        self.frame_height_width = self.metadata["frame_height_width"][0]
        self.num_channels = self.metadata["num_channels"][0]

        ## For FR_Module compatibility
        self.config = {
            "paths_videos": paths_videos,
            "contiguous": contiguous,
            "frame_rate_clamp": frame_rate_clamp,
            "verbose": verbose,
        }
        self.run_info = {
            "frame_rate": self.frame_rate,
            "num_frames_total": self.num_frames_total,
            "frame_height_width": self.frame_height_width,
            "num_channels": self.num_channels,
        }
        self.run_data = {
            "metadata": self.metadata,  ## this should be a lazy reference to the self.metadata 
        }
        ## Append the self.run_info data to self.run_data
        self.run_data.update(self.run_info)

    def __repr__(self):
        return f"Dataset_videos, frame_rate={self.frame_rate}"

    ## Define methods for loading and handling videos
    def __getitem__(self, index): return self.videos[index]
    def __len__(self): return len(self.videos)
    def __iter__(self): return iter(self.videos)
    def __next__(self): return next(self.videos)


class _VideoReaderWrapper(decord.VideoReader):
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


class _ContinuousBufferedVideoReader:
    """
    Allows for reading videos continuously without breaks between 
     videos or batches by buffering future frames in the background.
    Uses threading to read frames in the background.
    """
    def __init__(
        self,
        video_readers: list,
        paths_videos: list=None,
        buffer_size: int=1000,
        starting_video_index: int=0,
        starting_frame_index: int=0,
        verbose: int=1,
    ):
        """
        video_readers (list of decord.VideoReader): 
            list of decord.VideoReader objects
        """
        self.video_readers = video_readers
        self.buffer_size = buffer_size
        self.starting_video_index = starting_video_index
        self.starting_frame_index = starting_frame_index
        self._verbose = verbose

        ## Initialize the buffer
        ### Make a list containing a slot for each buffer chunk
        self.slots = [[None] * np.ceil(len(d)/self.buffer_size).astype(int) for d in self.video_readers]
        ### Make a list containing the bounding indices for each buffer video chunk. Upper bound should be min(buffer_size, num_frames)
        self.boundaries = [[(i*self.buffer_size, min((i+1)*self.buffer_size, len(d))-1) for i in range(len(s))] for d, s in zip(self.video_readers, self.slots)]

        ## Find first two slots to load
        idx_slot = (self.starting_video_index, self.starting_frame_index // self.buffer_size)  ## (idx_video, idx_buffer)
        idx_slot_next = (idx_slot[0], idx_slot[1]+1) if (idx_slot[1] < len(self.slots[idx_slot[0]])-1) else (idx_slot[0]+1, 0)

        ## Initialize the threads
        self.threads = []

        ## Make a list for which slots are loaded or loading
        self.loading = []
        self.loaded = []


        ## Load first two slots
        self._load_slots([idx_slot, idx_slot_next], wait_for_load=[True, False])

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
                self.threads.append(thread)

                ## Wait for the slot to load if wait_for_load is True
                if wait:
                    print(f"FR: Waiting for slot {idx_slot} to load") if self._verbose > 1 else None
                    thread.join()
                    print(f"FR: Slot {idx_slot} loaded") if self._verbose > 1 else None


    def _load_slot(self, idx_slot: tuple, blocking_thread: threading.Thread=None):
        """
        Load a single slot.
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

    
    ## Define __getitem__ method for getting slices of the video
    def __getitem__(self, idx: tuple):
        """
        Get a slice of frames from the video.

        Args:
            idx (tuple):
            A tuple containing the index of the video and a slice for the frames.
            (idx_video: int, idx_frames: slice)
        """
        ## Get the index of the video and the slice of frames
        idx_video, idx_frames = idx
        ## Bound the range of the slice
        idx_frames = slice(max(idx_frames.start, 0), min(idx_frames.stop, len(self.video_readers[idx_video])))

        ## Get the start and end indices for the slice of frames
        idx_frame_start = idx_frames.start if idx_frames.start is not None else 0
        idx_frame_end = idx_frames.stop if idx_frames.stop is not None else len(self.video_readers[idx_video])
        idx_frame_step = idx_frames.step if idx_frames.step is not None else 1

        ## Get the indices of the slots that contain the frames
        idx_slots = [(idx_video, i) for i in range(idx_frame_start // self.buffer_size, idx_frame_end // self.buffer_size + 1)]
        print(f"FR: Slots to load: {idx_slots}") if self._verbose > 1 else None
        ## Get the subsequent slot
        idx_slot_next = (idx_slots[-1][0], idx_slots[-1][1]+1) if (idx_slots[-1][1] < len(self.slots[idx_slots[-1][0]])-1) else (idx_slots[-1][0]+1, 0)
        ## Load the slots
        self._load_slots(idx_slots + [idx_slot_next], wait_for_load=[True]*len(idx_slots) + [False])
        ## Delete the slots that are no longer needed. 
        ### All slots from old videos should be deleted.
        self._delete_slots([idx_slot for idx_slot in self.loaded if idx_slot[0] < idx_video])
        ### All slots from previous buffers should be deleted.
        self._delete_slots([idx_slot for idx_slot in self.loaded if idx_slot[0] == idx_video and idx_slot[1] < idx_frame_start // self.buffer_size])

        ## Get the indices of the frames within the slots. Define them as slices.
        idx_frames_slots = [slice(max(0, idx_frame_start - i*self.buffer_size), min(self.buffer_size, idx_frame_end - i*self.buffer_size), idx_frame_step) for i in range(idx_frame_start // self.buffer_size, idx_frame_end // self.buffer_size + 1)]

        ## Get the frames. Then concatenate them along the first dimension using torch.cat
        ### Skip the concatenation if there is only one slot
        if len(idx_slots) == 1:
            frames = self.slots[idx_slots[0][0]][idx_slots[0][1]][idx_frames_slots[0]]
        else:
            print(f"FR: Warning. Slicing across multiple slots is SLOW. Consider increasing buffer size or adjusting batching method.") if self._verbose > 1 else None
            frames = torch.cat([self.slots[idx_slot[0]][idx_slot[1]][idx_frames_slot] for idx_slot, idx_frames_slot in zip(idx_slots, idx_frames_slots)], dim=0)
        
        return frames

from pathlib import Path
from typing import Union
from typing import List
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
import decord

from .util import FR_Module
from .helpers import VideoReaderWrapper, BufferedVideoReader

## Define Dataset class as a subclass of utils.FR_Module
class Dataset_videos(FR_Module):
    """
    Class for handling video data.
    Imports videos using decord and generates lazy video reader
     objects for each video.
    RH 2022
    """
    def __init__(
        self,
        bufferedVideoReader: BufferedVideoReader=None,
        paths_videos: Union[str, List[str]]=None,
        contiguous: bool=False,
        frame_rate_clamp: float=None,
        verbose: Union[bool, int]=1,
    ):
        """
        Initialize the Dataset_videos object.

        Args:
            bufferedVideoReader (BufferedVideoReader, optional):
                Optional. If None, then paths_videos must be specified.
                BufferedVideoReader object to use for reading videos.
                Object should have all the desired videos loaded.
            paths_videos (str or list of str):
                Optional. If None, then bufferedVideoReader must be
                 specified.
                Path(s) to video(s) to load.
            contiguous (bool):
                Whether the videos should be treated as contiguous.
                I.e. Whether the first frame of the next video should be
                 considered to be the subsequent frame of the last video.
            frame_rate_clamp (float):
                If None then the frame rate metadata from the videos will
                 be used.
                If a float then the frame rate will be set to this value.
            verbose (bool or int):
                Whether to print progress messages.
                0: No messages
                1: Warnings
                2: Info
        """
        ## Imports
        super().__init__()

        ## Set variables
        self.contiguous = bool(contiguous)
        self.verbose = int(verbose)

        ## Determine whether to use bufferedVideoReader or paths_videos
        self._videoDataType = 'BufferedVideoReader' if bufferedVideoReader is not None else 'paths_videos'
        
        ### Assert that either bufferedVideoReader or paths_videos is specified
        assert bufferedVideoReader is not None or paths_videos is not None, "FR ERROR: bufferedVideoReader or paths_videos must be specified"
        
        ## Workflow if method is 'paths_videos'
        ### Assert that if using 'paths_videos', it is either a list of strings or a string
        if self._videoDataType == 'paths_videos':
            if isinstance(paths_videos, list): 
                assert all([isinstance(path, str) for path in paths_videos]), "FR ERROR: paths_videos must be a string or list of strings to paths of videos"  
            else:
                assert isinstance(paths_videos, str), "FR ERROR: paths_videos must be a string or list of strings to paths of videos"
            ## If paths_videos is a string, convert it to a list of strings
            self.paths_videos = paths_videos if isinstance(paths_videos, list) else [paths_videos]

            ## Assert that all paths_videos exist
            exists_paths_videos = [bool(Path(path).exists()) for path in self.paths_videos]
            assert all(exists_paths_videos), f"FR ERROR: paths_videos must exist. The following paths do not exist: {[path for path, exists in zip(self.paths_videos, exists_paths_videos) if not exists]}"


            ## Load videos
            print("FR: Loading lazy video reader objects...") if self.verbose > 1 else None
            decord.bridge.set_bridge('torch')
            self.videos = [VideoReaderWrapper(path_video, ctx=decord.cpu(0), num_threads=mp.cpu_count()) for path_video in tqdm(self.paths_videos, disable=(self.verbose < 2))]

            ## make video metadata dataframe
            print("FR: Collecting video metadata...") if self.verbose > 1 else None
            self.metadata = {"paths_videos": self.paths_videos}
            self.num_frames, self.frame_rate, self.frame_height_width, self.num_channels = [], [], [], []
            for v in tqdm(self.videos):
                self.num_frames.append(int(len(v)))
                self.frame_rate.append(float(v.get_avg_fps()))
                frame_tmp = v[0]
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

        ## Workflow if method is 'BufferedVideoReader'
        elif self._videoDataType == 'BufferedVideoReader':
            ## Assert that bufferedVideoReader is a BufferedVideoReader object
            print('printing this line helps the bufferedVideoReader object load properly.  ', type(bufferedVideoReader), type(BufferedVideoReader), BufferedVideoReader.__class__, isinstance(bufferedVideoReader, BufferedVideoReader)) if self.verbose > 1 else None   ## line needed sometimes for next assert to work
            assert isinstance(bufferedVideoReader, BufferedVideoReader), "FR ERROR: bufferedVideoReader must be a BufferedVideoReader object"
            ## Set self.videos to bufferedVideoReader
            self.videos = bufferedVideoReader.video_readers
            ## Set self.paths_videos to bufferedVideoReader.paths_videos
            self.paths_videos = bufferedVideoReader.paths_videos
            ## Set self.metadata to bufferedVideoReader.metadata
            self.metadata = bufferedVideoReader.metadata

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
        self.paths_videos = [str(path) for path in self.paths_videos]  ## ensure paths are strings

        self.example_image = self.videos[0][0]

        ## For FR_Module compatibility
        self.config = {
            "paths_videos": self.paths_videos,
            "contiguous": contiguous,
            "frame_rate_clamp": frame_rate_clamp,
            "verbose": verbose,
        }
        self.run_info = {
            "frame_rate": self.frame_rate,
            "num_frames_total": self.num_frames_total,
            "frame_height_width": self.frame_height_width,
            "num_channels": self.num_channels,
            "metadata": self.metadata,
        }
        self.run_data = {
            "example_image": self.example_image,
        }
        ## Append the self.run_info data to self.run_data
        # self.run_data.update(self.run_info)

    def __repr__(self):
        return f"Dataset_videos, num_videos={len(self.paths_videos)}, num_frames_total={self.num_frames_total}, frame_rate={self.frame_rate}, frame_height_width={self.frame_height_width}, num_channels={self.num_channels}"

    ## Define methods for loading and handling videos
    def __getitem__(self, index): return self.videos[index]
    def __len__(self): return len(self.videos)
    def __iter__(self): return iter(self.videos)
    def __next__(self): return next(self.videos)

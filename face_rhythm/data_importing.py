from pathlib import Path
from typing import Union
from typing import List

import numpy as np
from tqdm import tqdm

from .utils import FR_Module

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
        import decord

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
        exists_paths_videos = [Path(path).exists() for path in self.paths_videos]
        assert all(exists_paths_videos), f"FR ERROR: paths_videos must exist. The following paths do not exist: {[path for path, exists in zip(self.paths_videos, exists_paths_videos) if not exists]}"

        print("FR: Loading lazy video reader objects...") if self.verbose > 1 else None
        self.videos = [decord.VideoReader(path_video, ctx=decord.cpu(0)) for path_video in tqdm(paths_videos, disable=(self.verbose < 2))]

        ## get video metadata
        self.metadata = [{
            "path_video": str(path_video),
            "num_frames": int(len(vr)),
            "frame_rate": float(vr.get_avg_fps()),
        } for path_video, vr in zip(paths_videos, self.videos)]

        ## set frame rate
        if frame_rate_clamp is None:
            frame_rates = [m["frame_rate"] for m in self.metadata]
            ## warn if any video's frame rate is very different from others
            max_diff = (np.max(frame_rates) - np.min(frame_rates)) / np.mean(frame_rates)
            print(f"FR WARNING: max frame rate difference is large: {max_diff*100:.2f}%") if ((max_diff > 0.1) and (self.verbose > 0)) else None
            self.frame_rate = float(np.median(frame_rates))
        else:
            self.frame_rate = float(frame_rate_clamp)

        ## For FR_Module compatibility
        self.run_info = {
            "paths_videos": self.paths_videos,
            "contiguous": contiguous,
            "frame_rate": self.frame_rate,
        }
        self.run_data = {
            "metadata": self.metadata,  ## this should be a lazy reference to self.metadata
        }

    def __repr__(self):
        return f"Dataset_videos, frame_rate={self.frame_rate}"

    ## Define methods for loading and handling videos
    def __getitem__(self, index): return self.videos[index]
    def __len__(self): return len(self.videos)
    def __iter__(self): return iter(self.videos)
    def __next__(self): return next(self.videos)
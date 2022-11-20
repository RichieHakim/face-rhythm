import numpy as np

from .utils import FR_Module

## Define Dataset class as a subclass of utils.FR_Module
class Dataset_videos(FR_Module):
    def __init__(
        self,
        paths_videos=None,
        contiguous=False,
        frame_rate_clamp=None,
        verbose=1,
    ):
        super().__init__()
        import decord
        self.paths_videos = paths_videos
        self.contiguous = contiguous
        self.verbose = verbose

        print("FR: Loading lazy video reader objects...") if self.verbose > 1 else None
        self.videos = [decord.VideoReader(path_video, ctx=decord.cpu(0)) for path_video in paths_videos]

        ## get video metadata
        self.metadata = [{
            "path_video": path_video,
            "num_frames": len(vr),
            "frame_rate": vr.get_avg_fps(),
        } for path_video, vr in zip(paths_videos, self.videos)]

        ## set frame rate
        if frame_rate_clamp is None:
            frame_rates = [m["frame_rate"] for m in self.metadata]
            ## warn if any video's frame rate is very different from others
            max_diff = (np.max(frame_rates) - np.min(frame_rates)) / np.mean(frame_rates)
            print(f"FR WARNING: max frame rate difference is large: {max_diff*100:.2f}%") if ((max_diff > 0.1) and (self.verbose > 0)) else None
            self.frame_rate = np.median(frame_rates)
        else:
            self.frame_rate = frame_rate_clamp

        ## For FR_Module compatibility
        self.runInfo = {
            "paths_videos": paths_videos,
            "contiguous": contiguous,
            "frame_rate": self.frame_rate,
        }
        self.runData = {
            "metadata": self.metadata,
        }

    def __repr__(self):
        return f"Dataset_videos, frame_rate={self.frame_rate}"

    ## Define methods for loading and handling videos
    def __getitem__(self, index): return self.videos[index]
    def __len__(self): return len(self.videos)
    def __iter__(self): return iter(self.videos)
    def __next__(self): return next(self.videos)
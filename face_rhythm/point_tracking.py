from typing import Union
import time

import numpy as np
from tqdm.notebook import tqdm
import decord
import cv2
import torch

from .util import FR_Module
from .data_importing import Dataset_videos
from .rois import ROIs
from .helpers import make_batches

## Define class for performing point tracking using optical flow
class PointTracker(FR_Module):
    def __init__(
        self,
        dataset_videos: Dataset_videos,
        rois_points: ROIs,
        rois_masks: ROIs=None,
        contiguous: bool=False,
        batch_size: int=1000,
        params_optical_flow: dict=None,
        verbose: Union[bool, int]=1,
    ):
        """
        Prepare for point tracking.
        Sets up parameters for optical flow, makes initial points,
         to track, and prepares masks for the video.

        Args:
            dataset_videos (Dataset_videos):
                A Dataset_videos object, containing the videos to track.
                Object comes from data_importing.Dataset_videos.
            rois_points (2D array of booleans or list of 2D arrays of booleans):
                An ROI is a 2D array of booleans, where True indicates a pixel
                 that is within the region of interest.
                If a list of ROIs is provided, then each ROI will be tracked
                 separately.
            rois_masks (2D array of booleans or list of 2D arrays of booleans):
                An ROI is a 2D array of booleans, where True indicates a pixel
                 that is within the region of interest.
                If a list of ROIs is provided, then each ROI will be tracked
                 separately.
            contiguous (bool, optional):
                Whether or not videos should be treated as contiguous.
                If True, the first frame of each video will be treated
                 as the next frame of the previous video.
                If False, the point tracking will be restarted for each
                 video.
            params_optical_flow (dict, optional):
                Parameters for optical flow.
                If None, the following parameters will be used:
                    params_optical_flow = {
                        "method": "lucas_kanade",
                        "point_spacing": 10,
                        "relaxation": 0.5,
                        "kwargs_method": {
                            "winSize": (15,15),
                            "maxLevel": 2,
                            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                        },
                    }
                See https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
                 and https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
                 for more information about the lucas kanade optical flow parameters.
        """
        ## Imports
        super().__init__()
        
        ## Set decord bridge (backend for video reading)
        decord.bridge.set_bridge('torch')

        ## Set variables
        self._contiguous = bool(contiguous)
        self._verbose = int(verbose)
        self._batch_size = int(batch_size)

        ## Assert that dataset_videos is a Dataset_videos object
        assert isinstance(dataset_videos, Dataset_videos), "FR ERROR: dataset_videos must be a Dataset_videos object"
        ## Assert that the rois variables are either 2D arrays or lists of 2D arrays
        if isinstance(rois_points, np.ndarray):
            rois_points = [rois_points]
        if isinstance(rois_masks, np.ndarray):
            rois_masks = [rois_masks]
        assert isinstance(rois_points, list) and all([isinstance(roi, np.ndarray) for roi in rois_points]), "FR ERROR: rois_points must be a 2D array of booleans or a list of 2D arrays of booleans"
        assert all([roi.ndim == 2 for roi in rois_points]), "FR ERROR: rois_points must be a 2D array of booleans or a list of 2D arrays of booleans"
        assert all([roi.dtype == bool for roi in rois_points]), "FR ERROR: rois_points must be a 2D array of booleans or a list of 2D arrays of booleans"
        

        ## Set parameters for optical flow
        params_default = {
                "method": "lucas_kanade",
                "point_spacing": 10,
                "relaxation": 0.5,
                "kwargs_method": {
                    "winSize": (15,15),
                    "maxLevel": 2,
                    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                },
            }
        if params_optical_flow is None:
            self.params_optical_flow = params_default
            print(f"FR: 'params_optical_flow' not provided. Using default parameters for optical flow: {self.params_optical_flow}") if self._verbose > 1 else None
        else:
            ## fill in missing parameters
            for key, value in params_default.items():
                if key not in params_optical_flow:
                    print(f"FR: 'params_optical_flow' does not contain key '{key}', using default value: {value}") if self._verbose > 0 else None
                    params_optical_flow[key] = value
            self.params_optical_flow = params_optical_flow

        ## Make points within rois_points with spacing of point_spacing
        ##  First make a single ROI boolean image, then make points
        rois_all = np.stack(rois_points, axis=0).all(axis=0)
        self.point_positions = self._make_points(rois_all, self.params_optical_flow["point_spacing"])

        ## Collapse masks into single mask
        if rois_masks is None:
            self.mask = torch.ones(dataset_videos[0][0].shape[:2], dtype=bool)
        else:
            self.mask = torch.stack(rois_masks, dim=0).all(dim=0)

        ## Store dataset_videos
        self.dataset_videos = dataset_videos


    def _make_points(self, roi, point_spacing):
        """
        Make points within a roi with spacing of point_spacing.
        
        Args:
            roi (np.ndarray, boolean):
                A 2D array of booleans, where True indicates a pixel
                 that is within the region of interest.
            point_spacing (int):
                The spacing between points, in pixels.

        Returns:
            points (np.ndarray, np.float32):
                A 2D array of integers, where each row is a point
                 to track.
        """
        ## Assert that roi is a 2D array of booleans
        assert isinstance(roi, np.ndarray), "FR ERROR: roi must be a numpy array"
        assert roi.ndim == 2, "FR ERROR: roi must be a 2D array"
        assert roi.dtype == bool, "FR ERROR: roi must be a 2D array of booleans"
        ## Warn if point_spacing is not an integer. It will be rounded.
        if not isinstance(point_spacing, int):
            print("FR WARNING: point_spacing must be an integer. It will be rounded.")
            point_spacing = int(round(point_spacing))

        ## make point cloud
        y, x = np.where(roi)
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x.min(), x.max()
        y_points = np.arange(y_min, y_max, point_spacing)
        x_points = np.arange(x_min, x_max, point_spacing)
        y_points, x_points = np.meshgrid(y_points, x_points)
        y_points = y_points.flatten()
        x_points = x_points.flatten()

        ## remove points outside of roi
        points = np.stack([y_points, x_points], axis=1)
        points = points[roi[points[:, 0], points[:, 1]]].astype(np.float32)

        return points

    def track_points(self):
        """
        Point tracking workflow.
        Tracks points in videos using specified parameters, roi_points,
         and roi_masks.
        """
        self.points_tracked = []
        ## Iterate through videos
        for i_video, video in tqdm(
            enumerate(self.dataset_videos), 
            desc='video #', 
            position=0, 
            leave=True, 
            disable=self._verbose < 2, 
            total=len(self.dataset_videos)
        ):
            ## If contiguous, use the last frame of the previous video
            if self._contiguous and i_video > 0:
                points_prev = points[-1]
                frame_prev = frame_last
            ## Otherwise, use the first frame of the current video
            else:
                points_prev = self.point_positions
                frame_prev = self._format_decordTorchVideo(video[0])

            ## Call point tracking function
            points, frame_last = self._track_points_singleVideo(video=video, points_prev=points_prev, frame_prev=frame_prev, batch_size=self._batch_size)

            ## Store points
            self.points_tracked.append(points)

    def _track_points_singleVideo(
        self,
        video,
        points_prev,
        frame_prev,
        batch_size=1000,
        ):
        """
        Track points in a single video.
        
        Args:
            video (Dataset_videos):
                A decord VideoReader object.
                Should be a single video object, where iterating over
                 the object returns frames. The frames should either
                  be 3D arrays of shape (height, width, channels) or
                  2D arrays of shape (height, width).
                If frames are not np.uint8, they will be converted.
            points_prev (np.ndarray, np.float32):
                A 2D array of np.float32, where each row is a point
                 to track. The points should be in the same order
                 as the points in the previous video. Order (y,x).
            frame_prev (np.ndarray, uint8):
                Previous frame. Should be formatted correctly, as no
                 corrective formatting will be done here.

        Returns:
            points (np.ndarray, np.float32):
                A 2D array of np.float32, where each row is a point
                 that has been tracked. Order (y,x).
            frame_last (np.ndarray, uint8):
                Last frame of the video. Formatted correctly.
        """
        ## Assert that video is a decord VideoReader object
        assert isinstance(video, decord.VideoReader), "FR ERROR: video must be a decord VideoReader object"
        ## Assert that points_prev is a 2D array of integers
        assert isinstance(points_prev, np.ndarray), "FR ERROR: points_prev must be a numpy array"
        assert points_prev.ndim == 2, "FR ERROR: points_prev must be a 2D array"
        assert points_prev.dtype == np.float32, "FR ERROR: points_prev must be a 2D array of np.float32"

        ## Preallocate points
        points_tracked = np.zeros((len(video), points_prev.shape[0], 2), dtype=np.float32)

        ## Make batches
        batches = make_batches(
            iterable=video,
            batch_size=batch_size,
            length=len(video),
        )

        ## Iterate through batches
        for i_batch, batch in tqdm(
            enumerate(batches),
            desc='batch #', 
            position=1, 
            leave=False, 
            disable=self._verbose < 2, 
            total=np.ceil(len(video)/batch_size).astype(int)
        ):
            ## Format batch
            batch = self._format_decordTorchVideo(batch, mask=self.mask)

            ## Iterate through frames in batch
            for i_frame, frame in tqdm(
                enumerate(batch),
                desc='frame #',
                position=2,
                leave=False,
                disable=self._verbose < 2,
                total=len(batch)
            ):
                ## Track points
                points_tracked[i_batch*batch_size + i_frame] = self._track_points_singleFrame(
                    frame_new=frame,
                    frame_prev=frame_prev,
                    points_prev=points_prev,
                )

                ## Update frame_prev
                frame_prev = frame
                ## Update points_prev
                points_prev = points_tracked[i_batch*batch_size + i_frame]

        frame_last = frame
        return points_tracked, frame_last

    def _track_points_singleFrame(self, frame_new, frame_prev, points_prev):
        """
        Track points in a single frame.
        
        Args:
            frame_new (np.ndarray, uint8):
                A 2D array of integers, where each element is a pixel
                 value. The frame should be the new frame.
            frame_prev (np.ndarray, uint8):
                A 2D array of integers, where each element is a pixel
                 value. The frame should be the previous frame.
            points_prev (np.ndarray, np.float32):
                A 2D array of np.float32, where each row is a point
                 to track. The points should be in the same order
                 as the points in the previous video. Order (y,x).
        """
        ## Call optical flow function
        points_new = self._optical_flow(frame_new=frame_new, frame_prev=frame_prev, points_prev=points_prev)

        return points_new

    def _format_decordTorchVideo(self, vid, mask=None):
        """
        Format a decord video array.
        
        Args:
            vid (decord NDAdarray):
                A 3D or 4D array of numbers, where each element is a pixel
                 value. Last dimension should be channels.
            mask (np.ndarray, bool):

        Returns:
            vid (np.ndarray, uint8):
                A 2D array of integers, where each element is a pixel
                 value.
        """
        ## Collapse channels
        vid = vid.type(torch.float32).mean(dim=-1).type(torch.uint8)

        ## Mask video
        if mask is not None:
            vid *= mask[None, :, :]

        ## Convert to numpy array
        vid = vid.numpy()

        return vid

    def _optical_flow(self, frame_new, frame_prev, points_prev):
        """
        Track points in a single frame using optical flow.
        
        Args:
            frame_new (np.ndarray, uint8):
                A 2D array of integers, where each element is a pixel
                 value. The frame should be the new frame.
            frame_prev (np.ndarray, uint8):
                A 2D array of integers, where each element is a pixel
                 value. The frame should be the previous frame.
            points_prev (np.ndarray, np.float32):
                A 2D array of np.float32, where each row is a point
                 to track. The points should be in the same order
                 as the points in the previous video. Order (y,x).
        """
        ## Call optical flow function
        if self.params_optical_flow['method'] == 'lucas_kanade':
            points_new, status, err = cv2.calcOpticalFlowPyrLK(frame_prev, frame_new, points_prev, None, **self.params_optical_flow['kwargs_method'])
        elif self.params_optical_flow['method'] == 'RLOF':
            points_new, status, err = cv2.calcOpticalFlowSparseRLOF(frame_prev, frame_new, points_prev, None, **self.params_optical_flow['kwargs_method'])

        return points_new

        # return points_prev


class Dataset_decord(torch.utils.data.Dataset):
    """
    Dataset class for decord videos.
    Allows for persistent workers to load videos into memory
     and perform preprocessing in background threads.
    """
    def __init__(
        self, 
        videoReader,
        transform=None,
    ):
        """
        Args:
            videoReader (decord.VideoReader):
                A decord VideoReader object.
            transform (callable, optional):
                Optional transform to be applied on a sample.
            verbose (int):
                The level of verbosity.
        """
        ## Store inputs
        self.videoReader = videoReader
        self.transform = transform

    def __len__(self):
        return len(self.videoReader)

    def __getitem__(self, idx):
        ## Get frame
        frame = self.videoReader[idx]

        ## Transform frame
        if self.transform:
            frame = self.transform(frame)

        return frame
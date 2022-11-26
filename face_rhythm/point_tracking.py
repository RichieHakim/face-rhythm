from typing import Union
import time

import numpy as np
from tqdm import tqdm
import decord
import cv2
import torch

from .util import FR_Module
from .data_importing import Dataset_videos
from .rois import ROIs
from .helpers import BufferedVideoReader
from .video_playback import FrameVisualizer

## Define class for performing point tracking using optical flow
class PointTracker(FR_Module):
    def __init__(
        self,
        buffered_video_reader: BufferedVideoReader,
        rois_points: ROIs,
        rois_masks: ROIs=None,
        contiguous: bool=False,
        params_optical_flow: dict={
                        "method": "lucas_kanade", ## method for optical flow. Only "lucas_kanade" is supported for now.
                        "point_spacing": 10,  ## spacing between points, in pixels
                        "mesh_rigidity": 0.005,  ## Rigidity of mesh. Changes depending on point spacing.
                        "relaxation": 0.5,  ## How quickly points relax back to their original position.
                        "kwargs_method": {
                            "winSize": (15,15),  ## Size of window to use for optical flow
                            "maxLevel": 2,  ## Maximum number of pyramid levels
                            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),  ## Stopping criteria for optical flow optimization
                        },
                    },
        params_outlier_handling: dict={
                        'threshold_displacement': 25,  ## Maximum displacement between frames, in pixels.
                        'framesHalted_before': 30,  ## Number of frames to halt tracking before a violation.
                        'framesHalted_after': 30,  ## Number of frames to halt tracking after a violation.
                    },
        visualize_video: bool=False,
        params_visualization: dict={
                        'alpha':1.0,
                        'point_sizes':1,
                        'writer_cv2':None,
        },
        verbose: Union[bool, int]=1,
    ):
        """
        Prepare for point tracking.
        Sets up parameters for optical flow, makes initial points,
         to track, and prepares masks for the video.

        Args:
            buffered_video_reader (BuffereVideoReader):
                A BufferedVideoReader object, containing the videos to track.
                Object comes from fr.helpers.BufferedVideoReader
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
                        "method": "lucas_kanade",  ## method for optical flow. Only "lucas_kanade" is supported for now.
                        "point_spacing": 10,  ## spacing between points, in pixels
                        "mesh_rigidity": 0.005,  ## Rigidity of mesh. Changes depending on point spacing.
                        "relaxation": 0.5,  ## How quickly points relax back to their original position.
                        "kwargs_method": {
                            "winSize": (15,15),  ## Size of window to use for optical flow
                            "maxLevel": 2,  ## Maximum number of pyramid levels
                            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),  ## Stopping criteria for optical flow optimization
                        },
                    }
                See https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
                 and https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
                 for more information about the lucas kanade optical flow parameters.
            params_outlier_handling (dict, optional):
                Parameters for outlier handling.
                Outliers/violations are frames when a point goes beyond a
                 threshold distance from its original position. For these
                 frames, the violating point has its velocity frozen for
                 'framesHalted_before' frames before and 'framesHalted_after'
                 frames after the violation.
                If None, the following parameters will be used:
                    params_outlier_handling = {
                        'threshold_displacement': 25,  ## Maximum displacement between frames, in pixels.
                        'framesHalted_before': 30,  ## Number of frames to halt tracking before a violation.
                        'framesHalted_after': 30,  ## Number of frames to halt tracking after a violation.
                    }
            visualize_video (bool, optional):
                Whether or not to visualize the video.
                If on a server or system without a display, this should be False.
            params_visualization (dict, optional):
                Parameters for visualization.
                If None, the following parameters will be used:
                    params_visualization = {
                        'alpha':1.0,
                        'point_sizes':1,
                        'writer_cv2':None,
                    }
                See fr.video_playback.FrameVisualizer for more information.
                Leave out 'points_colors' as this is reserved for outlier coloring.
            verbose (bool or int, optional):
                Whether or not to print progress updates.
                0: no progress updates
                1: warnings
                2: all info
        """
        ## Imports
        super().__init__()
        
        ## Set variables
        self._contiguous = bool(contiguous)
        self._verbose = int(verbose)
        self._visualize_video = bool(visualize_video)
        self._params_visualization = params_visualization.copy()
        self._params_outlier_handling = params_outlier_handling.copy()

        ## Assert that buffered_video_reader is a fr.helpers.BufferedVideoReader object
        type(buffered_video_reader)  ## For some reason this line is necessary for the next line to work
        assert isinstance(buffered_video_reader, BufferedVideoReader), "buffered_video_reader must be a fr.helpers.BufferedVideoReader object."
        ## Assert that the rois variables are either 2D arrays or lists of 2D arrays
        if isinstance(rois_points, np.ndarray):
            rois_points = [rois_points]
        if isinstance(rois_masks, np.ndarray):
            rois_masks = [rois_masks]
        assert isinstance(rois_points, list) and all([isinstance(roi, np.ndarray) for roi in rois_points]), "FR ERROR: rois_points must be a 2D array of booleans or a list of 2D arrays of booleans"
        assert all([roi.ndim == 2 for roi in rois_points]), "FR ERROR: rois_points must be a 2D array of booleans or a list of 2D arrays of booleans"
        assert all([roi.dtype == bool for roi in rois_points]), "FR ERROR: rois_points must be a 2D array of booleans or a list of 2D arrays of booleans"
        
        ## Set parameters for optical flow
        print("FR: Setting parameters for optical flow") if self._verbose > 1 else None
        params_default = {
                "method": "lucas_kanade",
                "point_spacing": 10,
                "mesh_rigidity": 0.005,
                "mesh_n_neighbors": 10,
                "relaxation": 0.5,
                "kwargs_method": {
                    "winSize": [15,15],
                    "maxLevel": 2,
                    "criteria": [cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03],
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
            self.params_optical_flow = params_optical_flow.copy()

        ## Make points within rois_points with spacing of point_spacing
        ##  First make a single ROI boolean image, then make points
        print("FR: Making points to track") if self._verbose > 1 else None
        rois_all = np.stack(rois_points, axis=0).all(axis=0)
        self.point_positions = self._make_points(rois_all, self.params_optical_flow["point_spacing"])
        print(f"FR: {self.point_positions.shape[0]} points will be tracked") if self._verbose > 1 else None

        ## Collapse masks into single mask
        print(f"FR: Collapsing mask ROI images into single mask") if self._verbose > 1 else None
        if rois_masks is None:
            self.mask = torch.ones(buffered_video_reader[0][0].shape[:2], dtype=bool)
        else:
            self.mask = torch.as_tensor(np.stack((rois_masks), axis=0).all(axis=0)).type(torch.bool)

        ## Store buffered_video_reader
        self.buffered_video_reader = buffered_video_reader
        ## Prepare video(s)
        self.buffered_video_reader.method_getitem = "by_video"
        if self._contiguous:
            video = self.buffered_video_reader
            video.method_getitem = "continuous"
            video.set_iterator_frame_idx(0)
            self.videos = [video]
        else:
            self.buffered_video_reader.method_getitem = "by_video"
            self.videos = [vid for vid in self.buffered_video_reader]

        ## Initialize mesh distances
        print("FR: Initializing mesh distances") if self._verbose > 1 else None
        p_0 = torch.as_tensor(self.point_positions.copy(), dtype=torch.float32)
        self.neighbors = torch.argsort(torch.linalg.norm(p_0.T[:,:,None] - p_0.T[:,None,:], ord=2, dim=0), dim=1)[:,:self.params_optical_flow["mesh_n_neighbors"]]
        self.d_0 = _vector_distance(torch.as_tensor(self.point_positions.copy(), dtype=torch.float32), self.neighbors)
        
        ## Preallocate points_tracked (will be overwrittenw with another empty list)
        self.points_tracked = []

        ## Prepare violation tracker
        self._pointIdx_violations_current = np.zeros((self.point_positions.shape[0]), dtype=bool)
        self._pointIdx_violations_countdown = np.zeros((self.point_positions.shape[0]), dtype=int)
        self._duration_violation_frames = self._params_outlier_handling["framesHalted_before"] + self._params_outlier_handling["framesHalted_after"]
        self._violation_event = False
        
        ## Prepare a playback visualizer
        if self._visualize_video:
            print("FR: Preparing playback visualizer") if self._verbose > 1 else None
            self.visualizer = FrameVisualizer(
                image_height_width=self.buffered_video_reader.frame_height_width,
                verbose=self._verbose,
            )


        ## For FR_Module compatibility
        self.config = {
            "contiguous": self._contiguous,
            "visualize_video": self._visualize_video,
            "params_optical_flow": self.params_optical_flow,
            "verbose": self._verbose,
        }
        self.run_info = {
        }
        self.run_data = {
            "point_positions": self.point_positions,
            "neighbors": self.neighbors,
            "mesh_d0": self.d_0,
            "mask": self.mask,
        }
        ## Append the self.run_info data to self.run_data
        self.run_data.update(self.run_info)


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
        ## flip to (x,y)
        points = np.fliplr(points)

        return points

    def track_points(self):
        """
        Point tracking workflow.
        Tracks points in videos using specified parameters, roi_points,
         and roi_masks.
        """
        ## Initialize points_tracked
        self.points_tracked = []

        ## Set the initial frame_prev as the first frame of the video
        print("FR: Setting initial frame_prev") if self._verbose > 1 else None
        frame_prev = self._format_decordTorchVideo_for_opticalFlow(vid=self.buffered_video_reader.get_frames_from_continuous_index(0), mask=self.mask)[0]
        ## Set the inital points_prev as the original points
        points_prev = self.point_positions.copy()

        
        ## Iterate through videos
        print("FR: Iterating point tracking through videos") if self._verbose > 1 else None
        for ii, video in tqdm(
            enumerate(self.videos), 
            desc='video #', 
            position=0, 
            leave=True, 
            disable=self._verbose < 2, 
            total=len(self.videos)
        ):
            ## If the video is not contiguous, set the iterator to the first frame
            frame_prev = self._format_decordTorchVideo_for_opticalFlow(vid=video.get_frames_from_continuous_index(0), mask=self.mask)[0] if not self._contiguous else frame_prev

            print(f"FR: Iterating through frames of video {ii}") if self._verbose > 2 else None
            points, frame_prev = self._track_points_singleVideo(
                video=video,
                points_prev=points_prev,
                frame_prev=frame_prev,
            )
            self.points_tracked.append(points)

        print(f"FR: Tracking complete") if self._verbose > 1 else None
        print(f"FR: Placing points_tracked into dictionary self.points_tracked where keys are video indices") if self._verbose > 1 else None
        self.points_tracked = {f"{ii}": points for ii, points in enumerate(self.points_tracked)}

        ## For FR_Module compatibility
        self.run_data["points_tracked"] = self.points_tracked


    def _track_points_singleVideo(
        self,
        video,
        points_prev,
        frame_prev,
    ):
        """
        Track points in a single video.
        
        Args:
            video (BufferedVideoReader):
                A BufferedVideoReader object.
                Should be a single video object, where iterating over
                 the object returns frames. The frames should either
                  be 3D arrays of shape (height, width, channels) or
                  2D arrays of shape (height, width).
                If frames are not np.uint8, they will be converted.
            points_prev (np.ndarray, np.float32):
                A 2D array of np.float32, where each row is a point
                 to track. The points should be in the same order
                 as the points in the previous video. Order (x,y).
            frame_prev (np.ndarray, uint8):
                Previous frame. Should be formatted correctly, as no
                 corrective formatting will be done here.

        Returns:
            points (np.ndarray, np.float32):
                A 2D array of np.float32, where each row is a point
                 that has been tracked. Order (x,y).
            frame_last (np.ndarray, uint8):
                Last frame of the video. Formatted correctly.
        """
        ## Assert that points_prev is a 2D array of integers
        assert isinstance(points_prev, np.ndarray), "FR ERROR: points_prev must be a numpy array"
        assert points_prev.ndim == 2, "FR ERROR: points_prev must be a 2D array"
        assert points_prev.dtype == np.float32, "FR ERROR: points_prev must be a 2D array of np.float32"

        ## Preallocate points
        points_tracked = np.zeros((len(video), points_prev.shape[0], 2), dtype=np.float32)

        ## Iterate through frames
        # video.set_iterator_frame_idx(0)
        # for i_frame, frame in tqdm(
        #     enumerate(video),
        #     desc='frame #',
        #     position=1,
        #     leave=False,
        #     disable=self._verbose < 2,
        #     total=len(video)
        # ):
        i_frame = 0
        video.set_iterator_frame_idx(0)
        with tqdm(total=len(video), desc='frame #', position=1, leave=False, disable=self._verbose < 2) as pbar:
            while (i_frame < len(video)):
                for frame in video:
                    frame_new = self._format_decordTorchVideo_for_opticalFlow(vid=frame[None,...], mask=self.mask)[0]
                    points_tracked[i_frame] = self._track_points_singleFrame(
                        frame_new=frame_new,
                        frame_prev=frame_prev,
                        points_prev=points_prev,
                    )
                    frame_prev = frame_new
                    points_prev = points_tracked[i_frame]

                    # print(i_frame)
                    # print(np.where(self._pointIdx_violations_current)[0])
                    if self._violation_event:
                        # print('FUCK')
                        self._violation_event = False
                        i_frame = max(i_frame - self._params_outlier_handling['framesHalted_before'], 0)
                        frame_prev = self._format_decordTorchVideo_for_opticalFlow(vid=video.get_frames_from_continuous_index(max(i_frame-1,0)), mask=self.mask)[0]
                        video.set_iterator_frame_idx(i_frame)
                        points_prev = points_tracked[i_frame]
                        break

                    pbar.n = i_frame
                    i_frame += 1
                    ## Update progress bar
                    pbar.update(1)


        ## clear buffered_video_reader
        video.delete_all_slots()

        return points_tracked, frame_prev


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
                 as the points in the previous video. Order (x,y).
        """
        ## Call optical flow function
        points_new = self._optical_flow(frame_new=frame_new, frame_prev=frame_prev, points_prev=points_prev)

        ## Update violations
        self._update_violations(points_new=points_new)

        ## Visualize points
        if self._visualize_video:
            self.visualizer.visualize_image_with_points(
                image=cv2.cvtColor(frame_new, cv2.COLOR_GRAY2BGR),
                # points=points_new[None,...].astype(np.int64),
                points=[points_new[self._pointIdx_violations_current].astype(np.int64), points_new[~self._pointIdx_violations_current].astype(np.int64)],
                points_colors=[(0,0,255), (0,255,0)],
                display=True,
                error_checking=True,
                **self._params_visualization,
            )

        return points_new

    def _update_violations(self, points_new):
        """
        Update violations.
        Finds points that violate the max displacement threshold.
        Sets the countdown for these points.
        Updates the countdown for existing violating points.
        Updates the violation event flag.
        Updates the current violation points.

        Args:
            points_new (np.ndarray, np.float32):
                The points that have been tracked. Order (x,y).
                A 2D array of np.float32, where each row is a point
                 to track.
        """
        displacement = points_new - self.point_positions
        pointIdx_violations_new = np.linalg.norm(displacement, axis=1) > self._params_outlier_handling['threshold_displacement']
        ## Find violating neighbors
        pointIdx_violations_new[self.neighbors.numpy()[pointIdx_violations_new].reshape(-1)] = True
        ## Determine if a violation event has occurred
        self._violation_event = np.any(pointIdx_violations_new)
        ## Update violation countdowns
        if self._violation_event:
            self._pointIdx_violations_countdown[self._pointIdx_violations_current] += self._params_outlier_handling['framesHalted_before'] + 1
        self._pointIdx_violations_countdown[pointIdx_violations_new] = self._duration_violation_frames + 1
        self._pointIdx_violations_countdown -= 1
        ## Find all points that are currently violating
        self._pointIdx_violations_current = self._pointIdx_violations_countdown > 0


    def _format_decordTorchVideo_for_opticalFlow(self, vid, mask=None):
        """
        Format a decord video array properly for optical flow.
        
        Args:
            vid (decord NDAdarray):
                A 4D array of numbers, where each element is a pixel
                 value. Last dimension should be channels.
                 (batch, height, width, channels)
            mask (np.ndarray, bool):

        Returns:
            vid (np.ndarray, uint8):
                A 3D array of integers, where each element is a pixel
                 value.
                 (batch, height, width)
        """
        ## Collapse channels and mask video
        vid = _helper_format_decordTorchVideo_for_opticalFlow(vid, mask=mask)

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
                 as the points in the previous video. Order (x,y).
        """
        ## Call optical flow function
        if self.params_optical_flow['method'] == 'lucas_kanade':
            points_new, status, err = cv2.calcOpticalFlowPyrLK(frame_prev, frame_new, points_prev, None, **self.params_optical_flow['kwargs_method'])
        else:
            raise ValueError("FR ERROR: optical flow method not recognized")

        ## Freeze violating points
        points_new[self._pointIdx_violations_current] = points_prev[self._pointIdx_violations_current]

        ## Apply mesh_rigity force
        points_new -= (_vector_displacement(torch.as_tensor(points_new, dtype=torch.float32), self.d_0, self.neighbors)*self.params_optical_flow['mesh_rigidity']).numpy()

        ## Apply relaxation force
        points_new -= (points_new-self.point_positions)*self.params_optical_flow['relaxation']
        

        return points_new
    
    def __repr__(self): return f"PointTracker(params_optical_flow={self.params_optical_flow}, visualize_video={self._visualize_video}, verbose={self._verbose})"
    def __getitem__(self, index): return self.points_tracked[index]
    def __len__(self): return len(self.points_tracked)
    def __iter__(self): return iter(self.points_tracked)
    def __next__(self): return next(self.points_tracked)
    

@torch.jit.script
def _helper_format_decordTorchVideo_for_opticalFlow(vid, mask=None):
    ## Collapse channels
    vid = vid.type(torch.float32).mean(dim=-1).type(torch.uint8)

    ## Mask video
    if mask is not None:
        return vid * mask[None, :, :]
    else:
        return vid

@torch.jit.script
def _vector_distance(pi, neighbors):
    """
    Calculate the distance between each point and its neighbors.

    Args:
        pi (torch.Tensor, dtype=torch.float32):
            A 2D array of torch.float32, where each row is a
             (y,x) point.
        neighbors (torch.Tensor, dtype=torch.int64):
            A 2D array of torch.int64 listing the indices of the
             neighbors of each point.

    Returns:
        d (torch.Tensor, dtype=torch.float32):
            A 2D array of torch.float32, where each row is a
             (y,x) vector describing the mean distance between
              each point and its neighbors.
    """
    pm = torch.tile(pi.T[:,:,None], (1,1,neighbors.shape[1]))
    d = pm - pi.T[:, neighbors]
    d2m = d.mean(2).T
    return d2m

@torch.jit.script
def _vector_displacement(di, dj, neighbors):
    """
    Calculate the displacement between each point and its
     neighbors relative to a reference distance (dj).

    Args:
        di (torch.Tensor, dtype=torch.float32):
            A 2D array of torch.float32, where each row is a
             (y,x) point.
        dj (torch.Tensor, dtype=torch.float32):
            A 2D array of torch.float32, where each row is a
             (y,x) distance vector.
        neighbors (torch.Tensor, dtype=torch.int64):
            A 2D array of torch.int64 listing the indices of the
             neighbors of each point.

    Returns:
        d (torch.Tensor, dtype=torch.float32):
            A 2D array of torch.float32, where each row is a
             (y,x) vector describing the mean displacement between
             each point and its neighbors relative to the reference
             distance.
    """
    return _vector_distance(di, neighbors) - dj
"""
Point tracking via Lucas-Kanade optical flow, with mesh-relaxation and outlier
handling.

``PointTracker`` advects a set of ``(x, y)`` seed points through a
``BufferedVideoReader`` using either CPU or CUDA OpenCV LK optical flow. Mesh
distances to k-nearest neighbors are regularized toward their initial values,
and frames with any point displaced beyond a threshold halt and replay the
surrounding region to suppress outlier streaks.
"""

from typing import Union, Optional, List
import time

import numpy as np
from tqdm.auto import tqdm
import cv2
import torch
import scipy.sparse

from .util import FR_Module
from .helpers import BufferedVideoReader
from .visualization import FrameVisualizer

## Define class for performing point tracking using optical flow
class PointTracker(FR_Module):
    """
    Tracks a set of seed points across video frames with Lucas-Kanade optical
    flow, mesh-rigidity regularization, and outlier handling.

    Wraps OpenCV LK (CPU or CUDA when available) and applies a k-nearest-
    neighbor mesh constraint plus a relaxation force toward the original
    point positions. Frames where any point is displaced beyond a threshold
    trigger a rewind-and-replay of the surrounding window so violating points
    are frozen. Optional ``frames_freeze`` masks proactively zero the optical
    flow delta on chosen frames.

    Args:
        buffered_video_reader (BufferedVideoReader):
            ``BufferedVideoReader`` object containing the videos to track.
            Created by ``fr.helpers.BufferedVideoReader``.
        point_positions (np.ndarray):
            Initial seed points to track. Each row is one point; columns
            are ``(x, y)``. Typically produced by ``fr.rois.ROIs`` via the
            ``ROIs.point_positions`` attribute. shape: *(n_points, 2)*,
            dtype: *float*.
        rois_masks (Union[np.ndarray, List[np.ndarray]]):
            ROI mask(s) used to zero-out non-ROI pixels before tracking.
            A single 2D bool array (shape: *(H, W)*) or a list of such
            arrays. When a list is provided, the masks are intersected
            into a single combined mask. (Default is ``None``)
        contiguous (bool):
            If ``True``, all videos are treated as one continuous stream
            (the first frame of each video continues from the previous
            video). If ``False``, point tracking restarts for each video.
            (Default is ``False``)
        params_optical_flow (dict):
            Parameters for optical flow. Missing keys fall back to defaults.
            Supported keys: \n
            * ``'method'``: Optical flow method. Only ``'lucas_kanade'`` is
              supported.
            * ``'mesh_rigidity'``: Strength of the mesh-distance restoring
              force. Depends on point spacing.
            * ``'mesh_n_neighbors'``: Number of nearest neighbors used for
              the mesh constraint.
            * ``'relaxation'``: Per-frame fraction by which points relax
              back toward their original positions.
            * ``'kwargs_method'``: Extra kwargs forwarded to
              ``cv2.calcOpticalFlowPyrLK`` (``winSize``, ``maxLevel``,
              ``criteria``). \n
            See the OpenCV LK optical flow docs for parameter meanings.
            (Default is the dict shown in the signature)
        params_clahe (dict):
            Keyword arguments forwarded to ``cv2.createCLAHE`` (or
            ``cv2.cuda.createCLAHE``). If ``None``, CLAHE is not applied.
            (Default is ``{'clipLimit': 40.0, 'tileGridSize': (150, 150)}``)
        params_outlier_handling (dict):
            Parameters for outlier (violation) handling. A violation is
            a frame in which a point exceeds the displacement threshold
            from its original position; on a violation the affected point
            has its velocity frozen for a window around the event.
            Supported keys: \n
            * ``'threshold_displacement'``: Maximum allowed displacement
              from the original position, in pixels.
            * ``'framesHalted_before'``: Number of frames to halt before
              a violation.
            * ``'framesHalted_after'``: Number of frames to halt after a
              violation. \n
            (Default is the dict shown in the signature)
        frames_freeze (Optional[np.ndarray]):
            1D bool array marking frames whose optical flow delta should
            be proactively zeroed. ``True`` = freeze the OF delta for that
            frame. Length should equal the total number of frames across
            all videos in contiguous mode, or per-video length in
            non-contiguous mode. (Default is ``None``)
        relaxation_during_freeze_frames (bool):
            Controls behavior on proactively frozen frames. If ``True``,
            the OF delta is zeroed but mesh rigidity and relaxation forces
            still apply, so the mesh can maintain shape and relax toward
            home positions. If ``False``, points are fully frozen and
            their positions are copied from the previous frame with no
            forces applied. (Default is ``True``)
        idx_start (Union[int, list, np.ndarray]):
            Index of the first frame to track. If an ``int``, it is used
            for all videos (or for the contiguous index when
            ``contiguous=True``). If a list/array and ``contiguous=False``,
            each entry is the start index for the corresponding video.
            (Default is ``0``)
        visualize_video (bool):
            If ``True``, displays the tracked frames via ``cv2.imshow``.
            Set to ``False`` on headless systems. (Default is ``False``)
        params_visualization (dict):
            Parameters forwarded to ``fr.visualization.FrameVisualizer``.
            Do not include ``'points_colors'`` since it is reserved for
            outlier coloring. (Default is
            ``{'alpha': 1.0, 'point_sizes': 1}``)
        verbose (Union[bool, int]):
            Verbosity level. \n
            * ``0``: silent.
            * ``1``: warnings only.
            * ``2``: all info. \n
            (Default is ``1``)

    Attributes:
        point_positions (np.ndarray):
            Initial seed point positions. shape: *(n_points, 2)*.
        num_points (int):
            Number of points being tracked.
        mask (torch.Tensor):
            Combined ROI mask. shape: *(H, W)*, dtype: *bool*.
        neighbors (torch.Tensor):
            Indices of the k-nearest neighbors of each point.
            shape: *(n_points, mesh_n_neighbors)*, dtype: *int64*.
        d_0 (torch.Tensor):
            Initial mean neighbor-distance vectors per point. dtype:
            *float32*.
        idx_start (Union[int, List[int]]):
            Resolved per-video (or contiguous) starting frame indices.
        videos (list):
            List of per-video iterables built from
            ``buffered_video_reader``.
        params_optical_flow (dict):
            Optical flow parameters with defaults filled in.
        params_outlier_handling (dict):
            Outlier-handling parameters with defaults filled in.
        params_visualization (dict):
            Visualization parameters with defaults filled in.
        points_tracked (Union[list, dict]):
            Tracked point arrays. Populated by ``track_points``; first
            stored as a list of arrays per video, then re-keyed as a dict
            ``{str(video_idx): np.ndarray}``.
        violations (list):
            Per-video sparse COO matrices of violation flags (populated
            by ``track_points``).
        violations_sparseCOO (dict):
            Per-video violation flags packed as ``{'row', 'col', 'data',
            'shape'}`` dicts (populated by ``track_points``).
        violation_fraction (List[float]):
            Per-video fraction of frames that contain at least one
            violation.
        config (dict):
            Snapshot of the configuration used to construct the tracker.
        run_info (dict):
            Run summary populated by ``track_points``.
        run_data (dict):
            Run data dictionary used by ``FR_Module`` save/load.
    """
    def __init__(
        self,
        buffered_video_reader: BufferedVideoReader,
        point_positions: np.ndarray,
        rois_masks: Optional[Union[np.ndarray, List[np.ndarray]]]=None,
        contiguous: bool=False,
        params_optical_flow: dict={
                        "method": "lucas_kanade", ## method for optical flow. Only "lucas_kanade" is supported for now.
                        "mesh_rigidity": 0.005,  ## Rigidity of mesh. Changes depending on point spacing.
                        "relaxation": 0.5,  ## How quickly points relax back to their original position.
                        "kwargs_method": {
                            "winSize": (15,15),  ## Size of window to use for optical flow
                            "maxLevel": 2,  ## Maximum number of pyramid levels
                            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),  ## Stopping criteria for optical flow optimization
                        },
                    },
        params_clahe: dict={
                        "clipLimit": 40.0,
                        "tileGridSize": (150, 150),
                    },
        params_outlier_handling: dict={
                        'threshold_displacement': 25,  ## Maximum displacement between frames, in pixels.
                        'framesHalted_before': 30,  ## Number of frames to halt tracking before a violation.
                        'framesHalted_after': 30,  ## Number of frames to halt tracking after a violation.
                    },
        frames_freeze: Optional[np.ndarray]=None,
        relaxation_during_freeze_frames: bool=True,
        idx_start: Union[int, list, np.ndarray]=0,
        visualize_video: bool=False,
        params_visualization: dict={
                        'alpha':1.0,
                        'point_sizes':1,
        },
        verbose: Union[bool, int]=1,
    ):
        """Initializes the tracker, validates inputs, and prepares masks, mesh, and optical flow state."""
        ## Imports
        super().__init__()
        
        ## Set variables
        self._contiguous = bool(contiguous)
        self._verbose = int(verbose)
        self._visualize_video = bool(visualize_video)
        self._params_visualization = params_visualization.copy()
        self._params_clahe = params_clahe.copy() if params_clahe is not None else None
        self._params_outlier_handling = params_outlier_handling.copy()
        self._frame_start = int(idx_start)
        self._relaxation_during_freeze_frames = bool(relaxation_during_freeze_frames)

        ## Store and validate frames_freeze
        if frames_freeze is not None:
            assert isinstance(frames_freeze, np.ndarray), "FR ERROR: frames_freeze must be a 1D boolean numpy array."
            assert frames_freeze.ndim == 1, "FR ERROR: frames_freeze must be a 1D array."
            assert frames_freeze.dtype == bool, "FR ERROR: frames_freeze must have dtype bool."
            self._frames_freeze = frames_freeze
            print(f"FR: frames_freeze provided with {frames_freeze.sum()} frozen frames out of {len(frames_freeze)} total.") if self._verbose > 1 else None
            print(f"FR: relaxation_during_freeze_frames={self._relaxation_during_freeze_frames}") if self._verbose > 1 else None
        else:
            self._frames_freeze = None

        ## Detect CUDA OpenCV support
        ## cv2.cuda may not exist if OpenCV was built without CUDA, so
        ## we check for the attribute explicitly rather than catching errors.
        self._use_cuda_cv2 = (
            hasattr(cv2, 'cuda')
            and hasattr(cv2.cuda, 'getCudaEnabledDeviceCount')
            and cv2.cuda.getCudaEnabledDeviceCount() > 0
        )
        if self._use_cuda_cv2:
            print("FR: CUDA OpenCV detected, using GPU-accelerated optical flow and CLAHE") if verbose > 1 else None
        else:
            print("FR: CUDA OpenCV not available, using CPU optical flow") if verbose > 1 else None

        ## Cache CLAHE object (avoid re-creating every frame)
        if self._params_clahe is not None:
            if self._use_cuda_cv2:
                self._clahe = cv2.cuda.createCLAHE(**self._params_clahe)
            else:
                self._clahe = cv2.createCLAHE(**self._params_clahe)
        else:
            self._clahe = None

        ## Setup CUDA optical flow if available
        self._lk_gpu = None
        self._frame_prev_gpu = None

        ## Assert that buffered_video_reader is a fr.helpers.BufferedVideoReader object
        type(buffered_video_reader), isinstance(buffered_video_reader, BufferedVideoReader)  ## line needed sometimes for next assert to work
        assert isinstance(buffered_video_reader, BufferedVideoReader), "buffered_video_reader must be a fr.helpers.BufferedVideoReader object."
        ## Assert that point_positions is a 2D array of floats
        assert isinstance(point_positions, np.ndarray), "point_positions must be a 2D array of floats. Use the fr.rois.ROIs class to generate using .point_positions attribute."
        ## Assert that the rois variables are either 2D arrays or lists of 2D arrays
        if isinstance(rois_masks, np.ndarray):
            rois_masks = [rois_masks]
        ## Assert that params_optical_flow is a dict
        assert isinstance(params_optical_flow, dict), "FR ERROR: params_optical_flow must be a dict"
        ## Assert that params_outlier_handling is a dict
        assert isinstance(params_outlier_handling, dict), "FR ERROR: params_outlier_handling must be a dict"
        ## Assert that params_visualization is a dict
        assert isinstance(params_visualization, dict), "FR ERROR: params_visualization must be a dict"
        ## Assert that idx_start is an integer, list, or numpy array
        assert isinstance(idx_start, (int, list, np.ndarray)), "FR ERROR: idx_start must be an integer, list, or numpy array"
        
        ## Prepare start indices
        if self._contiguous:
            assert isinstance(idx_start, int), "FR ERROR: idx_start must be an integer if contiguous is True"
            self.idx_start = idx_start
        else:
            if isinstance(idx_start, int):
                self.idx_start = [idx_start]*len(buffered_video_reader)
            elif isinstance(idx_start, (list, np.ndarray)):
                assert len(idx_start) == len(buffered_video_reader), "FR ERROR: idx_start must have the same length as the number of videos"
                assert all([isinstance(i, int) for i in idx_start]), "FR ERROR: idx_start must be a list of integers"
                self.idx_start = list(idx_start)
            else:
                raise ValueError("FR ERROR: idx_start must be an integer, list, or numpy array")

        ## Define default parameters
        params_optFlow_default = {
                "method": "lucas_kanade",
                "mesh_rigidity": 0.005,
                "mesh_n_neighbors": 10,
                "relaxation": 0.5,
                "kwargs_method": {
                    "winSize": [15,15],
                    "maxLevel": 2,
                    "criteria": [cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03],
                },
            }
        params_outlierHandling_default = {
                'threshold_displacement': 25,
                'framesHalted_before': 30,
                'framesHalted_after': 30,
            }
        params_visualization_default = {
                'alpha':1.0,
                'point_sizes':1,
            }

        ## Fill in default parameters
        ## if the parameter was passed as an arg then place it in the args dict, otherwise use the default
        ### params_optical_flow
        params_missing = {key: params_optFlow_default[key] for key in params_optFlow_default if key not in params_optical_flow}
        print(f"FR WARNING: Following parameters for optical flow were not specified and will be set to default values: {params_missing}") if ((self._verbose > 0) and (len(params_missing) > 0)) else None
        self.params_optical_flow = {**params_optical_flow, **params_missing}
        ### params_outlier_handling
        params_missing = {key: params_outlierHandling_default[key] for key in params_outlierHandling_default if key not in params_outlier_handling}
        print(f"FR WARNING: Following parameters for outlier handling were not specified and will be set to default values: {params_missing}") if ((self._verbose > 0) and (len(params_missing) > 0)) else None
        self.params_outlier_handling = {**params_outlier_handling, **params_missing}
        ### params_visualization
        params_missing = {key: params_visualization_default[key] for key in params_visualization_default if key not in params_visualization}
        print(f"FR WARNING: Following parameters for visualization were not specified and will be set to default values: {params_missing}") if ((self._verbose > 0) and (len(params_missing) > 0)) else None
        self.params_visualization = {**params_visualization, **params_missing}

        ## Setup CUDA LK optical flow (requires params_optical_flow to be set)
        if self._use_cuda_cv2:
            self._lk_gpu = cv2.cuda.SparsePyrLKOpticalFlow_create(
                winSize=tuple(self.params_optical_flow['kwargs_method']['winSize']),
                maxLevel=self.params_optical_flow['kwargs_method']['maxLevel'],
            )

        ## Retrive point positions
        self.point_positions = point_positions
        self.num_points = self.point_positions.shape[0]
        print(f"FR: {self.point_positions.shape[0]} points will be tracked") if self._verbose > 1 else None

        ## Collapse masks into single mask
        print(f"FR: Collapsing mask ROI images into single mask") if self._verbose > 1 else None
        if rois_masks is None:
            ## Use the reader's (H, W) metadata, not an indexed frame: ``buffered_video_reader[0][0]``
            ## is a 3D frame (H, W, C) in 'continuous' mode but a 4D batch (1, H, W, C) in 'by_video'
            ## mode (the standard pipeline mode), so ``shape[:2]`` would give a degenerate (1, H) mask.
            self.mask = torch.ones(tuple(buffered_video_reader.frame_height_width), dtype=bool)
        else:
            self.mask = torch.as_tensor(np.stack((rois_masks), axis=0).all(axis=0)).type(torch.bool)

        ## Store buffered_video_reader
        self.buffered_video_reader = buffered_video_reader
        ## Prepare video(s)
        self.buffered_video_reader.method_getitem = "by_video"
        if self._contiguous:
            video = self.buffered_video_reader
            video.method_getitem = "continuous"

            self.videos = [video]
        else:
            self.buffered_video_reader.method_getitem = "by_video"
            self.videos = [vid for vid in self.buffered_video_reader]

        ## Initialize mesh distances
        print("FR: Initializing mesh distances") if self._verbose > 1 else None
        p_0 = torch.as_tensor(self.point_positions.copy(), dtype=torch.float32)
        self.neighbors = torch.argsort(torch.linalg.norm(p_0.T[:,:,None] - p_0.T[:,None,:], ord=2, dim=0), dim=1)[:,:self.params_optical_flow["mesh_n_neighbors"]]
        self.d_0 = _vector_distance(torch.as_tensor(self.point_positions.copy(), dtype=torch.float32), self.neighbors)
        
        ## Preallocate points_tracked (will be overwritten with another empty list)
        self.points_tracked = []

        ## Prepare violation tracker
        self._pointIdx_violations_current = np.zeros((self.point_positions.shape[0]), dtype=bool)
        self._violation_event = False
        
        ## Prepare a playback visualizer
        self._handle_cv2Imshow = "PointTracker"
        if self._visualize_video:
            print("FR: Preparing playback visualizer") if self._verbose > 1 else None
            self.visualizer = FrameVisualizer(
                display=True,
                handle_cv2Imshow=self._handle_cv2Imshow,
                path_save=None,
                frame_height_width=self.buffered_video_reader.frame_height_width,
                error_checking=False,
                verbose=self._verbose,
            )


        ## For FR_Module compatibility
        self.config = {
            "contiguous": self._contiguous,
            "visualize_video": self._visualize_video,
            "params_optical_flow": self.params_optical_flow,
            "params_outlier_handling": self._params_outlier_handling,
            "params_visualization": self._params_visualization,
            "frames_freeze": self._frames_freeze is not None,
            "relaxation_during_freeze_frames": self._relaxation_during_freeze_frames,
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

    # def _make_points(self, roi, point_spacing):
    #     """
    #     Make points within a roi with spacing of point_spacing.
        
    #     Args:
    #         roi (np.ndarray, boolean):
    #             A 2D array of booleans, where True indicates a pixel
    #              that is within the region of interest.
    #         point_spacing (int):
    #             The spacing between points, in pixels.

    #     Returns:
    #         points (np.ndarray, np.float32):
    #             A 2D array of integers, where each row is a point
    #              to track.
    #     """
    #     ## Assert that roi is a 2D array of booleans
    #     assert isinstance(roi, np.ndarray), "FR ERROR: roi must be a numpy array"
    #     assert roi.ndim == 2, "FR ERROR: roi must be a 2D array"
    #     assert roi.dtype == bool, "FR ERROR: roi must be a 2D array of booleans"
    #     ## Warn if point_spacing is not an integer. It will be rounded.
    #     if not isinstance(point_spacing, int):
    #         print("FR WARNING: point_spacing must be an integer. It will be rounded.")
    #         point_spacing = int(round(point_spacing))

    #     ## make point cloud
    #     y, x = np.where(roi)
    #     y_min, y_max = y.min(), y.max()
    #     x_min, x_max = x.min(), x.max()
    #     y_points = np.arange(y_min, y_max, point_spacing)
    #     x_points = np.arange(x_min, x_max, point_spacing)
    #     y_points, x_points = np.meshgrid(y_points, x_points)
    #     y_points = y_points.flatten()
    #     x_points = x_points.flatten()
    #     ## remove points outside of roi
    #     points = np.stack([y_points, x_points], axis=1)
    #     points = points[roi[points[:, 0], points[:, 1]]].astype(np.float32)
    #     ## flip to (x,y)
    #     points = np.fliplr(points)

    #     return points

    def cleanup(self):
        """
        Deletes all instance attributes and runs garbage collection to free
        the large arrays held by the tracker.
        """
        import gc
        print(f"FR: Deleting all attributes")
        while len(self.__dict__.keys()) > 0:
            key = list(self.__dict__.keys())[0]
            del self.__dict__[key]
        gc.collect()
        gc.collect()


    def track_points(self):
        """
        Runs the full point tracking workflow across all videos.

        Tracks the seed points through every video using the configured
        optical flow, mesh, outlier handling, and freeze parameters.
        Populates ``self.points_tracked``, ``self.violations``,
        ``self.violations_sparseCOO``, ``self.violation_fraction``, and the
        ``run_info``/``run_data`` dictionaries used by ``FR_Module``.
        """
        ## Initialize points_tracked
        self.points_tracked = []
        self.violations = []

        ## Set the initial frame_prev as the first frame of the video
        print("FR: Setting initial frame_prev") if self._verbose > 1 else None
        frame_prev = self._format_decordTorchVideo_for_opticalFlow(vid=self.buffered_video_reader.get_frames_from_continuous_index(0), mask=self.mask)[0]
        self.buffered_video_reader.wait_for_loading()
        ## Set the inital points_prev as the original points
        points_prev = self.point_positions.copy()

        
        ## Iterate through videos
        print("FR: Iterating point tracking through videos") if self._verbose > 1 else None
        for ii, video in tqdm(
            enumerate(self.videos), 
            desc='video #', 
            position=1, 
            leave=True, 
            disable=self._verbose < 2, 
            total=len(self.videos)
        ):
            ## If the video is not contiguous, set the iterator to the first frame
            if not self._contiguous:
                frame_prev = self._format_decordTorchVideo_for_opticalFlow(vid=video.get_frames_from_continuous_index(0), mask=self.mask)[0]
                self._frame_prev_gpu = None  ## Reset GPU frame cache for new video

            ## Determine per-video freeze array
            if self._frames_freeze is not None:
                if self._contiguous:
                    frames_freeze_video = self._frames_freeze
                else:
                    ## Slice the freeze array for this video
                    n_frames_before = sum(len(self.videos[jj]) for jj in range(ii))
                    n_frames_video = len(video)
                    frames_freeze_video = self._frames_freeze[n_frames_before : n_frames_before + n_frames_video]
            else:
                frames_freeze_video = None

            print(f"FR: Iterating through frames of video {ii}") if self._verbose > 2 else None
            points, frame_prev = self._track_points_singleVideo(
                video=video,
                points_prev=points_prev,
                frame_prev=frame_prev,
                idx_start=self.idx_start if self._contiguous else self.idx_start[ii],
                frames_freeze=frames_freeze_video,
            )
            self.points_tracked.append(points)
            self.violations.append(self.violations_currentVideo.tocoo())

        ## Destroy the cv2.imshow window
        if self._visualize_video:
            cv2.destroyWindow(self._handle_cv2Imshow)

        print(f"FR: Tracking complete") if self._verbose > 1 else None
        print(f"FR: Placing points_tracked into dictionary self.points_tracked where keys are video indices") if self._verbose > 1 else None
        self.points_tracked = {f"{ii}": points for ii, points in enumerate(self.points_tracked)}
        print(f"FR: Placing violations into dictionary self.violations where keys are video indices") if self._verbose > 1 else None
        self.violations_sparseCOO = {f"{ii}": {'row': v.row, 'col': v.col, 'data': v.data, 'shape': v.shape} for ii, v in enumerate(self.violations)}

        self.violation_fraction = [np.array(v.max(1).toarray()).squeeze().mean() for v in self.violations]

        ## For FR_Module compatibility
        self.run_info["num_videos"] = int(len(self.videos))
        self.run_info["num_frames"] = [int(v.num_frames_total) for v in self.videos]
        self.run_info["num_frames_total"] = int(sum(self.run_info["num_frames"]))
        self.run_info["num_points"] = int(self.num_points)
        self.run_info["violation_fraction"] = [int(v) for v in self.violation_fraction]

        self.run_data["points_tracked"] = self.points_tracked
        self.run_data["violations"] = self.violations_sparseCOO


    def _track_points_singleVideo(
        self,
        video,
        points_prev,
        frame_prev,
        idx_start=None,
        frames_freeze=None,
    ):
        """
        Tracks points through a single video, handling rewind on violations.

        Args:
            video (BufferedVideoReader):
                Single video iterable yielding frames as 3D arrays
                (shape: *(H, W, C)*) or 2D arrays (shape: *(H, W)*).
                Non-uint8 frames are converted internally.
            points_prev (np.ndarray):
                Previous-frame point positions, ordered ``(x, y)``.
                shape: *(n_points, 2)*, dtype: *float32*.
            frame_prev (np.ndarray):
                Already-formatted previous frame; no corrective formatting
                is applied here. dtype: *uint8*.
            idx_start (Optional[int]):
                Index of the first frame to track. If ``None``, falls back
                to the start index resolved in ``__init__``. (Default is
                ``None``)
            frames_freeze (Optional[np.ndarray]):
                1D bool array for this video. ``True`` = freeze the OF
                delta for that frame. Length must equal the number of
                frames in the video. If ``None``, no proactive freezing
                is applied. (Default is ``None``)

        Returns:
            (tuple): tuple containing:
                points_tracked (np.ndarray):
                    Tracked point positions for every frame of this video.
                    Order ``(x, y)``. shape: *(n_frames, n_points, 2)*,
                    dtype: *float32*.
                frame_last (np.ndarray):
                    Final formatted frame from the video. dtype: *uint8*.
        """
        ## Assert that points_prev is a 2D array of integers
        assert isinstance(points_prev, np.ndarray), "FR ERROR: points_prev must be a numpy array"
        assert points_prev.ndim == 2, "FR ERROR: points_prev must be a 2D array"
        points_prev = points_prev.astype(np.float32)

        ## Store per-video freeze array
        if frames_freeze is not None:
            assert len(frames_freeze) == len(video), f"FR ERROR: frames_freeze length ({len(frames_freeze)}) must match video length ({len(video)})."
            self._frames_freeze_currentVideo = frames_freeze
        else:
            self._frames_freeze_currentVideo = None

        ## Preallocate points
        points_tracked = np.zeros((len(video), points_prev.shape[0], 2), dtype=np.float32)
        self.violations_currentVideo = scipy.sparse.lil_matrix((len(video), points_prev.shape[0]), dtype=np.bool_)

        self.i_frame = self._frame_start if idx_start is None else idx_start
        video.set_iterator_frame_idx(self._frame_start)
        with tqdm(total=len(video), desc='frame #', position=0, leave=True, disable=self._verbose < 2, mininterval=5.0) as pbar:
            while (self.i_frame < len(video)):
                for frame in video:
                    frame_new = self._format_decordTorchVideo_for_opticalFlow(vid=frame[None,...], mask=self.mask)[0]
                    points_tracked[self.i_frame] = self._track_points_singleFrame(
                        frame_new=frame_new,
                        frame_prev=frame_prev,
                        points_prev=points_prev,
                    )
                    frame_prev = frame_new
                    points_prev = points_tracked[self.i_frame]
                    if self._violation_event:
                        self._violation_event = False
                        self.i_frame = max(self.i_frame - self._params_outlier_handling['framesHalted_before'], 0)
                        frame_prev = self._format_decordTorchVideo_for_opticalFlow(vid=video.get_frames_from_continuous_index(max(self.i_frame-1,0)), mask=self.mask)[0]
                        self._frame_prev_gpu = None  ## Reset GPU frame cache on rewind
                        video.set_iterator_frame_idx(self.i_frame)
                        points_prev = points_tracked[self.i_frame]
                        break

                    pbar.n = self.i_frame
                    self.i_frame += 1
                    ## Update progress bar
                    pbar.update(1)


        ## clear buffered_video_reader
        video.delete_all_slots()

        return points_tracked, frame_prev


    def _track_points_singleFrame(self, frame_new, frame_prev, points_prev):
        """
        Tracks points from one frame to the next and updates violations.

        Calls the optical flow routine, updates the violation tracker, and
        optionally renders the result via the ``FrameVisualizer``.

        Args:
            frame_new (np.ndarray):
                Current frame. shape: *(H, W)*, dtype: *uint8*.
            frame_prev (np.ndarray):
                Previous frame. shape: *(H, W)*, dtype: *uint8*.
            points_prev (np.ndarray):
                Previous-frame point positions, ordered ``(x, y)``.
                shape: *(n_points, 2)*, dtype: *float32*.

        Returns:
            (np.ndarray):
                points_new (np.ndarray):
                    Tracked point positions for this frame. Order
                    ``(x, y)``. shape: *(n_points, 2)*, dtype: *float32*.
        """
        ## Call optical flow function
        points_new = self._optical_flow(frame_new=frame_new, frame_prev=frame_prev, points_prev=points_prev)

        ## Update violations
        violations_frame = self._update_violations(points_new=points_new)

        ## Visualize points
        if self._visualize_video:
            self.visualizer.visualize_image_with_points(
                image=cv2.cvtColor(frame_new, cv2.COLOR_GRAY2BGR),
                # points=points_new[None,...].astype(np.int64),
                points=[points_new[self._pointIdx_violations_current].astype(np.int64), points_new[~self._pointIdx_violations_current].astype(np.int64)],
                points_colors=[(0,0,255), (0,255,0)],
                # points_colors=[(0,0,255), (0,0,255)],
                **self._params_visualization,
            )

        return points_new

    def _update_violations(self, points_new):
        """
        Updates the violation tracker for the current frame.

        Flags points (and their mesh neighbors) whose displacement from
        the original position exceeds the threshold, marks the surrounding
        ``framesHalted_before``/``framesHalted_after`` window in
        ``self.violations_currentVideo``, sets ``self._violation_event``,
        and refreshes ``self._pointIdx_violations_current``.

        Args:
            points_new (np.ndarray):
                Tracked point positions for the current frame, ordered
                ``(x, y)``. shape: *(n_points, 2)*, dtype: *float32*.
        """
        displacement = points_new - self.point_positions
        pointIdx_violations_new = np.linalg.norm(displacement, axis=1) > self._params_outlier_handling['threshold_displacement']
        ## Find violating neighbors
        pointIdx_violations_new[np.unique(self.neighbors.numpy()[pointIdx_violations_new].reshape(-1))] = True
        ## Determine if a violation event has occurred
        self._violation_event = np.any(pointIdx_violations_new)
        ## Update violation countdowns
        if self._violation_event:
            self.violations_currentVideo[
                max(self.i_frame - self._params_outlier_handling['framesHalted_before'], 0) : \
                min(self.i_frame + self._params_outlier_handling['framesHalted_after'], self.violations_currentVideo.shape[0]),
                pointIdx_violations_new
            ] = True
        ## Find all points that are currently violating
        self._pointIdx_violations_current = self.violations_currentVideo[self.i_frame].toarray().squeeze()


    def _format_decordTorchVideo_for_opticalFlow(self, vid, mask=None):
        """
        Formats a decord/torch video tensor into the layout expected by the
        optical flow routine.

        Performs grayscale conversion and ROI masking (on GPU when ``vid``
        is CUDA), transfers to CPU NumPy, and optionally applies CLAHE.

        Args:
            vid (torch.Tensor):
                Input frames. shape: *(batch, H, W, C)*, dtype: *uint8*.
                Can live on CPU or CUDA; grayscale and masking run on the
                same device as ``vid``.
            mask (torch.Tensor):
                ROI mask. Pixels where ``mask`` is ``False`` are zeroed.
                shape: *(H, W)*, dtype: *bool*. (Default is ``None``)

        Returns:
            (np.ndarray):
                vid (np.ndarray):
                    Grayscale, masked, optionally CLAHE-enhanced frames.
                    shape: *(batch, H, W)*, dtype: *uint8*.
        """
        ## Move mask to same device as vid for the JIT function
        if mask is not None and mask.device != vid.device:
            mask = mask.to(device=vid.device)

        ## Grayscale + mask (runs on whatever device vid is on)
        vid = _helper_format_decordTorchVideo_for_opticalFlow(vid, mask=mask)

        ## Transfer to CPU numpy for CLAHE (cv2 requires numpy input)
        vid = vid.cpu().numpy() if vid.is_cuda else vid.numpy()

        ## Perform CLAHE
        if self._clahe is not None:
            if self._use_cuda_cv2:
                frames_out = []
                for frame in vid:
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(frame)
                    frames_out.append(self._clahe.apply(gpu_mat, cv2.cuda.Stream_Null()).download())
                vid = np.stack(frames_out, axis=0)
            else:
                vid = np.stack([self._clahe.apply(frame) for frame in vid], axis=0)

        return vid

    def _optical_flow(self, frame_new, frame_prev, points_prev):
        """
        Runs one optical flow step and applies freeze, mesh, and relaxation
        forces.

        Dispatches to CUDA or CPU Lucas-Kanade based on availability,
        applies proactive (``frames_freeze``) and reactive (violation)
        freezing, then subtracts the mesh-rigidity and relaxation forces.

        Args:
            frame_new (np.ndarray):
                Current frame. shape: *(H, W)*, dtype: *uint8*.
            frame_prev (np.ndarray):
                Previous frame. shape: *(H, W)*, dtype: *uint8*.
            points_prev (np.ndarray):
                Previous-frame point positions, ordered ``(x, y)``.
                shape: *(n_points, 2)*, dtype: *float32*.

        Returns:
            (np.ndarray):
                points_new (np.ndarray):
                    Updated point positions for the current frame after
                    freeze, mesh-rigidity, and relaxation forces. Order
                    ``(x, y)``. shape: *(n_points, 2)*, dtype: *float32*.
        """
        ## Call optical flow function
        if self.params_optical_flow['method'] == 'lucas_kanade':
            if self._use_cuda_cv2 and self._lk_gpu is not None:
                ## Upload frames and points to GPU
                frame_new_gpu = cv2.cuda_GpuMat()
                frame_new_gpu.upload(frame_new)
                if self._frame_prev_gpu is None:
                    self._frame_prev_gpu = cv2.cuda_GpuMat()
                    self._frame_prev_gpu.upload(frame_prev)
                pts_gpu = cv2.cuda_GpuMat()
                pts_gpu.upload(points_prev.reshape(1, -1, 2))
                ## Run CUDA LK
                pts_new_gpu, status_gpu, err_gpu = self._lk_gpu.calc(
                    self._frame_prev_gpu, frame_new_gpu, pts_gpu, None
                )
                points_new = pts_new_gpu.download().reshape(-1, 2)
                ## Keep current frame on GPU for next iteration
                self._frame_prev_gpu = frame_new_gpu
            else:
                points_new, status, err = cv2.calcOpticalFlowPyrLK(frame_prev, frame_new, points_prev, None, **self.params_optical_flow['kwargs_method'])
        else:
            raise ValueError("FR ERROR: optical flow method not recognized")

        ## Proactive freeze — zero OF delta for this frame
        _is_frozen_frame = (
            self._frames_freeze_currentVideo is not None
            and self._frames_freeze_currentVideo[self.i_frame]
        )
        if _is_frozen_frame and not self._relaxation_during_freeze_frames:
            ## Fully frozen: copy previous positions exactly, no forces applied
            return points_prev.copy()

        if _is_frozen_frame:
            ## Zero the OF delta but allow mesh rigidity and relaxation to apply
            points_new = points_prev.copy()

        ## Freeze violating points (reactive)
        points_new[self._pointIdx_violations_current] = points_prev[self._pointIdx_violations_current]

        ## Apply mesh_rigity force
        points_new -= (_vector_displacement(torch.as_tensor(points_new, dtype=torch.float32), self.d_0, self.neighbors) * self.params_optical_flow['mesh_rigidity']).numpy()
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
    """
    TorchScript helper that converts a 4D color video tensor to grayscale
    and applies an optional ROI mask.

    Args:
        vid (torch.Tensor):
            Input frames. shape: *(batch, H, W, C)*, dtype: *uint8*.
        mask (torch.Tensor):
            Optional ROI mask; pixels where ``mask`` is ``False`` are
            zeroed. shape: *(H, W)*, dtype: *bool*. (Default is ``None``)

    Returns:
        (torch.Tensor):
            vid (torch.Tensor):
                Grayscale (and optionally masked) frames.
                shape: *(batch, H, W)*, dtype: *uint8*.
    """
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
    Computes the mean ``(x, y)`` distance vector between each point and its
    nearest neighbors.

    Args:
        pi (torch.Tensor):
            Point coordinates, one ``(y, x)`` per row.
            shape: *(n_points, 2)*, dtype: *float32*.
        neighbors (torch.Tensor):
            Indices of each point's nearest neighbors.
            shape: *(n_points, k)*, dtype: *int64*.

    Returns:
        (torch.Tensor):
            d (torch.Tensor):
                Mean neighbor-distance vector per point.
                shape: *(n_points, 2)*, dtype: *float32*.
    """
    pm = torch.tile(pi.T[:,:,None], (1,1,neighbors.shape[1]))
    d = pm - pi.T[:, neighbors]
    d2m = d.mean(2).T
    return d2m

@torch.jit.script
def _vector_displacement(di, dj, neighbors):
    """
    Computes the mean displacement of each point from its neighbors relative
    to a reference distance vector.

    Args:
        di (torch.Tensor):
            Current point coordinates, one ``(y, x)`` per row.
            shape: *(n_points, 2)*, dtype: *float32*.
        dj (torch.Tensor):
            Reference per-point distance vectors (e.g. ``self.d_0``).
            shape: *(n_points, 2)*, dtype: *float32*.
        neighbors (torch.Tensor):
            Indices of each point's nearest neighbors.
            shape: *(n_points, k)*, dtype: *int64*.

    Returns:
        (torch.Tensor):
            d (torch.Tensor):
                Per-point mean displacement relative to ``dj``.
                shape: *(n_points, 2)*, dtype: *float32*.
    """
    return _vector_distance(di, neighbors) - dj
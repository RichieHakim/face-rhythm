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
                frame_prev = self._format_decordTorchVideo_for_opticalFlow(video[0])

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
            batch = self._format_decordTorchVideo_for_opticalFlow(batch, mask=self.mask)

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

    def _format_decordTorchVideo_for_opticalFlow(self, vid, mask=None):
        """
        Format a decord video array properly for optical flow.
        
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


class Dataset_VideoReader(torch.utils.data.Dataset):
    """
    demo:
    ds = Basic_dataset(X, device='cuda:0')
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    """
    def __init__(self, 
                 video, 
                #  device='cpu',
                #  dtype=torch.float32
    ):
        """
        Make a basic dataset.
        RH 2021

        Args:
            X (torch.Tensor or np.array):
                Data to make dataset from.
            device (str):
                Device to use.
            dtype (torch.dtype):
                Data type to use.
        """
        self.video = video
        self.buffer_length = 1000
        self.X = video[:self.buffer_length]
        self.idx_available = torch.arange(len(self.X))
        
    def __len__(self):
        return len(self.video)
    
    def __getitem__(self, idx):
        """
        Returns a single sample.

        Args:
            idx (int):
                Index of sample to return.
        """
        idx_in_buffer = torch.where(self.idx_available == idx)[0]
        if idx_in_buffer > self.buffer_length/2:
            self.X = torch.cat((self.X[idx_in_buffer:], self.video[self.idx_available[-1]+1:self.idx_available[-1]+1+self.buffer_length-idx_in_buffer+2]))
            self.idx_available = torch.cat((self.idx_available[idx_in_buffer:], torch.arange(int(self.idx_available[-1]+1), int(self.idx_available[-1]+1+self.buffer_length-idx_in_buffer+2))))
            return self.X[0]

        return self.X[idx_in_buffer]



def visualize_image_with_points(
    image,
    points=None,

    points_colors=(255, 255, 255),
    points_sizes=1,
    
    text=None,
    text_positions=None,
    text_color='white',
    text_size=1,
    text_thickness=1,

    display=False,
    handle_cv2Imshow='FaceRhythmPointVisualizer',
    writer_cv2=None,

    in_place=False,
    error_checking=False,
):
    """
    Visualize an image with points and text.
    Be careful to follow input formats as little error checking is done
     to save time.

    Args:
        image (np.ndarray, uint8):
            3D array of integers, where each element is a 
             pixel value. Last dimension should be channels.
        points (np.ndarray, int):
            3D array: First dimension is batch of points to plot.
                Each batch can have different colors and sizes.
                Second dimension is point number, and third dimension
                is point coordinates. Order (y,x).
        points_colors (tuple of int or list of tuple of int):
            Used as argument for cv2.circle.
            If tuple: All points will be this color.
                Elements of tuple should be 3 integers between 0 and 255.
            If list: Each element is a color for a batch of points.
                Length of list must match the first dimension of points.
                points must be 3D array.
        points_sizes (int or list):
            Used as argument for cv2.circle.
            If int: All points will be this size.
            If list: Each element is a size for a batch of points.
                Length of list must match the first dimension of points.
                points must be 3D array.
        text (str or list):
            Used as argument for cv2.putText.
            If None: No text will be plotted.
            If str: All text will be this string.
            If list: Each element is a string for a batch of text.
                text_positions must be 3D array.
        text_positions (np.ndarray, np.float32):
            Must be specified if text is not None.
            2D array: Each row is a text position. Order (y,x).
        text_color (str or list):
            Used as argument for cv2.putText.
            If str: All text will be this color.
            If list: Each element is a color for a different text.
                Length of list must match the length of text.
        text_size (int or list):
            Used as argument for cv2.putText.
            If int: All text will be this size.
            If list: Each element is a size for a different text.
                Length of list must match the length of text.
        text_thickness (int or list):
            Used as argument for cv2.putText.
            If int: All text will be this thickness.
            If list: Each element is a thickness for a different text.
                Length of list must match the length of text.
        display (bool):
            If True: Display image using cv2.imshow.
        handle_cv2Imshow (str):
            Used as argument for cv2.imshow.
            Can be used to close window later.
        writer_cv2 (cv2.VideoWriter):
            If not None: Write image to video using writer_cv2.write.
        in_place (bool):
            If True: Modify image in place. Otherwise, return a copy.
        error_checking (bool):
            If True: Perform error checking.

    Returns:
        image (np.ndarray, uint8):
            A 3D array of integers, where each element is a 
             pixel value.
    """
    ## Check inputs
    if error_checking:
        ## Check image
        assert isinstance(image, np.ndarray), 'image must be a numpy array.'
        assert image.dtype == np.uint8, 'image must be a numpy array of uint8.'
        assert image.ndim == 3, 'image must be a 3D array.'
        assert image.shape[-1] == 3, 'image must have 3 channels.'

        ## Check points
        if points is not None:
            assert isinstance(points, np.ndarray), 'points must be a numpy array.'
            assert points.dtype == int, 'points must be a numpy array of int.'
            assert points.ndim == 3, 'points must be a 3D array.'
            assert points.shape[-1] == 2, 'points must have 2 coordinates.'
            assert np.all(points >= 0), 'points must be non-negative.'
            assert np.all(points[:, :, 0] < image.shape[0]), 'points must be within image.'
            assert np.all(points[:, :, 1] < image.shape[1]), 'points must be within image.'

        ## Check points_colors
        if points_colors is not None:
            if isinstance(points_colors, tuple):
                assert len(points_colors) == 3, 'points_colors must be a tuple of 3 integers.'
                assert all([isinstance(c, int) for c in points_colors]), 'points_colors must be a tuple of 3 integers.'
                assert all([c >= 0 and c <= 255 for c in points_colors]), 'points_colors must be a tuple of 3 integers between 0 and 255.'
            elif isinstance(points_colors, list):
                assert all([isinstance(c, tuple) for c in points_colors]), 'points_colors must be a list of tuples.'
                assert all([len(c) == 3 for c in points_colors]), 'points_colors must be a list of tuples of 3 integers.'
                assert all([all([isinstance(c_, int) for c_ in c]) for c in points_colors]), 'points_colors must be a list of tuples of 3 integers.'
                assert all([all([c_ >= 0 and c_ <= 255 for c_ in c]) for c in points_colors]), 'points_colors must be a list of tuples of 3 integers between 0 and 255.'

        ## Check points_sizes
        assert isinstance(points_sizes, int) or isinstance(points_sizes, list), 'points_sizes must be an integer or a list.'
        if isinstance(points_sizes, list):
            assert len(points_sizes) == points.shape[0], 'Length of points_sizes must match the first dimension of points.'
            assert all([isinstance(size, int) for size in points_sizes]), 'All elements of points_sizes must be integers.'

        ## Check text
        if text is not None:
            assert isinstance(text, str) or isinstance(text, list), 'text must be a string or a list.'
            if isinstance(text, list):
                assert len(text) == text_positions.shape[0], 'Length of text must match the first dimension of text_positions.'
                assert all([isinstance(string, str) for string in text]), 'All elements of text must be strings.'

        ## Check text_positions
        if text_positions is not None:
            assert isinstance(text_positions, np.ndarray), 'text_positions must be a numpy array.'
            assert text_positions.dtype == np.float32, 'text_positions must be a numpy array of np.float32.'
            assert text_positions.ndim == 2, 'text_positions must be a 2D array.'
            assert text_positions.shape[-1] == 2, 'text_positions must have 2 coordinates (y,x).'

        ## Check text_color
        assert isinstance(text_color, str) or isinstance(text_color, list), 'text_color must be a string or a list.'
        if isinstance(text_color, list):
            assert len(text_color) == len(text), 'Length of text_color must match the length of text.'
            assert all([isinstance(color, str) for color in text_color]), 'All elements of text_color must be strings.'

        ## Check text_size
        assert isinstance(text_size, int) or isinstance(text_size, list), 'text_size must be an integer or a list.'
        if isinstance(text_size, list):
            assert len(text_size) == len(text), 'Length of text_size must match the length of text.'
            assert all([isinstance(size, int) for size in text_size]), 'All elements of text_size must be integers.'

        ## Check text_thickness
        assert isinstance(text_thickness, int) or isinstance(text_thickness, list), 'text_thickness must be an integer or a list.'
        if isinstance(text_thickness, list):
            assert len(text_thickness) == len(text), 'Length of text_thickness must match the length of text.'
            assert all([isinstance(thickness, int) for thickness in text_thickness]), 'All elements of text_thickness must be integers.'


    ## Copy image
    if not in_place:
        image = image.copy()

    ## Convert point colors to list of BGR tuples
    if isinstance(points_colors, tuple) and points is not None:
        points_colors = [points_colors] * points.shape[0]

    ## Convert points_sizes to list
    if isinstance(points_sizes, int) and points is not None:
        points_sizes = [points_sizes] * points.shape[0]

    ## Convert text to list
    if isinstance(text, str):
        text = [text]

    ## Convert text_color to list
    if isinstance(text_color, str):
        text_color = [text_color]

    ## Convert text_size to list
    if isinstance(text_size, int):
        text_size = [text_size]

    ## Convert text_thickness to list
    if isinstance(text_thickness, int):
        text_thickness = [text_thickness]

    ## Plot points
    if points is not None:
        ## Plot points
        for i_batch in range(points.shape[0]):
            for i_point in range(points.shape[1]):
                cv2.circle(
                    img=image,
                    center=tuple(points[i_batch][i_point][::-1]),
                    radius=points_sizes[i_batch],
                    color=points_colors[i_batch],
                    thickness=-1,
                )

    ## Plot text
    if text is not None:
        ## Plot text
        for i in range(len(text)):
            cv2.putText(
                img=image,
                text=text[i],
                org=tuple(text_positions[i, :][::-1]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=text_size[i],
                color=text_color[i],
                thickness=text_thickness[i],
            )

    ## Display image
    if display:
        cv2.imshow(handle_cv2Imshow, image)
        cv2.waitKey(1)

    ## Write image
    if writer_cv2 is not None:
        writer_cv2.write(image)

    return image
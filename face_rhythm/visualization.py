"""Frame and video visualization: overlay points/text on images and write videos.

``FrameVisualizer`` wraps OpenCV's video writer and draw primitives; helper
functions play back buffered readers with overlaid trajectories and produce
interactive image stacks for Jupyter contexts.
"""

from typing import Union, Tuple, List
from pathlib import Path
import copy
import gc

import numpy as np
import cv2
import torch
# import scipy.sparse
from tqdm.auto import tqdm

from .helpers import BufferedVideoReader

class FrameVisualizer:
    """
    Wraps OpenCV draw primitives and an optional ``cv2.VideoWriter`` to overlay
    points and text on single frames, optionally displaying them via
    ``cv2.imshow`` and/or writing them to a video file. RH 2022

    Args:
        display (bool):
            If ``True``, display each frame using ``cv2.imshow``.
            (Default is ``False``)
        handle_cv2Imshow (str):
            Window name passed to ``cv2.imshow``. Used to close the window
            later. (Default is ``'FaceRhythmPointVisualizer'``)
        path_save (Optional[str]):
            If not ``None``, frames are written to this video file path. Use
            an ``.avi`` extension (e.g. ``'directory/filename.avi'``).
            (Default is ``None``)
        frame_height_width (Tuple[int, int]):
            Height and width of the displayed and/or saved video.
            (Default is ``(480, 640)``)
        frame_rate (Optional[int]):
            Frame rate of the displayed and/or saved video. If ``None``,
            playback runs at top speed and saved videos default to 60 fps.
            (Default is ``None``)
        fourcc (str):
            Four-character codec passed to ``cv2.VideoWriter``.
            (Default is ``'MJPG'``)
        error_checking (bool):
            If ``True``, perform input validation in
            ``visualize_image_with_points``. (Default is ``True``)
        verbose (int):
            Verbosity level. \n
            * ``0``: No messages.
            * ``1``: Warnings.
            * ``2``: Info. \n
            (Default is ``1``)
        point_sizes (Optional[Union[int, List[int]]]):
            Optional override applied in every call to
            ``visualize_image_with_points``. Passed to ``cv2.circle``. If an
            int, all points use this radius; if a list, each element is the
            radius for one batch of points. (Default is ``None``)
        points_colors (Optional[Union[Tuple[int, int, int], List]]):
            Optional override applied in every call to
            ``visualize_image_with_points``. Passed to ``cv2.circle``. If a
            tuple of 3 ints in ``[0, 255]``, all points use this color; if a
            list, each element is a color or a per-point color array of
            shape *(N, 3)* for one batch. (Default is ``None``)
        alpha (Optional[float]):
            Optional override applied in every call to
            ``visualize_image_with_points``. Transparency of the overlaid
            points; values other than ``1`` are slow. (Default is ``None``)
        text (Optional[Union[str, List[str]]]):
            Optional override applied in every call to
            ``visualize_image_with_points``. If ``None``, no text is drawn;
            if a string, the same string is drawn at every position; if a
            list, each element is drawn at the corresponding row of
            ``text_positions``. (Default is ``None``)
        text_positions (Optional[np.ndarray]):
            Optional override applied in every call to
            ``visualize_image_with_points``. Must be specified if ``text`` is
            not ``None``. shape: *(n_text, 2)*, order *(x, y)*.
            (Default is ``None``)
        text_color (Optional[Union[str, List[str]]]):
            Optional override applied in every call to
            ``visualize_image_with_points``. Passed to ``cv2.putText``. If a
            string, the same color is used for all text; if a list, each
            element is the color for one text item. (Default is ``None``)
        text_size (Optional[Union[int, List[int]]]):
            Optional override applied in every call to
            ``visualize_image_with_points``. Passed to ``cv2.putText``. If
            an int, the same scale is used for all text; if a list, each
            element is the scale for one text item. (Default is ``None``)
        text_thickness (Optional[Union[int, List[int]]]):
            Optional override applied in every call to
            ``visualize_image_with_points``. Passed to ``cv2.putText``. If
            an int, the same thickness is used for all text; if a list, each
            element is the thickness for one text item. (Default is ``None``)

    Attributes:
        display (bool):
            Whether ``cv2.imshow`` is called on each visualized frame.
        error_checking (bool):
            Whether input validation runs in ``visualize_image_with_points``.
        handle_cv2Imshow (str):
            Window name used by ``cv2.imshow``.
        path_save (Optional[str]):
            Resolved absolute path to the output video file, or ``None``.
        frame_height_width (Tuple[int, int]):
            Height and width of frames written to the video file.
        frame_rate (Optional[int]):
            Frame rate used for both display timing and the video writer.
        fourcc (str):
            Four-character codec used by ``cv2.VideoWriter``.
        video_writer (Optional[object]):
            Underlying ``cv2.VideoWriter`` instance, or ``None`` if
            ``path_save`` is not set.
    """
    def __init__(
        self,
        ## can be tuple of int or list of int
        display=False,
        
        handle_cv2Imshow='FaceRhythmPointVisualizer',
        path_save=None,
        frame_height_width=(480, 640),
        frame_rate=None,
        fourcc="MJPG",
        error_checking=True,
        verbose: int=1,

        point_sizes=None,
        points_colors=None,
        alpha=None,
        text=None,
        text_positions=None,
        text_color=None,
        text_size=None,
        text_thickness=None,

    ):
        """Initializes the visualizer and opens the underlying video writer when ``path_save`` is provided."""
        ## Store arguments
        self.point_sizes = point_sizes if point_sizes is not None else None
        self.points_colors = points_colors if points_colors is not None else None
        self.alpha = alpha if alpha is not None else None
        self.text = text if text is not None else None
        self.text_positions = text_positions if text_positions is not None else None
        self.text_color = text_color if text_color is not None else None
        self.text_size = text_size if text_size is not None else None
        self.text_thickness = text_thickness if text_thickness is not None else None

        ## Set variables
        self.display = bool(display)
        self.error_checking = bool(error_checking)
        self.handle_cv2Imshow = str(handle_cv2Imshow)
        self.path_save = str(Path(path_save).resolve()) if path_save is not None else None
        self.frame_height_width = tuple(frame_height_width)
        self.frame_rate = int(frame_rate) if frame_rate is not None else None
        self.fourcc = str(fourcc)
        self._verbose = int(verbose)

        ## Make video writer
        if self.path_save is not None:
            print(f'Initializing video writer with frame_rate={self.frame_rate}, fourcc={self.fourcc}, frame_height_width={self.frame_height_width}, path_save={self.path_save}') if self._verbose > 1 else None
            self.video_writer = cv2.VideoWriter(
                    self.path_save,
                    cv2.VideoWriter_fourcc(*self.fourcc),
                    frame_rate,
                    frame_height_width[::-1],
                ) 
        else:
            self.video_writer = None

    def visualize_image_with_points(
        self,
        image,
        points=None,

        point_sizes=None,

        points_colors=(0, 255, 255),
        alpha=None,
        
        text=None,
        text_positions=None,
        text_color='white',
        text_size=1,
        text_thickness=1,
    ):
        """
        Draws points and text onto a single image and optionally displays
        and/or writes the result. Input validation is intentionally minimal
        for performance, so the caller must follow the documented formats.

        Args:
            image (np.ndarray):
                Image to draw on. shape: *(H, W, 3)*, dtype: *uint8*. The
                last dimension is channels.
            points (Optional[Union[np.ndarray, List[np.ndarray]]]):
                Points to overlay. If a single ``np.ndarray`` of shape
                *(n_points, 2)* and integer dtype, it is treated as one
                batch and clamped to the image bounds. If a list, each
                element is one batch of shape *(n_points, 2)* and dtype
                *int*; column order is *(x, y)*. (Default is ``None``)
            point_sizes (Optional[Union[int, List[int]]]):
                Radius passed to ``cv2.circle``. If an int, every point
                uses this size; if a list, each element is the size for one
                batch of ``points``. (Default is ``None``)
            points_colors (Union[Tuple[int, int, int], List]):
                Color passed to ``cv2.circle``. If a tuple of 3 ints in
                ``[0, 255]``, every point uses this color; if a list, each
                element is either a 3-tuple for one batch or an
                ``np.ndarray`` of shape *(n_points, 3)* with per-point
                colors in ``[0, 255]``. (Default is ``(0, 255, 255)``)
            alpha (Optional[float]):
                Transparency of the overlaid points; values other than
                ``1`` are slow. (Default is ``None``)
            text (Optional[Union[str, List[str]]]):
                Text passed to ``cv2.putText``. If ``None``, no text is
                drawn; if a string, the same string is drawn at every row
                of ``text_positions``; if a list, each element is drawn at
                the matching row. (Default is ``None``)
            text_positions (Optional[np.ndarray]):
                Positions for each text item. Required if ``text`` is not
                ``None``. shape: *(n_text, 2)*, order *(x, y)*.
                (Default is ``None``)
            text_color (Union[str, List[str]]):
                Color passed to ``cv2.putText``. If a string, all text uses
                this color; if a list, each element is the color for one
                text item. (Default is ``'white'``)
            text_size (Union[int, List[int]]):
                Font scale passed to ``cv2.putText``. If an int, all text
                uses this scale; if a list, each element is the scale for
                one text item. (Default is ``1``)
            text_thickness (Union[int, List[int]]):
                Stroke thickness passed to ``cv2.putText``. If an int, all
                text uses this thickness; if a list, each element is the
                thickness for one text item. (Default is ``1``)

        Returns:
            (np.ndarray):
                image_out (np.ndarray):
                    Copy of ``image`` with points and text drawn on top.
                    shape: *(H, W, 3)*, dtype: *uint8*.
        """
        ## Get arguments from self if not None
        point_sizes = self.point_sizes if self.point_sizes is not None else point_sizes
        points_colors = self.points_colors if self.points_colors is not None else points_colors
        alpha = alpha if alpha is not None else self.alpha
        text = self.text if self.text is not None else text
        text_positions = self.text_positions if self.text_positions is not None else text_positions
        text_color = self.text_color if self.text_color is not None else text_color
        text_size = self.text_size if self.text_size is not None else text_size
        text_thickness = self.text_thickness if self.text_thickness is not None else text_thickness

        ## Check inputs
        if self.error_checking:
            ## Check image
            assert isinstance(image, np.ndarray), 'image must be a numpy array.'
            assert image.dtype == np.uint8, 'image must be a numpy array of uint8.'
            assert image.ndim == 3, 'image must be a 3D array.'
            assert image.shape[-1] == 3, 'image must have 3 channels.'
            if self.frame_height_width is not None:
                assert image.shape[:2] == self.frame_height_width, f'image must have shape {self.frame_height_width}, specified in self.frame_height_width, but has shape {image.shape[:2]}.'

            ## Check points
            if points is not None:
                if isinstance(points, np.ndarray):
                    points = points.astype(np.intp)
                    ## Clamp points to image
                    points[:,0] = np.clip(points[:,0], 0, image.shape[1])
                    points[:,1] = np.clip(points[:,1], 0, image.shape[0])
                    points = [points]
                assert isinstance(points, list), 'points must be a list.'
                assert len(points) > 0, 'points must have at least one element.'
                assert isinstance(points[0], np.ndarray), 'points must be a list of numpy arrays.'
                assert points[0].dtype == np.intp, 'points must be a list of numpy arrays of int.'
                assert points[0].ndim == 2, 'points must be a list of 2D numpy arrays.'
                assert points[0].shape[1] == 2, 'points must be a list of 2D numpy arrays with 2 columns.'
                # ## all points must be non-negative
                # assert np.all([np.all(points[i] >= 0) for i in range(len(points))]), f'points must be non-negative. {points[0][points[0]<0], points[1][points[1]<0]}'
                # ## all points must be within image
                # assert np.all([np.all(points[i][:,0] < image.shape[1]) for i in range(len(points))]), 'points must be within image.'
                # assert np.all([np.all(points[i][:,1] < image.shape[0]) for i in range(len(points))]), 'points must be within image.'

            ## Check points_sizes
            if point_sizes is not None:
                assert isinstance(point_sizes, (int, list)), 'points_sizes must be an integer or a list.'
                if isinstance(point_sizes, list):
                    assert len(point_sizes) == points.shape[0], 'Length of points_sizes must match the first dimension of points.'
                    assert all([isinstance(size, int) for size in point_sizes]), 'All elements of points_sizes must be integers.'

            ## Check points_colors
            if points_colors is not None:
                if isinstance(points_colors, tuple):
                    assert len(points_colors) == 3, 'points_colors must be a tuple of 3 integers.'
                    assert all([isinstance(c, int) for c in points_colors]), 'points_colors must be a tuple of 3 integers.'
                    assert all([c >= 0 and c <= 255 for c in points_colors]), 'points_colors must be a tuple of 3 integers between 0 and 255.'
                elif isinstance(points_colors, list):
                    if isinstance(points_colors[0], tuple):
                        assert all([isinstance(c, tuple) for c in points_colors]), 'points_colors must be a list of tuples.'
                        assert all([len(c) == 3 for c in points_colors]), 'points_colors must be a list of tuples of 3 integers.'
                        assert all([all([isinstance(c_, int) for c_ in c]) for c in points_colors]), 'points_colors must be a list of tuples of 3 integers.'
                        assert all([all([c_ >= 0 and c_ <= 255 for c_ in c]) for c in points_colors]), 'points_colors must be a list of tuples of 3 integers between 0 and 255.'
                    elif isinstance(points_colors[0], np.ndarray):
                        assert all([isinstance(c, np.ndarray) for c in points_colors]), 'points_colors must be a list of numpy arrays.'
                        assert all([c.dtype == np.intp for c in points_colors]), 'points_colors must be a list of numpy arrays of int.'
                        assert all([c.ndim == 2 for c in points_colors]), 'points_colors must be a list of 2D numpy arrays.'
                        assert all([c.shape[1] == 3 for c in points_colors]), 'points_colors must be a list of 2D numpy arrays with 3 columns.'
                        assert all([np.all(c >= 0) for c in points_colors]), 'points_colors must be a list of 2D numpy arrays with values between 0 and 255.'
                        assert all([np.all(c <= 255) for c in points_colors]), 'points_colors must be a list of 2D numpy arrays with values between 0 and 255.'
                    else:
                        raise ValueError('points_colors must be a list of tuples or a list of numpy arrays.')

            ## Check text
            if text is not None:
                assert isinstance(text, str) or isinstance(text, list), 'text must be a string or a list.'
                if isinstance(text, list):
                    assert len(text) == text_positions.shape[0], 'Length of text must match the first dimension of text_positions.'
                    assert all([isinstance(string, str) for string in text]), 'All elements of text must be strings.'

            ## Check text_positions
            if text_positions is not None:
                assert isinstance(text_positions, np.ndarray), 'text_positions must be a numpy array.'
                # assert text_positions.dtype == np.float32, 'text_positions must be a numpy array of np.float32.'
                assert text_positions.ndim == 2, 'text_positions must be a 2D array.'
                assert text_positions.shape[-1] == 2, 'text_positions must have 2 coordinates (x,y).'

            ## Check text_color
            assert isinstance(text_color, str) or isinstance(text_color, list), 'text_color must be a string or a list.'
            if isinstance(text_color, list):
                assert len(text_color) == len(text), 'Length of text_color must match the length of text.'
                # assert all([isinstance(color, str) for color in text_color]), 'All elements of text_color must be strings.'

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

        ## Make a copy of image
        image_out = copy.copy(image)

        ## Convert point colors to list of BGR tuples
        if isinstance(points_colors, tuple) and points is not None:
            points_colors = [points_colors] * len(points)

        ## Convert text to list
        if isinstance(text, str):
            text = [text]

        ## Convert points_sizes to list
        point_sizes = int(2) if point_sizes is None else point_sizes
        if isinstance(point_sizes, int) and points is not None:
            point_sizes = [point_sizes] * len(points)

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
            for i_batch, (points_batch, size_batch, colors_batch) in enumerate(zip(points, point_sizes, points_colors)):
                if isinstance(colors_batch, (list, tuple)):
                    for ii,points in enumerate(points_batch):
                        cv2.circle(
                            img=image_out,
                            center=tuple(points),
                            radius=int(size_batch),
                            color=colors_batch,
                            thickness=-1,
                        )
                else:
                    for ii,(points, color) in enumerate(zip(points_batch, colors_batch)):
                        cv2.circle(
                            img=image_out,
                            center=tuple(points),
                            radius=int(size_batch),
                            color=color.tolist(),
                            thickness=-1,
                        )
            ## Do weighted addition
            image_out = cv2.addWeighted(image_out, alpha, image, (1-alpha), 0.0)


        ## Plot text
        if text is not None:
            ## Plot text
            for i in range(len(text)):
                cv2.putText(
                    img=image_out,
                    text=text[i],
                    org=tuple(text_positions[i, :]),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=text_size[i],
                    color=text_color[i],
                    thickness=text_thickness[i],
                )


        ## Display image_out
        if self.display:
            cv2.imshow(self.handle_cv2Imshow, image_out)
            cv2.waitKey(1) if self.frame_rate is None else cv2.waitKey(int(1000/self.frame_rate))

        ## Write image_out
        if self.video_writer is not None:
            self.video_writer.write(image_out)

        return image_out

    def close(self):
        """Closes the OpenCV display window and releases the video writer if either is active."""
        if self.video_writer is not None:
            cv2.destroyWindow(self.handle_cv2Imshow)
            try:
                self.video_writer.release()
            except Exception:
                pass

    def __call__(self, *args, **kwds):
        """Forwards positional and keyword arguments to ``visualize_image_with_points``."""
        self.visualize_image_with_points(*args, **kwds)
    def __del__(self):
        self.close()

    def __repr__(self):
        return f'FrameVisualizer(handle_cv2Imshow={self.handle_cv2Imshow}, display={self.display}, video_writer={self.video_writer}, path_video={self.path_save}, frame_rate={self.frame_rate}, frame_height_width={self.frame_height_width})'


def play_video_with_points(
        bufferedVideoReader,
        frameVisualizer=None,
        points=None,
        idx_frames=None,
    ):
        """
        Plays a video with optional point overlays and optionally writes it
        to disk via the supplied ``FrameVisualizer``. RH 2022

        Args:
            bufferedVideoReader (BufferedVideoReader):
                Source of frames, created with
                ``fr.helpers.BufferedVideoReader``.
            frameVisualizer (FrameVisualizer):
                Visualizer that draws and optionally saves each frame,
                created with ``fr.visualization.FrameVisualizer``. Required
                in practice despite the default. (Default is ``None``)
            points (Optional[np.ndarray]):
                Points to overlay on the video. shape:
                *(num_frames, num_points, 2)*. (Default is ``None``)
            idx_frames (Optional[np.ndarray]):
                Indices of frames to play. If ``None``, all frames in the
                reader are played. (Default is ``None``)
        """
        ## Check arguments
        print(type(bufferedVideoReader)) if frameVisualizer._verbose > 1 else None
        assert isinstance(bufferedVideoReader, BufferedVideoReader), 'bufferedVideoReader must be a BufferedVideoReader object.'
        assert isinstance(frameVisualizer, FrameVisualizer), 'frameVisualizer must be a FrameVisualizer object.'

        ## Prep idx_frames
        idx_frames = np.arange(bufferedVideoReader.num_frames_total) if idx_frames is None else idx_frames
        if idx_frames.max() > bufferedVideoReader.num_frames_total:
            idx_frames = idx_frames[idx_frames < bufferedVideoReader.num_frames_total]
            print(f'Warning: idx_frames contained frames that were out of bounds. Truncating to {idx_frames.max()}.') if frameVisualizer._verbose > 0 else None
        ## Prep points
        points_int = points.astype(int) if points is not None else None

        ## Loop through frames
        ### Set buffered video reader to first frame
        bufferedVideoReader.set_iterator_frame_idx(int(idx_frames[0]))
        ### Iterate through frames
        ### Use context manager to close frameVisualizer
        class CM:
            def __init__(self, frameVisualizer):
                self.frameVisualizer = frameVisualizer
            def __enter__(self):
                return self.frameVisualizer
            def __exit__(self, exc_type, exc_value, traceback):
                self.frameVisualizer.close()
                try:
                    cv2.destroyWindow(self.frameVisualizer.handle_cv2Imshow)
                except cv2.error:
                    pass  ## Headless (no GUI backend): nothing to destroy
                gc.collect()
        with CM(frameVisualizer) as f:
            for idx_frame in tqdm(idx_frames):
                frame = bufferedVideoReader[idx_frame][0]
                frame = frame.numpy() if isinstance(frame, torch.Tensor) else frame
                p = points_int[idx_frame] if points_int is not None else None
                f.visualize_image_with_points(
                    image=frame,
                    points=[p],
                )
            f.close()


# def display_toggle_image_stack(images, clim=None, **kwargs):
#     """
#     Display a stack of images using a slider.
#     REQUIRES use of Jupyter Notebook.
#     RH 2022

#     Args:
#         images (np.ndarray):
#             Stack of images.
#             Shape: (num_frames, height, width)
#             Optionally, shape: (num_frames, height, width, num_channels)
#         clim (tuple, optional):
#             Color limits.
#         kwargs (dict, optional):
#             Keyword arguments to pass to imshow.
#     """
#     import matplotlib.pyplot as plt
#     from ipywidgets import interact, widgets

#     fig = plt.figure()
#     ax = fig.add_subplot(1, 1, 1)
#     imshow_FOV = ax.imshow(
#         images[0],
# #         vmax=clim[1]
#         **kwargs
#     )

#     def update(i_frame = 0):
#         fig.canvas.draw_idle()
#         imshow_FOV.set_data(images[i_frame])
#         imshow_FOV.set_clim(clim)


#     interact(update, i_frame=widgets.IntSlider(min=0, max=len(images)-1, step=1, value=0));

def display_toggle_image_stack(images, image_size=None, clim=None, interpolation='nearest'):
    """
    Renders an HTML image slider in a Jupyter notebook to scrub through a
    stack of images. RH 2023

    Args:
        images (List[Union[np.ndarray, torch.Tensor]]):
            Images to display, each as a 2D or 3D ``np.ndarray`` or
            ``torch.Tensor``. All images must share an interpretation
            compatible with PIL ``fromarray``.
        image_size (Optional[Union[Tuple[int, int], float]]):
            Output size per image. \n
            * ``Tuple[int, int]``: explicit ``(width, height)`` applied to
              every image.
            * ``float``: scale factor applied to each image's native shape.
            * ``None``: images are displayed at their native size. \n
            (Default is ``None``)
        clim (Optional[Tuple[float, float]]):
            ``(min, max)`` intensity bounds used to scale pixel values to
            ``[0, 255]``. If ``None``, the per-image min and max are used.
            (Default is ``None``)
        interpolation (str):
            Resampling method used when resizing. One of \n
            * ``'nearest'``
            * ``'box'``
            * ``'bilinear'``
            * ``'hamming'``
            * ``'bicubic'``
            * ``'lanczos'`` \n
            Mapped to the matching ``PIL.Image.Resampling.*`` constant.
            (Default is ``'nearest'``)
    """
    from IPython.display import display, HTML
    import numpy as np
    import base64
    from PIL import Image
    from io import BytesIO
    import torch
    import datetime
    import hashlib
    import sys
    
    def normalize_image(image, clim=None):
        """Normalize the input image using the min-max scaling method. Optionally, use the given clim values for scaling."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if clim is None:
            clim = (np.min(image), np.max(image))

        norm_image = (image - clim[0]) / (clim[1] - clim[0])
        norm_image = np.clip(norm_image, 0, 1)
        return (norm_image * 255).astype(np.uint8)
    def resize_image(image, size_new, interpolation):
        """Resize the given image to the specified new size using the specified interpolation method."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        pil_image = Image.fromarray(image.astype(np.uint8))
        resized_image = pil_image.resize(size_new, resample=interpolation)
        return np.array(resized_image)
    def numpy_to_base64(numpy_array):
        """Convert a numpy array to a base64 encoded string."""
        img = Image.fromarray(numpy_array.astype('uint8'))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("ascii")
    def process_image(image, size=None):
        """Normalize, resize, and convert image to base64."""
        # Normalize image
        norm_image = normalize_image(image, clim)

        # Resize image if requested
        if size is not None:
            norm_image = resize_image(norm_image, size, interpolation_method)

        # Convert image to base64
        return numpy_to_base64(norm_image)


    # Check if being called from a Jupyter notebook
    if 'ipykernel' not in sys.modules:
        raise RuntimeError("This function must be called from a Jupyter notebook.")

    # Create a dictionary to map interpolation string inputs to Image objects
    interpolation_methods = {
        'nearest': Image.Resampling.NEAREST,
        'box': Image.Resampling.BOX,
        'bilinear': Image.Resampling.BILINEAR,
        'hamming': Image.Resampling.HAMMING,
        'bicubic': Image.Resampling.BICUBIC,
        'lanczos': Image.Resampling.LANCZOS,
    }

    # Check if provided interpolation method is valid
    if interpolation not in interpolation_methods:
        raise ValueError("Invalid interpolation method. Choose from 'nearest', 'box', 'bilinear', 'hamming', 'bicubic', or 'lanczos'.")

    # Get the actual Image object for the specified interpolation method
    interpolation_method = interpolation_methods[interpolation]

    # Generate a unique identifier for the slider
    slider_id = hashlib.sha256(str(datetime.datetime.now()).encode()).hexdigest()

    # Get the image sizes for processing and display
    if image_size is not None:
        image_sizes = [tuple((np.array(img.shape[:2]) * image_size).astype(int)) for img in images] if isinstance(image_size, (int, float)) else image_size
        size_frame = image_sizes[0]
    else:
        image_sizes = [None] * len(images)
        size_frame = images[0].shape[:2]

    # Process all images in the input list
    base64_images = [process_image(img, size=sz) for img,sz in zip(images, image_sizes)]

    # Generate the HTML code for the slider
            # <img id="displayedImage_{slider_id}" src="data:image/png;base64,{base64_images[0]}" ;">

    html_code = f"""
    <div>
        <input type="range" id="imageSlider_{slider_id}" min="0" max="{len(base64_images) - 1}" value="0">
        <img id="displayedImage_{slider_id}" src="data:image/png;base64,{base64_images[0]}" style="width: {size_frame[1]}px; height: {size_frame[0]}px;">
        <span id="imageNumber_{slider_id}">Image 0/{len(base64_images) - 1}</span>
    </div>

    <script>
        (function() {{
            let base64_images = {base64_images};
            let current_image = 0;
    
            function updateImage() {{
                let slider = document.getElementById("imageSlider_{slider_id}");
                current_image = parseInt(slider.value);
                let displayedImage = document.getElementById("displayedImage_{slider_id}");
                displayedImage.src = "data:image/png;base64," + base64_images[current_image];
                let imageNumber = document.getElementById("imageNumber_{slider_id}");
                imageNumber.innerHTML = "Image " + current_image + "/{len(base64_images) - 1}";
            }}
            
            document.getElementById("imageSlider_{slider_id}").addEventListener("input", updateImage);
        }})();
    </script>
    """

    display(HTML(html_code))


def complex_colormap(
    mags: np.ndarray, 
    angles: np.ndarray, 
    normalize_mags: bool = True,
    color_sin: Tuple[int, int, int] = (255, 0, 0),
    color_cos: Tuple[int, int, int] = (0, 0, 255),
) -> np.ndarray:
    """
    Generates an RGB colormap for complex values, where hue tracks the
    angle and brightness tracks the magnitude.

    Args:
        mags (np.ndarray):
            Magnitudes of the complex values. Must broadcast against
            ``angles``.
        angles (np.ndarray):
            Angles in radians. Must share shape with ``mags``.
        normalize_mags (bool):
            If ``True``, apply min-max normalization to ``mags`` before
            scaling brightness. (Default is ``True``)
        color_sin (Tuple[int, int, int]):
            RGB color contributed in proportion to ``sin(angles)``.
            (Default is ``(255, 0, 0)``)
        color_cos (Tuple[int, int, int]):
            RGB color contributed in proportion to ``cos(angles)``.
            (Default is ``(0, 0, 255)``)

    Returns:
        (np.ndarray):
            rgb (np.ndarray):
                RGB values per element. shape: *(mags.size, 3)*.
    """
    assert mags.shape == angles.shape, "The shapes of mags and angles must be the same."

    ## Normalize the magnitudes to the range [0, 1]
    if normalize_mags:
        mags_norm = (mags - np.min(mags)) / (np.max(mags) - np.min(mags))
    mags_norm = np.clip(mags_norm, 0, 1)

    ## Initialize the RGB array
    rgb = np.zeros((mags.size, 3))

    ## Apply the colors according to the angles
    rgb += np.array(color_sin)[None, :] * np.sin(angles)[:, None]
    rgb += np.array(color_cos)[None, :] * np.cos(angles)[:, None]

    ## Apply the brightness according to the normalized magnitudes
    rgb *= mags_norm[:, None]

    return rgb

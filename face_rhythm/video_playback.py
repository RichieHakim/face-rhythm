from typing import Union

import numpy as np
import cv2
import scipy.sparse

from .helpers import BufferedVideoReader, Toeplitz_convolution2d

class FrameVisualizer:
    """
    Class for playing back a video.
    Allows for playing back one frame at a time, or playing back
     an array of frames.
    """
    def __init__(
        self,
        ## can be tuple of int or list of int
        image_height_width: Union[tuple, list],
        points_sizes: Union[float, int, list, list]=[3.0],
        verbose: int=1,
    ):
        """
        Initialize the VideoPlayback object.
        This class wraps the primary function which is:
         self.visualize_image_with_points. It is used to visualize
         single frame inputs of images and overlayed points.
        The reason why this is a class is to initialize the
         Toeplitz convolution object which is used to control the
         size of the points being visualized.

        Args:
            image_height_width (tuple or list):
                The height and width of the images to be played back.
            points_sizes (float or list of floats):
                Diameter of the points to draw on the video.
                If a list is passed, then each element should be the
                 diameter for that batch of points.
                See self.visualize_image_with_points for details on 
                 batches.
        """
        ## Assertions
        assert isinstance(image_height_width, (tuple(int), list(int))), "FR ERROR: image_height_width must be a tuple or list of ints"
        assert len(image_height_width) == 2, "FR ERROR: image_height_width must be a tuple or list of length 2: (height, width)"
        assert isinstance(points_sizes, (float, int, list)), "FR ERROR: points_sizes must be a float, int, or list of floats or ints"
        if isinstance(points_sizes, list):
            assert all([isinstance(x, (float, int)) for x in points_sizes]), "FR ERROR: points_sizes must be a float, int, or list of floats or ints"
        assert isinstance(verbose, int), "FR ERROR: verbose must be an int"

        ## Set variables
        self.image_height_width = tuple(image_height_width)
        self.points_sizes = [float(p) for p in points_sizes]
        self._verbose = int(verbose)

        ## Make convolutional kernels for points
        def _make_convolution_kernel(point_size):
            ## Make a cosine kernel
            d = int((point_size//2)*2 + 1)
            c = int(point_size//2)
            x = np.linspace(-c, c, d)
            y = np.linspace(-c, c, d)
            xx, yy = np.meshgrid(x, y)
            r = np.sqrt(xx**2 + yy**2)
            kernel = np.cos(r/point_size*np.pi/2)
            kernel[r > point_size/2] = 0
            kernel = kernel / np.sum(kernel)
            return kernel
        self.kernels = [_make_convolution_kernel(p) for p in self.points_sizes]
        
        ## Initialize Toeplitz convolution objects
        self.convolvers = [Toeplitz_convolution2d(
            x_shape=self.image_height_width,
            k=kernel,
            mode='same',
            dtype=np.float32,
        ) for kernel in self.kernels]


def visualize_image_with_points(
    image,
    points=None,

    points_colors=(0, 255, 255),
    alpha=1.0,
    points_sizes=1,
    
    text=None,
    text_positions=None,
    text_color='white',
    text_size=1,
    text_thickness=1,

    display=False,
    handle_cv2Imshow='FaceRhythmPointVisualizer',
    writer_cv2=None,

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
                is point coordinates. Order (x,y).
        points_colors (tuple of int or list of tuple of int):
            Used as argument for cv2.circle.
            If tuple: All points will be this color.
                Elements of tuple should be 3 integers between 0 and 255.
            If list: Each element is a color for a batch of points.
                Length of list must match the first dimension of points.
                points must be 3D array.
        alpha (float):
            Transparency of points.
            Note that values other than 1 will be slow for now.
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
            2D array: Each row is a text position. Order (x,y).
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
            assert np.all(points[:, :, 1] < image.shape[0]), 'points must be within image.'
            assert np.all(points[:, :, 0] < image.shape[1]), 'points must be within image.'

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
            assert text_positions.shape[-1] == 2, 'text_positions must have 2 coordinates (x,y).'

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


    ## Make a copy of image
    image_out = image.copy()

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
            # for i_point in range(points.shape[1]):
            #     cv2.circle(
            #         img=image_out,
            #         center=tuple(points[i_batch][i_point]),
            #         radius=points_sizes[i_batch],
            #         color=points_colors[i_batch],
            #         thickness=-1,
            #     )
            mat = scipy.sparse.coo_matrix((image_out.shape[0], image_out.shape[1]), dtype=np.uint8)
            mat.row = points[i_batch][:, 1]
            mat.col = points[i_batch][:, 0]
            mat.data = np.ones(points[i_batch].shape[0], dtype=np.uint8)*(int(255*0.2))
            
            # image_out = cv2.add(image_out, mat.toarray()[:, :, None]*np.array(points_colors[i_batch], dtype=np.uint8)[None, None, :])
            image_out = cv2.add(image_out, cv2.cvtColor(mat.toarray(), cv2.COLOR_GRAY2BGR))


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
    if display:
        cv2.imshow(handle_cv2Imshow, image_out)
        cv2.waitKey(1)

    ## Write image_out
    if writer_cv2 is not None:
        writer_cv2.write(image_out)

    return image_out
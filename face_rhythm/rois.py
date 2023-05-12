from pathlib import Path

import cv2
import numpy as np
import hdfdict
import scipy.interpolate
from tqdm import tqdm
import matplotlib.pyplot as plt

from .util import FR_Module
from . import h5_handling


class ROIs(FR_Module):
    def __init__(
        self,
        select_mode="gui_notebook",

        exampleImage=None,
        path_file=None,
        points=None,
        mask_images=None,
        verbose=1,
    ):
        """
        Initialize the class.
        Four ways to initialize depending on the select_mode.
        Look carefully at options for each mode.
        RH 2022

            Args:
                select_mode (str):
                    'gui' (default):
                        Make a GUI using matplotlib and ipywidgets.
                        'exampleImage' must be provided.
                    'file':
                        Load a file with the ROI points.
                        This should be an existing 'ROIs.h5' file from
                         a previous run.
                        'path_file' must be provided, and either 
                    'custom':
                        Must provide:
                            - points
                            - exampleImage
                        Will compute the masks from the points.
                exampleImage (np.ndarray):
                    Image to show in the GUI to select the ROIs.
                    Only used if select_mode is 'gui'.
                path_file (str):
                    Path to the file to load.
                    Only used if select_mode is 'file'.
                points (list of dictionaries containing either np.ndarray or 2-element lists of float):
                    Dictionary of points of the boundaries of the ROIs.
                    Should be in format:
                        {'ROI_0': np.ndarray, 'ROI_1': np.ndarray, ...}
                        OR
                        {'ROI_0': [[x1, y1], [x2, y2], ...], 'ROI_1': [[x1, y1], [x2, y2], ...], ...}
                    Only used if select_mode is 'custom'.
                mask_images (list of np.ndarray):
                    List of mask images of the ROIs.
                    Must have same height and width as the videos.
                    Should be in format:
                        [mask1, mask2, ...]
                        where each mask is a 2D boolean np.ndarray.
                    Only used if select_mode is 'mask'.
                verbose (int):
                    Verbosity level.
                    0: No output
                    1: Warnings
                    2: All output
        """
        super().__init__()
        self._select_mode = select_mode
        self.exampleImage = exampleImage
        self._path_file = path_file
        self.roi_points = points
        self.mask_images = mask_images
        self._verbose = int(verbose)


        self.img_hw = self.exampleImage.shape[:2] if self.exampleImage is not None else None

        ## Assert that the correct arguments are provided for the select_mode
        assert isinstance(select_mode, str), "FR ERROR: select_mode must be a string."
        if (select_mode == "gui"):
            assert exampleImage is not None, "FR ERROR: 'exampleImage' must be provided for select_mode 'gui'."
        elif select_mode == "file":
            assert self._path_file is not None, "FR ERROR: 'path_file' must be provided for select_mode 'file'."
            assert isinstance(self._path_file, str), "FR ERROR: 'path_file' must be a string."
            assert Path(self._path_file).exists(), f"FR ERROR: 'path_file' does not exist: {self._path_file}"
        elif select_mode == "custom":
            assert self.roi_points is not None, "FR ERROR: 'points' must be provided for select_mode 'custom'."
            assert self.exampleImage is not None, "FR ERROR: 'exampleImage' must be provided for select_mode 'custom'."
            assert isinstance(self.roi_points, dict), "FR ERROR: 'points' must be a dictionary."
            assert all([isinstance(v, (np.ndarray, list)) for v in self.roi_points.values()]), "FR ERROR: 'points' must be a dictionary of numpy arrays or lists."
            if isinstance(self.roi_points[list(self.roi_points.keys())[0]], np.ndarray):
                assert all([v.shape[1] == 2 for v in self.roi_points.values()]), "FR ERROR: 'points' must be a dictionary of numpy arrays of shape (N, 2)."
            elif isinstance(self.roi_points[list(self.roi_points.keys())[0]], list):
                assert all([len(v) == 2 for v in self.roi_points.values()]), "FR ERROR: 'points' must be a dictionary of lists of length 2."
        else:
            raise ValueError("FR ERROR: 'select_mode' must be one of 'gui', 'file', 'custom'.")


        if select_mode == "gui":
            print(f"FR: Initializing GUI...") if self._verbose > 1 else None
            self._gui = _Select_ROI(exampleImage)
            self._gui._ax.set_title('Select ROIs by clicking the image.')
            self.roi_points = self._gui.selected_points
            self.mask_images = self._gui.mask_frames
        
        if select_mode == "file":
            file = h5_handling.simple_load(self._path_file)
            ## Check that the file has the correct format
            assert "mask_images" in file, "FR ERROR: 'mask_images' not found in file."
            self.mask_images = file["mask_images"]
            ## Check that the mask images have the correct format
            assert isinstance(self.mask_images, (dict, hdfdict.hdfdict.LazyHdfDict)), "FR ERROR: 'mask_images' must be a dict or hdfdict.hdfdict.LazyHdfDict containing boolean numpy arrays representing the mask images."
            assert all([isinstance(mask, np.ndarray) for mask in self.mask_images.values()]), "FR ERROR: 'mask_images' from file is expected to be a 3D or list of 2D boolean np.ndarray."
            assert all([mask.shape == self.mask_images[list(self.mask_images.keys())[0]].shape for mask in self.mask_images.values()]), "FR ERROR: 'mask_images' must all have the same shape."
            assert all([mask.dtype == bool for mask in self.mask_images.values()]), "FR ERROR: 'mask_images' must be boolean."
            self.mask_images = {k: np.array(v, dtype=np.bool_) for k, v in self.mask_images.items()}  ## Ensure that the masks are boolean np arrays
            ## Check that roi_points has the correct format
            assert "roi_points" in file, "FR ERROR: 'roi_points' not found in file."
            self.roi_points = file["roi_points"]
            self.roi_points = {k: np.array(v, dtype=np.float32) for k, v in self.roi_points.items()}  ## Ensure that the roi_points are float np arrays
            ## Check that exampleImage has the correct format
            assert "exampleImage" in file, "FR ERROR: 'exampleImage' not found in file."
            self.exampleImage = file["exampleImage"]
            self.exampleImage = np.array(self.exampleImage, dtype=np.float32)  ## Ensure that the exampleImage is a float np array
            self.img_hw = self.exampleImage.shape
            
        elif select_mode == "custom":
            print(f"FR: Initializing ROIs from points...") if self._verbose > 1 else None
            self.mask_images = _Select_ROI._compute_mask_frames(
                selected_points=self.roi_points,
                exampleImage=self.exampleImage,
            )
        

        ## For FR_Module compatibility
        self.config = {
            "select_mode": self._select_mode,
            "exampleImage": (self.exampleImage is not None),
            "path_file": path_file,
            "roi_points": (points is not None),
            "mask_images": (mask_images is not None),
            "verbose": self._verbose,
        }
        self.run_info = {
            "img_hw": self.img_hw,
        }
        self.run_data = {
            "mask_images": self.mask_images,
            "roi_points": self.roi_points,
            "exampleImage": self.exampleImage,
        }
        # ## Append the self.run_info data to self.run_data
        # self.run_data.update(self.run_info)

    def make_points(self, rois, point_spacing=10):
        """
        Make points from a list of ROIs.

        Args:
            rois (list of np.ndarray): 
                List of ROIs.
                Each ROI should be a 2D boolean numpy array.
            point_spacing (int):
                Spacing between points in pixels.

        Returns:
            self.point_positions (np.ndarray):
                Array of point positions.
                Shape is (n_points, 2).
                (x, y) coordinates.
        """
        ## Assertions
        ## rois should either be a list of 2D arrays or 3D array or a single 2D array
        assert isinstance(rois, (list, np.ndarray)), "FR ERROR: 'rois' must be a list of 2D arrays or a 3D array or a single 2D array."
        if isinstance(rois, list):
            assert all([isinstance(roi, np.ndarray) for roi in rois]), "FR ERROR: 'rois' must be a list of 2D arrays or a 3D array or a single 2D array."
            assert all([roi.shape == rois[0].shape for roi in rois]), "FR ERROR: shapes of all 'rois' must be the same."
            assert all([roi.dtype == bool for roi in rois]), "FR ERROR: 'rois' must be boolean."
        elif isinstance(rois, np.ndarray):
            if rois.ndim == 2:
                assert rois.dtype == bool, "FR ERROR: 'rois' must be boolean."
                rois = [rois]
            elif rois.ndim == 3:
                assert all([roi.dtype == bool for roi in rois]), "FR ERROR: 'rois' must be boolean."
                rois = [roi for roi in rois]

        ## Make points within rois_points with spacing of point_spacing
        ##  First make a single ROI boolean image, then make points
        print("FR: Making points to track") if self._verbose > 1 else None
        rois_all = np.stack(rois, axis=0).all(axis=0)
        self.point_positions = self._helper_make_points(rois_all, point_spacing)
        self.num_points = self.point_positions.shape[0]
        print(f"FR: {self.point_positions.shape[0]} points total") if self._verbose > 1 else None

        self.run_data.update({
            "point_positions": self.point_positions,
        })

    def _helper_make_points(self, roi, point_spacing):
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

    def __repr__(self):
        return f"ROIs object. Select mode: '{self._select_mode}'. Number of ROIs: {len(self.mask_images)}."

    ## Define methods for loading and handling videos
    def __getitem__(self, index): 
        if isinstance(index, int):
            index = list(self.mask_images.keys())[index]
        return self.mask_images[index]
    def __len__(self): return len(self.maks_images)
    def __iter__(self): return iter(self.mask_images)
    def __next__(self): return next(self.mask_images)

    def plot_rois(self, image=None, **kwargs_imshow):
        """
        Plot the rois.
        If an image exists, it makes polygons of the rois on top of 
         the image in different colors.
        If no image exists, it plots the rois on the existing
         self.exampleImage.

        Args:
            image (np.ndarray):
                Image to plot the rois on top of.
                If None, the rois are plotted on the existing
                 self.exampleImage.
            **kwargs_imshow:
                Keyword arguments for plt.imshow().

        Returns:
            fig (plt.figure):
                Figure object.
            ax (plt.axis):
                Axis object.
        """
        import matplotlib.pyplot as plt
        if image is None:
            if hasattr(self, "exampleImage"):
                image = self.exampleImage
            else:
                print("FR WARNING: self.exampleImage does not exist. Plotting a blank image.")
                image = np.zeros((self.img_hw[0], self.img_hw[1]), dtype=np.uint8)


        ## set backend to non-interactive
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, **kwargs_imshow)
        ## Make mask polygons
        for ii, mask in enumerate(self.mask_images.values()):
            ax.contour(mask, colors=[plt.cm.tab20(ii)], linewidths=2, alpha=0.5)
        ## Show points on the image
        if self.point_positions is not None:
            ax.scatter(
                self.point_positions[:, 0], 
                self.point_positions[:, 1], 
                s=2, 
                color="red",
                alpha=0.5,
            )
        ## show figure
        plt.show()
        return fig, ax



class _Select_ROI:
    """
    A simple GUI to select ROIs from an image.
    Uses matplotlib and ipywidgets.
    Only works in a Jupyter notebook.
    Select regions of interest in an image using matplotlib.
    Use %matplotlib notebook or qt backend to use this.
    It currently uses cv2.polylines to draw the ROIs.
    Output is self.mask_frames
    RH 2021

    outputs:
        self.selected_points (list):
            List of points selected by the user.
            Shape: (n_points, 2)
            (x, y) coordinates of the points.
        self.mask_frames (list):
            List of mask images.
            Shape: (n_roi, img_height, img_width)
            Each mask image is a 2D array of booleans.
    """

    def __init__(self, image, kwargs_subplots={}, kwargs_imshow={}):
        """
        Initialize the class

        Args:
            im:
                Image to select the ROI from
        """
        from ipywidgets import widgets
        import IPython.display as Disp
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        import torch

        ## Make sure that the image is a numpy array or torch.Tensor
        assert isinstance(image, (np.ndarray, torch.Tensor)), "FR ERROR: 'image' must be a numpy array or torch.Tensor."
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        ## set jupyter notebook to use interactive matplotlib.
        ## equivalent to %matplotlib notebook
        mpl.use("nbagg")        
        plt.ion()

        ## Set variables                
        self._img_input = image.copy()
        self.selected_points = {}
        self._selected_points_last_ROI = []
        self.mask_frames = {}
        self._completed_status = False

        ## Prepare figure
        self._fig, self._ax = plt.subplots(**kwargs_subplots)
        self._img_current = self._ax.imshow(self._img_input.copy(), **kwargs_imshow)
        self._fig.canvas.draw()

        ## Connect the click event
        self._buttonRelease = self._fig.canvas.mpl_connect('button_release_event', self._onclick)
        ## Make and connect the buttons
        disconnect_button = widgets.Button(description="Confirm ROI")
        new_ROI_button = widgets.Button(description="New ROI")
        Disp.display(disconnect_button)
        Disp.display(new_ROI_button)
        disconnect_button.on_click(self._disconnect_mpl)
        new_ROI_button.on_click(self._new_ROI)

    def _poly_img(self, img, pts):
        """
        Draw a polygon on an image.
        """
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(
            img=img, 
            pts=[pts],
            isClosed=True,
            color=(255,255,255,),
            thickness=2
        )
        return img

    def _onclick(self, event):
        """
        When the mouse is clicked, add the point to the list.
        """
        ## If the click is outside the image, ignore it
        if (event.xdata is None) or (event.ydata is None):
            return None
        self._selected_points_last_ROI.append([float(event.xdata), float(event.ydata)])
        if len(self._selected_points_last_ROI) > 1:
            self._fig
            im = self._img_input.copy()
            for p in self.selected_points.values():
                im = self._poly_img(im, p)
            im = self._poly_img(im, self._selected_points_last_ROI)
            self._img_current.set_data(im)
        self._fig.canvas.draw()


    def _disconnect_mpl(self, _):
        """
        Disconnect the click event and collect the points.
        """
        self.selected_points.update({f"ROI_{len(self.selected_points)}": np.array(self._selected_points_last_ROI)})

        self._fig.canvas.mpl_disconnect(self._buttonRelease)
        self._completed_status = True

        self.mask_frames.update(self._compute_mask_frames(
            selected_points=self.selected_points, 
            exampleImage=self._img_input,
            verbose=True,
        ))

    @staticmethod
    def _compute_mask_frames(
        selected_points,
        exampleImage,
        verbose=False,
    ):
        import skimage
        mask_frames = {}
        for ii, pts in enumerate(selected_points.values()):
            pts = np.array(pts)
            mask_frame = np.zeros((exampleImage.shape[0], exampleImage.shape[1]))
            pts_y, pts_x = skimage.draw.polygon(pts[:, 1], pts[:, 0])
            mask_frame[pts_y, pts_x] = 1
            mask_frame = mask_frame.astype(np.bool)
            mask_frames.update({f"mask_{ii}": mask_frame})
        print(f'mask_frames computed') if verbose else None
        return mask_frames
        
    def _new_ROI(self, _):
        """
        Start a new ROI.
        """
        self.selected_points.update({f"ROI_{len(self.selected_points)}": np.array(self._selected_points_last_ROI)})
        self._selected_points_last_ROI = []
        
        

##########################################################################################################################################
######################################################### MULTISESSION ALIGNMENT #########################################################
##########################################################################################################################################


class ROI_Alinger:
    """
    A class for registering a template image to a 
     set of images, and warping points from the
     template image to the set of images.
    Currently relies on available OpenCV methods for 
     non-rigid registration.
    RH 2022
    """
    def __init__(
        self,
        method='createOptFlow_DeepFlow',
        kwargs_method=None,
        verbose=1,
    ):
        """
        Initialize the class.

        Args:
            method (str):
                The method to use for optical flow calculation.
                The following are currently supported:
                    'calcOpticalFlowFarneback',
                    'createOptFlow_DeepFlow',
            kwargs_method (dict):
                The keyword arguments to pass to the method.
                See the documentation for the method for the
                 required arguments.
                If None, hard-coded defaults will be used.
            verbose (bool):
                Whether to print progress updates.
                0: No updates
                1: Warnings
                2: All updates
        """
        self._verbose = verbose
        self._method = method
        self._kwargs_method = kwargs_method

    def align_and_make_ROIs(
        self,
        ROIs_object_template,
        images_new,
        image_template=None,
        template_method='image',
        shifts=None,
        normalize=True,
    ):
        """
        Perform non-rigid registration of a template image
         to a set of images.
        Currently relies on available OpenCV methods for
         non-rigid registration.
        RH 2022

        Args:
            ROIs_object_template (face_rhythm.rois.ROIs):
                A single ROIs object made using the template image.
                The ROIs and points from this object will be aligned
                 (warped) onto the new images.
            images_new (list of numpy.ndarray):
                The images to project the points onto.
                Template image will be warped onto each image.
                Each image should be of shape (height, width, n_channels)
                 and have dtype uint8.
            image_template (numpy.ndarray):
                The template image to warp onto the new images.
                Optional. If None, then the template image will be
                 taken from the ROIs object: ROIs.exampleImage
                shape: (height, width, n_channels)
                dtype: uint8
            template_method (str):
                The method used to register the images.
                Either 'image' or 'sequential'.
                If 'image':      image_template must be a single image.
                If 'sequential': image_template must be an integer corresponding 
                 to the index of the image to set as 'zero' offset.
            shifts (numpy.ndarray):
                The shifts to apply to the points.
                If None, no shifts will be applied.
                The shifts describe the relative shift between the 
                 original image and the provided image in images_new.
                This will be non-zero if the 
                 input iamges have been shifted using the phase-
                 correlation shifter. 
            normalize (bool):
                If True, the images will be normalized to be in the
                 range [0, 255] before registration. Min and max values
                 will be used to set range.
        """
        ### Assert images_new is a list of 2D or 3D numpy.ndarray
        if isinstance(images_new, list):
            assert all([isinstance(ii, np.ndarray) for ii in images_new]), 'images_new must be a list of 2D numpy.ndarray'
            assert all([len(ii.shape) == 3 for ii in images_new]), 'images_new must be a list of 2D numpy.ndarray'
        if isinstance(images_new, np.ndarray):
            assert len(images_new.shape) == 4, 'images_new must be a list of 3D numpy.ndarray'
            images_new = [images_new[ii] for ii in range(images_new.shape[0])]
        ### Assert template_method is a string in ['image', 'sequential']
        assert isinstance(template_method, str), 'template_method must be a string'
        assert template_method in ['image', 'sequential'], 'template_method must be a string in ["image", "sequential"]'
        
        self._image_template = image_template if image_template is not None else ROIs_object_template.exampleImage
        self._images_new = images_new
        self._template_method = template_method
        self._normalize = normalize

        self._shifts = [(0,0)] * len(self._images_new) if shifts is None else shifts

        self._pointPositions_template = ROIs_object_template.point_positions ## List of point positions for each ROI
        self._roiPoints_template = ROIs_object_template.roi_points  ## List of points describing the outline of each ROI

        ## Make grid of indices for image remapping
        self._dims = self._image_template.shape
        self._x_arange, self._y_arange = np.arange(0., self._dims[1]).astype(np.float32), np.arange(0., self._dims[0]).astype(np.float32)
        self._x_grid, self._y_grid = np.meshgrid(self._x_arange, self._y_arange)

        ## Register images
        print('Registering images...')
        self.flows = [self._register_image(
            image_moving=self._image_template,
            image_template=im_new,
            shifts=shift,
            normalize=self._normalize,
        ) for im_new, shift in tqdm(zip(self._images_new, self._shifts), total=len(self._images_new))]

        ## Warp point_positions
        print('Warping point positions...')
        self.pointPositions_new = [self._warp_points(
            points=self._pointPositions_template,
            flow=flow,
        ) for flow in tqdm(self.flows)]

        ## Warp ROI outlines
        print('Warping ROI outlines...')
        self.roiPoints_new = [{
            key: self._warp_points(
                points=points,
                flow=flow,
            ) for key,points in self._roiPoints_template.items()} for flow in tqdm(self.flows)]

        ## Make mask images
        print('Making mask images...')
        self.maskImages_new = [_Select_ROI._compute_mask_frames(
            selected_points=points,
            exampleImage=self._image_template,
        ) for points in tqdm(self.roiPoints_new)]

        ## Make ROIs object
        print('Making ROIs objects...')
        self.ROIs_objects_new = [ROIs(
            select_mode='custom',
            exampleImage=self._image_template,
            points=points,
        ) for points in tqdm(self.roiPoints_new)]

        ## Warp images
        print('Warping images...')
        self.images_warped = [self._warp_image(
            image=img,
            flow=flow,
        ) for flow,img in tqdm(zip(self.flows, self._images_new), total=len(self._images_new))]



    def _register_image(
        self,
        image_moving,
        image_template,
        shifts=None,
        normalize=True,
    ):
        """
        Perform non-rigid registration of a template image
         to a set of images.
        Currently relies on available OpenCV methods for
         non-rigid registration.
        RH 2022

        Args:
            image_moving (list of numpy.ndarray):
                The image to warp onto image_template.
                Image should be of shape (height, width, n_channels)
                 and have dtype uint8.
            image_template (numpy.ndarray):
                The template image to align (warp) onto.
                shape: (height, width, n_channels)
                dtype: uint8
            normalize (bool):
                If True, the images will be normalized to be in the
                 range [0, 255] before registration. Min and max values
                 will be used to set range.

        Returns:
            image_moving_aligned (numpy.ndarray):
                The moving image warped onto the template image.
            flow (list of numpy.ndarray):
                The optical flow to warp the moving image onto the 
                 template image.
        """

        # Check inputs
        ### Assert image_template is a 3D numpy.ndarray
        assert isinstance(image_template, np.ndarray), 'image_template must be a numpy.ndarray'
        assert image_template.ndim in [2,3], 'image_template must be a 2D or 3D numpy.ndarray'
        if image_template.ndim == 3:
            image_template = image_template.mean(axis=2)
        ### Assert image_moving is a 2D or 3D numpy.ndarray
        assert isinstance(image_moving, np.ndarray), 'image_moving must be a numpy.ndarray'
        assert image_moving.ndim in [2, 3], 'image_moving must be a 2D or 3D numpy.ndarray'
        if image_moving.ndim == 3:
            image_moving = image_moving.mean(axis=2)
        ### Assert shifts is a numpy.ndarray or None or tuple of length 2
        assert isinstance(shifts, (np.ndarray, type(None), tuple)), 'shifts must be a numpy.ndarray or None or tuple of length 2'
        if isinstance(shifts, np.ndarray):
            assert len(shifts.shape) == 2, 'shifts must be a 2D numpy.ndarray'
            assert shifts.shape[1] == 2, 'shifts must be of shape (n_images, 2)'
        if isinstance(shifts, tuple):
            assert len(shifts) == 2, 'shifts must be a tuple of length 2'
            shifts = np.array(shifts)

        # Normalize images
        if normalize:
            image_moving = ((image_moving - image_moving.min()) / (image_moving.max() - image_moving.min()) * 255).astype(np.uint8)
            image_template = ((image_template - image_template.min()) / (image_template.max() - image_template.min()) * 255).astype(np.uint8)
        
        if self._method == 'calcOpticalFlowFarneback':
            if self._kwargs_method is None:
                self._kwargs_method = {
                    'pyr_scale': 0.3, 
                    'levels': 3,
                    'winsize': 128, 
                    'iterations': 7,
                    'poly_n': 7, 
                    'poly_sigma': 1.5,
                    'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                }
            flow = cv2.calcOpticalFlowFarneback(
                prev=image_moving,
                next=image_template, 
                flow=None, 
                **self._kwargs_method,
            )
    
        elif self._method == 'createOptFlow_DeepFlow':
            flow = cv2.optflow.createOptFlow_DeepFlow().calc(
                image_moving,
                image_template,
                None
            )

        ### Apply shifts
        if shifts is not None:
            flow += shifts
            
        return flow 


    def _warp_points(
        self,
        points,
        flow,
    ):
        """
        Warp points using provided flow field.
        RH 2022

        Args:
            points (numpy.ndarray):
                The points to warp.
                shape: (n_points, 2)
                dtype: float
                (x, y) coordinates
            flow (numpy.ndarray):
                The flow field to warp the points.
                shape: (height, width, 2)
                dtype: float
                last dim is (x, y) coordinates
        """
        from functools import partial
        ### Assert points is a 2D numpy.ndarray of shape (n_points, 2) and that all points are within the image and that points are float
        assert isinstance(points, np.ndarray), 'points must be a numpy.ndarray'
        assert len(points.shape) == 2, 'points must be a 2D numpy.ndarray'
        assert points.shape[1] == 2, 'points must be of shape (n_points, 2)'
        assert (points[:,0] >= 0).all(), 'points must be within the image'
        assert (points[:,0] < flow.shape[1]).all(), 'points must be within the image'
        assert (points[:,1] >= 0).all(), 'points must be within the image'
        assert (points[:,1] < flow.shape[0]).all(), 'points must be within the image'
        assert np.issubdtype(points.dtype, np.floating), 'points must be a float subtype'

        x_remap = (self._x_grid + flow[:, :, 0]).astype(np.float32)
        y_remap = (self._y_grid + flow[:, :, 1]).astype(np.float32)

        ## Use a RectBivariateSpline to remap points
        splineGrid = partial(
            scipy.interpolate.RectBivariateSpline,
            x=self._y_arange,
            y=self._x_arange,
            kx=1,
            ky=1,
            s=0
        )
        splineGrid_x, splineGrid_y = splineGrid(z=x_remap), splineGrid(z=y_remap)
        points_remap_x = splineGrid_x.ev(
            points[:, 1],
            points[:, 0],
        )
        points_remap_y = splineGrid_y.ev(
            points[:, 1],
            points[:, 0],
        )

        points_remap = np.array([points_remap_x, points_remap_y]).T

        ## Clip points to image size
        points_remap[:, 0] = np.clip(points_remap[:, 0], 0, flow.shape[1] - 1)
        points_remap[:, 1] = np.clip(points_remap[:, 1], 0, flow.shape[0] - 1)

        return points_remap


    def _warp_image(
        self,
        image,
        flow,
    ):
        """
        Warp image using provided flow field.
        RH 2022

        Args:
            image (numpy.ndarray):
                The image to warp.
                shape: (height, width)
                 or (height, width, 3)
                dtype: float
            flow (numpy.ndarray):
                The flow field to warp the image.
                shape: (height, width, 2)
                dtype: float
                last dim is (x, y) coordinates
        """
        def safe_remap(image, x_remap, y_remap):
            image_remap = cv2.remap(
                image.astype(np.float32),
                x_remap,
                y_remap, 
                cv2.INTER_LINEAR
            )
            if image_remap.sum() == 0:
                image_remap = image
            return image_remap

        if image.ndim == 3:
            image = image.mean(axis=2)
        assert image.ndim == 2, 'image must be 2D'
        assert image.shape[0] == flow.shape[0], 'image and flow must have the same height'
        assert image.shape[1] == flow.shape[1], 'image and flow must have the same width'
        assert np.issubdtype(image.dtype, np.floating), 'image must be a float subtype'

        x_remap = (self._x_grid + flow[:, :, 0]).astype(np.float32)
        y_remap = (self._y_grid + flow[:, :, 1]).astype(np.float32)

        image_remap = safe_remap(
            image=image,
            x_remap=x_remap,
            y_remap=y_remap
        )

        return image_remap

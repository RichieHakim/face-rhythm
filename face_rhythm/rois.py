from pathlib import Path
import time
import threading

import cv2
import numpy as np
import hdfdict

from .util import FR_Module
from . import h5_handling

class ROIs(FR_Module):
    def __init__(
        self,
        select_mode="gui_notebook",

        exampleImage=None,
        file_path=None,
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
                        'file_path' must be provided, and either 
                    'points':
                        Provide a list of points of the boundaries of
                         the ROIs.
                        'points' and 'exampleImage' must be provided.
                    'mask':
                        Provide a list of mask images of the ROIs.
                        'mask_images' must be provided.

                exampleImage (np.ndarray):
                    Image to show in the GUI to select the ROIs.
                    Only used if select_mode is 'gui'.
                file_path (str):
                    Path to the file to load.
                    Only used if select_mode is 'file'.
                points (list of list of 2-element lists of float):
                    List of points of the boundaries of the ROIs.
                    Should be in format:
                        [[[y1, x1], [y2, x2], ...], ...]
                        where each outer list element is an ROI and
                            each inner list element is a [y, x] point.
                    Only used if select_mode is 'points'.
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
        self._file_path = file_path
        self.points = points
        self.mask_images = mask_images
        self.verbose = int(verbose)


        self.img_hw = self.exampleImage.shape[:2] if self.exampleImage is not None else None

        ## Assert that the correct arguments are provided for the select_mode
        assert isinstance(select_mode, str), "FR ERROR: select_mode must be a string."
        if (select_mode == "gui"):
            assert exampleImage is not None, "FR ERROR: 'exampleImage' must be provided for select_mode 'gui' or 'gui_tk'."
        elif select_mode == "file":
            assert self._file_path is not None, "FR ERROR: 'file_path' must be provided for select_mode 'file'."
            assert isinstance(self._file_path, str), "FR ERROR: 'file_path' must be a string."
            assert Path(self._file_path).exists(), f"FR ERROR: 'file_path' does not exist: {self._file_path}"
        elif select_mode == "points":
            assert points is not None, "FR ERROR: 'points' argument must be provided for select_mode 'points'."
            assert exampleImage is not None, "FR ERROR: 'exampleImage' must be provided for select_mode 'points'."
            assert isinstance(points, list), "FR ERROR: 'points' must be a list of list of 2-element lists of float."
            assert all([isinstance(p, list) for p in points]), "FR ERROR: 'points' must be a list of list of 2-element (y,x) lists of float."
            assert all([[isinstance(p, list) for p in p_] for p_ in points]), "FR ERROR: 'points' must be a list of list of 2-element (y,x) lists of float."
            assert all([len(p) == 2 for p in p_] for p_ in points), "FR ERROR: 'points' must be a list of list of 2-element (y,x) lists of float."
            assert all([all([isinstance(p, float) for p in p_]) for p_ in points]), "FR ERROR: 'points' must be a list of list of 2-element (y,x) lists of float."
        elif select_mode == "mask":
            assert mask_images is not None, "FR ERROR: 'mask_images' must be provided for select_mode 'mask'."
            assert isinstance(mask_images, list), "FR ERROR: 'mask_images' must be a list of np.ndarray."
            assert all([isinstance(m, np.ndarray) for m in mask_images]), "FR ERROR: 'mask_images' must be a list of np.ndarray."
        else:
            raise ValueError("FR ERROR: 'select_mode' must be one of 'gui', 'file', 'points', or 'mask'.")


        if (select_mode == "gui") or (select_mode == "points"):
            if select_mode == "gui":
                print(f"FR: Initializing GUI...") if self.verbose > 1 else None
                self._gui = _Select_ROI(exampleImage)
                self._gui._ax.set_title('Select ROIs by clicking the image.')
                self.points = self._gui.selected_points
                self.mask_images = self._gui.mask_frames
            elif select_mode == "points":
                self.points = points.copy()
            
                import skimage
                self.mask_images = {}
                for ii, pts in enumerate(self.points):
                    pts = np.array(pts, dtype=float)
                    mask_frame = np.zeros((self.img_hw[0], self.img_hw[1]), dtype=bool)
                    pts_y, pts_x = skimage.draw.polygon(pts[:, 1], pts[:, 0])
                    mask_frame[pts_y, pts_x] = 1
                    mask_frame = mask_frame.astype(np.bool)
                    self.mask_images.update({f"mask_{ii}": mask_frame.astype(bool)})
        
        if select_mode == "file":
            file = h5_handling.simple_load(self._file_path)
            ## Check that the file has the correct format
            assert "mask_images" in file, "FR ERROR: 'mask_images' not found in file."
            self.mask_images = file["mask_images"]
            ## Check that the mask images have the correct format
            assert isinstance(self.mask_images, (dict, hdfdict.hdfdict.LazyHdfDict)), "FR ERROR: 'mask_images' must be a dict or hdfdict.hdfdict.LazyHdfDict containing boolean numpy arrays representing the mask images."
            assert all([isinstance(mask, np.ndarray) for mask in self.mask_images.values()]), "FR ERROR: 'mask_images' from file is expected to be a 3D or list of 2D boolean np.ndarray."
            assert all([mask.shape == self.mask_images[list(self.mask_images.keys())[0]].shape for mask in self.mask_images.values()]), "FR ERROR: 'mask_images' must all have the same shape."
            assert all([mask.dtype == bool for mask in self.mask_images.values()]), "FR ERROR: 'mask_images' must be boolean."
            ## Check that the file has the correct format
            assert "points" in file, "FR ERROR: 'points' not found in file."
            self.points = file["points"]
            ## Check that the points have the correct format

        elif select_mode == "mask":
            print(f"FR: Initializing ROIs from mask images...") if self.verbose > 1 else None
        

        ## For FR_Module compatibility
        self.config = {
            "select_mode": self._select_mode,
            "exampleImage": (self.exampleImage is not None),
            "file_path": file_path,
            "points": (points is not None),
            "mask_images": (mask_images is not None),
            "verbose": self.verbose,
        }
        self.run_info = {
        }
        self.run_data = {
            "mask_images": self.mask_images,
            "points": self.points,
        }
        # ## Append the self.run_info data to self.run_data
        # self.run_data.update(self.run_info)

    def __repr__(self):
        return f"ROIs. Select mode: {self._select_mode}. Number of ROIs: {len(self.mask_images)}."

    ## Define methods for loading and handling videos
    def __getitem__(self, index): 
        if isinstance(index, int):
            index = list(self.mask_images.keys())[index]
        return self.mask_images[index]
    def __len__(self): return len(self.maks_images)
    def __iter__(self): return iter(self.mask_images)
    def __next__(self): return next(self.mask_images)

    def plot_masks(self, image=None, **kwargs_imshow):
        """
        Plot the masks.
        If an image exists, it makes polygons of the masks on top of 
         the image in different colors.
        If no image exists, it plots the masks on a black background.

        Args:
            image (np.ndarray):
                Image to plot the masks on top of.
                If None, the masks are plotted on a black background.
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
            image = np.zeros((self.img_hw[0], self.img_hw[1]), dtype=np.uint8)

        ## set backend to non-interactive
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, **kwargs_imshow)
        ## Make mask polygons
        for ii, mask in enumerate(self.mask_images.values()):
            ax.contour(mask, colors=[plt.cm.tab20(ii)], linewidths=2, alpha=0.5)
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

        import skimage
        for ii, pts in enumerate(self.selected_points.values()):
            pts = np.array(pts)
            mask_frame = np.zeros((self._img_input.shape[0], self._img_input.shape[1]))
            pts_y, pts_x = skimage.draw.polygon(pts[:, 1], pts[:, 0])
            mask_frame[pts_y, pts_x] = 1
            mask_frame = mask_frame.astype(np.bool)
            self.mask_frames.update({f"mask_{ii}": mask_frame})
        print(f'mask_frames computed')
        
    def _new_ROI(self, _):
        """
        Start a new ROI.
        """
        self.selected_points.update({f"ROI_{len(self.selected_points)}": np.array(self._selected_points_last_ROI)})
        self._selected_points_last_ROI = []
        
        

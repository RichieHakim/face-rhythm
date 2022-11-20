import cv2
import numpy as np

from .util import FR_Module

class ROIs(FR_Module):
    def __init__(
        self,
        use_GUI=True,


        points_ROI=None,
        points_mask=None,
        frame_height_width=None,
    ):
        super().__init__()
        self.use_GUI = bool(use_GUI)
        
        ## Assert that if points_ROI is specified, it is a list of lists of 2-tuples (y, x) of floats and that all points are within the image
        # if points_ROI is not None:


class _Select_ROI:
    """
    Select regions of interest in an image using matplotlib.
    Use %matplotlib notebook or qt backend to use this.
    It currently uses cv2.polylines to draw the ROIs.
    Output is self.mask_frames
    RH 2021
    """

    def __init__(self, im, kwargs_subplots={}, kwargs_imshow={}):
        """
        Initialize the class

        Args:
            im:
                Image to select the ROI from
        """
        from ipywidgets import widgets
        import IPython.display as Disp
        import matplotlib.pyplot as plt

        self._im = im
        self.selected_points = []
        self._fig, self._ax = plt.subplots(**kwargs_subplots)
        self._img = self._ax.imshow(self._im.copy(), **kwargs_imshow)
        self._completed_status = False
        self._ka = self._fig.canvas.mpl_connect('button_press_event', self._onclick)
        disconnect_button = widgets.Button(description="Confirm ROI")
        new_ROI_button = widgets.Button(description="New ROI")
        Disp.display(disconnect_button)
        Disp.display(new_ROI_button)
        disconnect_button.on_click(self._disconnect_mpl)
        new_ROI_button.on_click(self._new_ROI)

        self._selected_points_last_ROI = []

    def _poly_img(self, img, pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, 
                      [pts],
                      True,
                      (255,255,255),
                      2)
        return img

    def _onclick(self, event):
        self._selected_points_last_ROI.append([event.xdata, event.ydata])
        if len(self._selected_points_last_ROI) > 1:
            self._fig
            im = self._im.copy()
            for ii in range(len(self.selected_points)):
                im = self._poly_img(im, self.selected_points[ii])
            im = self._poly_img(im, self._selected_points_last_ROI)
            self._img.set_data(im)


    def _disconnect_mpl(self, _):
        import skimage.draw

        self.selected_points.append(self._selected_points_last_ROI)

        self._fig.canvas.mpl_disconnect(self._ka)
        self._completed_status = True
        
        self.mask_frames = []
        for ii, pts in enumerate(self.selected_points):
            pts = np.array(pts)
            mask_frame = np.zeros((self._im.shape[0], self._im.shape[1]))
            pts_y, pts_x = skimage.draw.polygon(pts[:, 1], pts[:, 0])
            mask_frame[pts_y, pts_x] = 1
            mask_frame = mask_frame.astype(np.bool)
            self.mask_frames.append(mask_frame)
        print(f'mask_frames computed')

    def _new_ROI(self, _):
        self.selected_points.append(self._selected_points_last_ROI)
        self._selected_points_last_ROI = []
        
        

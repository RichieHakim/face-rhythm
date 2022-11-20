import tkinter as tk
from tkinter import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import threading

## import raw_input function
# from

## Make a GUI to select ROIs
## Use tkinter
## The GUI should not require mainloop() to be called

class ROI_GUI(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(ROI_GUI, self).__init__(*args, **kwargs)
        self.test = 'hiho'
        self.start()

    def run(self):
        self.root = tk.Tk()
        self.root.title('ROI GUI')
        self.root.geometry('800x600')
        self.root.resizable(False, False)
        print(self.test)

        # self.f = plt.figure(figsize=(6, 6))
        roi = _Select_ROI(np.random.rand(20,30))
        self.f = roi.get_fig()
        # self.ax = self.f.add_subplot(111)
        # self.ax.imshow(np.random.rand(10, 10))
        # self.ax.set_title('Select ROIs')

        self.canvas = FigureCanvasTkAgg(self.f, master=self.root)
        self.canvas.get_tk_widget().grid(row=10, column=0, columnspan=3, sticky='nsew')
        # self.canvas.bind('<Button-1>', self.draw_line)

        self.button = Button(self.root, text='Select ROIs', command=self.select_rois)
        self.button.grid(row=1, column=0, sticky='nsew')

        self.button = Button(self.root, text='Save ROIs', command=self.save_rois)
        self.button.grid(row=1, column=1, sticky='nsew')

        self.root.mainloop()

    # Define a function to draw the line between two points
    def draw_line(self, event):
        x1=event.x
        y1=event.y
        x2=event.x
        y2=event.y
        # Draw an oval in the given co-ordinates
        self.canvas.create_oval(x1,y1,x2,y2,fill="black", width=20)

    def select_rois(self):
        print('selecting rois')

    def save_rois(self):
        print('saving rois')

    def quit(self):
        self.root.quit()

import cv2
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
        # self._fig = plt.figure()
        self._img = self._ax.imshow(self._im.copy(), **kwargs_imshow)
        self._completed_status = False
        self._ka = self._fig.canvas.mpl_connect('button_press_event', self._onclick)
        # disconnect_button = widgets.Button(description="Confirm ROI")
        # new_ROI_button = widgets.Button(description="New ROI")
        # Disp.display(disconnect_button)
        # Disp.display(new_ROI_button)
        # disconnect_button.on_click(self._disconnect_mpl)
        # new_ROI_button.on_click(self._new_ROI)

        self._selected_points_last_ROI = []

    def get_fig(self):
        return self._fig

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
        print(self._selected_points_last_ROI)

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
        
        

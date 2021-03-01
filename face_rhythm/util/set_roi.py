from face_rhythm.util import helpers
import cv2
import numpy as np
import skimage.draw

import matplotlib.pyplot as plt
import IPython.display as Disp
from ipywidgets import widgets

from pathlib import Path

import h5py


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
colors = (WHITE, RED, GREEN, BLUE, CYAN, MAGENTA, YELLOW)
pts = []


def load_video(vid_to_set, frame_to_set, videos):
    """
    loads the chosen video and returns chosen frame
    Parameters
    ----------
    vidToSet (int): index of vid to load
    frameToSet (int): index of frame to load
    path_vid_allFiles (list): list of all video paths
    Returns
    -------
    frame (cv2.image): frame read using cv2
    """
    path_vid = videos[vid_to_set - 1]
    vid_reader = cv2.VideoCapture(path_vid)

    vid_reader.set(1, frame_to_set)
    ok, frame = vid_reader.read()
    return frame


# cv2.waitKey(1);


## The below block is adapted code. It makes a GUI, then allows a user to click to define the
## outline of the ROI to use. 'pts' are the clicked points.
# prepare for appending. I'm using this global in functions like a pleb. please forgive
def draw(x):
    """
    draws line using clicks
    Parameters
    ----------
    x :
    Returns
    -------
    """
    d = cv2.getTrackbarPos('thickness', 'window')
    d = 1 if d == 0 else d
    i = cv2.getTrackbarPos('color', 'window')
    color = colors[i]
    cv2.polylines(frame, np.array([pts]), False, color, d)
    cv2.imshow('window', frame)
    text = f'color={color}, thickness={d}'


#     cv2.displayOverlay('window', text)


def mouse(event, x, y, flags, param):
    """
    tracks mouse events and adds click locations to the list
    Parameters
    ----------
    event (event object): event object
    x (int): x location
    y (int): y location
    Returns
    -------
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        draw(0)


def create_seg(frame):
    """
    complete polygon from lines, then fill to create mask
    Parameters
    ----------
    frame (cv2.image): the frame of interest
    Returns
    -------
    """
    cv2.imshow('window', frame)
    cv2.setMouseCallback('window', mouse)
    cv2.createTrackbar('color', 'window', 0, 6, draw)
    cv2.createTrackbar('thickness', 'window', 1, 10, draw)
    draw(0)
    cv2.waitKey(0)
    ## The below block "fills in" the indices of all the points within the above defined bounds
    mask_frame = np.zeros((frame.shape[0], frame.shape[1]))
    pts_y, pts_x = skimage.draw.polygon(np.array(pts)[:, 1], np.array(pts)[:, 0])
    mask_frame[pts_y, pts_x] = 1

    cv2.imshow('window', frame * np.uint8(np.repeat(mask_frame[:, :, None], 3, axis=2)))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pts_y, pts_x, mask_frame


def get_bbox(mask_frame):
    """
    get rectangular bounding box for irregular roi
    Parameters
    ----------
    mask_frame (np.ndarray): the frame containing the mask
    Returns
    -------
    bbox (np.ndarray): numpy array containing the indexes of the bounding box
    """
    bbox = np.zeros(4)
    bbox[0] = np.min(np.where(np.max(mask_frame, axis=0)))  # x top left
    bbox[1] = np.min(np.where(np.max(mask_frame, axis=1)))  # y top left
    bbox[2] = np.max(np.where(np.max(mask_frame, axis=0))) - bbox[0]  # x size
    bbox[3] = np.max(np.where(np.max(mask_frame, axis=1))) - bbox[1]  # y size
    bbox = np.int64(bbox)
    return bbox


def roi_workflow(config_filepath):
    """
    manages the workflow for loading and collecting rois
    Parameters
    ----------
    config_filepath (Path): path to config
    Returns
    -------
    """
    config = helpers.load_config(config_filepath)
    if config['load_from_file']:
        with h5py.File(Path(config['path_data']) / 'pts_all.h5', 'r') as pt:
            pts_all = helpers.h5_to_dict(pt)
        helpers.save_h5(config_filepath, 'pts_all', pts_all)
        return

    global frame
    frame = load_video(config['vidToSet'], config['frameToSet'], config['path_vid_allFiles'])
    pts_y, pts_x, mask_frame = create_seg(frame)
    bbox = get_bbox(mask_frame)
    bbox_subframe_displacement = bbox
    pts_displacement, pts_x_displacement, pts_y_displacement = pts, pts_x, pts_y
    mask_frame_displacement = mask_frame
    cv2.destroyAllWindows()
    pts_all = dict([
        ('bbox_subframe_displacement', bbox_subframe_displacement),
        ('pts_displacement', pts_displacement),
        ('pts_x_displacement', pts_x_displacement),
        ('pts_y_displacement', pts_y_displacement),
        ('mask_frame_displacement', mask_frame_displacement)
    ])
    helpers.save_h5(config_filepath, 'pts_all', pts_all)


class BBoxSelect:

    def __init__(self, im):
        self.im = im
        self.selected_points = []
        self.fig, ax = plt.subplots()
        self.img = ax.imshow(self.im.copy())
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        disconnect_button = widgets.Button(description="Disconnect mpl")
        Disp.display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)

    def poly_img(self, img, pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True,
                      (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)), 2)
        return img

    def onclick(self, event):
        # display(str(event))
        self.selected_points.append([event.xdata, event.ydata])
        if len(self.selected_points) > 1:
            self.fig
            self.img.set_data(self.poly_img(self.im.copy(), self.selected_points))

    def disconnect_mpl(self, _):
        self.fig.canvas.mpl_disconnect(self.ka)


def get_roi(config_filepath):
    config = helpers.load_config(config_filepath)
    roi = config['ROI']
    paths = config['Paths']
    general = config['General']

    if roi['load_from_file']:
        with h5py.File(Path(paths['data']) / 'pts_all.h5', 'r') as pt:
            pts_all = helpers.h5_to_dict(pt)
        helpers.save_h5(config_filepath, 'pts_all', pts_all)
        return None, None

    video_list = general['sessions'][roi['session_to_set']]['videos']
    frame = load_video(roi['vid_to_set'], roi['frame_to_set'], video_list)
    return frame, BBoxSelect(frame)


def process_roi(config_filepath, frame, bs):
    pts = bs.selected_points
    mask_frame = np.zeros((frame.shape[0], frame.shape[1]))
    pts_y, pts_x = skimage.draw.polygon(np.array(pts)[:, 1], np.array(pts)[:, 0])
    mask_frame[pts_y, pts_x] = 1

    bbox = get_bbox(mask_frame)
    bbox_subframe_displacement = bbox
    pts_displacement, pts_x_displacement, pts_y_displacement = pts, pts_x, pts_y
    mask_frame_displacement = mask_frame
    cv2.destroyAllWindows()
    pts_all = dict([
        ('bbox_subframe_displacement', bbox_subframe_displacement),
        ('pts_displacement', pts_displacement),
        ('pts_x_displacement', pts_x_displacement),
        ('pts_y_displacement', pts_y_displacement),
        ('mask_frame_displacement', mask_frame_displacement)
    ])
    helpers.save_h5(config_filepath, 'pts_all', pts_all)

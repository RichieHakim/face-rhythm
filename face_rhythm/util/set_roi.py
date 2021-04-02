from face_rhythm.util import helpers
import cv2
import numpy as np
import skimage.draw

import matplotlib.pyplot as plt
import IPython.display as Disp
from ipywidgets import widgets

from pathlib import Path

import h5py


def load_video(vid_to_set, frame_to_set, videos):
    """
    loads the chosen video and returns chosen frame

    Args:
        vidToSet (int): index of vid to load
        frameToSet (int): index of frame to load
        path_vid_allFiles (list): list of all video paths

    Returns:
        frame (cv2.image): frame read using cv2
    """
    path_vid = videos[vid_to_set - 1]
    vid_reader = cv2.VideoCapture(path_vid)

    vid_reader.set(1, frame_to_set)
    ok, frame = vid_reader.read()
    return frame


def get_bbox(mask_frame):
    """
    get rectangular bounding box for irregular roi

    Args:
        mask_frame (np.ndarray): the frame containing the mask

    Returns:
        bbox (np.ndarray): numpy array containing the indexes of the bounding box
    """
    bbox = np.zeros(4)
    bbox[0] = np.min(np.where(np.max(mask_frame, axis=0)))  # x top left
    bbox[1] = np.min(np.where(np.max(mask_frame, axis=1)))  # y top left
    bbox[2] = np.max(np.where(np.max(mask_frame, axis=0))) - bbox[0]  # x size
    bbox[3] = np.max(np.where(np.max(mask_frame, axis=1))) - bbox[1]  # y size
    bbox = np.int64(bbox)
    return bbox

class BBoxSelect:

    def __init__(self, im):
        self.im = im
        self.selected_points = []
        self.fig, ax = plt.subplots()
        self.img = ax.imshow(self.im.copy())
        self.ka = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        disconnect_button = widgets.Button(description="Confirm ROI")
        Disp.display(disconnect_button)
        disconnect_button.on_click(self.disconnect_mpl)

    def poly_img(self, img, pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True,
                      (40, 240, 240, 2),2)
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
    """
    loads a interactive tool to set the roi in the user's window

    Args:
        config_filepath (str): path to config file

    Returns:
        frame (cv2.image): frame read using cv2
        BBoxSelect(frame) (BBoxSelect): a bounding box selection object
    """

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
    """
    saves a set of points derived from a bounding box drawn by a user

    Args:
        config_filepath (str): path to config file
        frame (cv2.image) : current frame being analyzed
        bs (BBoxSelect): a completed bbox select object

    Returns:

    """
    pts = bs.selected_points
    mask_frame = np.zeros((frame.shape[0], frame.shape[1]))
    pts_y, pts_x = skimage.draw.polygon(np.array(pts)[:, 1], np.array(pts)[:, 0])
    mask_frame[pts_y, pts_x] = 1

    bbox = get_bbox(mask_frame)
    bbox_subframe_displacement = bbox
    pts_displacement, pts_x_displacement, pts_y_displacement = pts, pts_x, pts_y
    mask_frame_displacement = mask_frame
    cv2.destroyAllWindows()

    config = helpers.load_config(config_filepath)
    general = config['General']
    roi = config['ROI']
    video = config['Video']
    session = general['sessions'][roi['session_to_set']]
    helpers.create_nwb_group(session['nwb'], 'Original Points')

    pts_all = dict([
        ('bbox_subframe_displacement', np.array(bbox_subframe_displacement)),
        ('pts_displacement', np.array(pts_displacement)),
        ('pts_x_displacement', np.array(pts_x_displacement)),
        ('pts_y_displacement', np.array(pts_y_displacement)),
        ('mask_frame_displacement', np.array(mask_frame_displacement))
    ])
    for point_name, points in pts_all.items():
        helpers.create_nwb_ts(session['nwb'], 'Original Points', point_name, points, video['Fs'])

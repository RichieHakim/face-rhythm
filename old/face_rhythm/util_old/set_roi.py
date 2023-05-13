from face_rhythm.util_old import helpers
import cv2
import numpy as np
import skimage.draw

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.path
import IPython.display as Disp
from ipywidgets import widgets

from pathlib import Path
import copy

from face_rhythm.util_old import helpers

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
        self.completed_status = False
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
        self.completed_status = True


def get_roi(config_filepath):
    """
    Loads a interactive tool to set the roi in the user's window
    Update 20210919: New feature allows user to load an ROI from
     a previous session .nwb file. In order to accomoate this,
     the return of this function is either the BBoxSelect object
     OR the pts_all dictionary

    Args:
        config_filepath (str): path to config file

    Returns:
        frame (cv2.image): frame read using cv2
        
        EITHER 1 of the 2 following:
            BBoxSelect(frame) (BBoxSelect): a bounding box selection object
            pts_all (dict): dictionary of the point variables
    """

    config = helpers.load_config(config_filepath)
    roi = config['ROI']
    paths = config['Paths']
    general = config['General']

    video_list = general['sessions'][roi['session_to_set']]['videos']
    frame = load_video(roi['vid_to_set'], roi['frame_to_set'], video_list)

    if roi['load_from_file']:
        bbox_subframe_displacement = helpers.load_nwb_ts(config['ROI']['path_to_oldNWB'], 'Original Points', 'bbox_subframe_displacement')
        mask_frame_displacement = helpers.load_nwb_ts(config['ROI']['path_to_oldNWB'], 'Original Points', 'mask_frame_displacement')
        pts_displacement = helpers.load_nwb_ts(config['ROI']['path_to_oldNWB'], 'Original Points', 'pts_displacement')
        pts_x_displacement = helpers.load_nwb_ts(config['ROI']['path_to_oldNWB'], 'Original Points', 'pts_x_displacement')
        pts_y_displacement = helpers.load_nwb_ts(config['ROI']['path_to_oldNWB'], 'Original Points', 'pts_y_displacement')
        
        pts_all = dict([
                ('bbox_subframe_displacement', np.array(bbox_subframe_displacement)),
                ('pts_displacement', np.array(pts_displacement)),
                ('pts_x_displacement', np.array(pts_x_displacement)),
                ('pts_y_displacement', np.array(pts_y_displacement)),
                ('mask_frame_displacement', np.array(mask_frame_displacement))
                ])
        return frame, pts_all
    else:
        return frame, BBoxSelect(frame)


def save_roi(config_filepath, frame, bs_OR_pts_all):
    """
    saves a set of points derived from a bounding box drawn by a user

    Args:
        config_filepath (str): path to config file
        frame (cv2.image) : current frame being analyzed
        bs_OR_pts_all (dict): EITHER BBoxSelect(frame) OR pts_all

    Returns:

    """

    if isinstance(bs_OR_pts_all, BBoxSelect):
        bs = bs_OR_pts_all
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
                ('bbox_subframe_displacement', np.array(bbox_subframe_displacement)),
                ('pts_displacement', np.array(pts_displacement)),
                ('pts_x_displacement', np.array(pts_x_displacement)),
                ('pts_y_displacement', np.array(pts_y_displacement)),
                ('mask_frame_displacement', np.array(mask_frame_displacement))
                ])
    else:
        pts_all = bs_OR_pts_all
        bbox_subframe_displacement = pts_all['bbox_subframe_displacement']
        pts_displacement = pts_all['pts_displacement']
        pts_x_displacement = pts_all['pts_x_displacement']
        pts_y_displacement = pts_all['pts_y_displacement']
        mask_frame_displacement = pts_all['mask_frame_displacement']

    config = helpers.load_config(config_filepath)
    general = config['General']
    roi = config['ROI']
    video = config['Video']
    session = general['sessions'][roi['session_to_set']]

    fig, ax = plt.subplots()
    verts = [(ii[0], ii[1]) for ii in pts_displacement]
    codes = [matplotlib.path.Path.MOVETO] + [matplotlib.path.Path.LINETO] * (len(verts) - 2) + [matplotlib.path.Path.CLOSEPOLY]
    path = matplotlib.path.Path(verts, codes)
    patch = matplotlib.patches.PathPatch(path, facecolor='none', edgecolor='c', lw=2)
    ax.add_patch(patch)
    ax.imshow(frame)

    print(session.keys())
    helpers.save_pts(session['nwb'], pts_all)
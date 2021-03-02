import numpy as np
import copy
from face_rhythm.util import helpers

import cv2
import imageio

from pathlib import Path
from tqdm.notebook import tqdm, trange


def create_frame(config, session, frame, point_inds_tracked_list, color_tuples, counters):
    """
    creates a single frame of points overlayed on a video 
    returns that frame to be displayed or saved in a higher level function

    Parameters
    ----------
    config (dict): dictionary of config parameters
    session (): 
    frame ():
    points ():
    color_tuples ():
    counters ():

    Returns
    -------
    frame (cv2.image): labeled and processed image with points
    """
    dot_size = config['Optic']['dot_size']
    vidNums_toUse = config['Optic']['vidNums_toUse']
    numFrames_total_rough = session['frames_total']
    iter_frame, vidNum_iter, ind_concat, fps = counters
    point_inds_tracked, point_inds_tracked_tuple = point_inds_tracked_list
    numFrames_rough = session['vid_lens'][vidNum_iter]

    for ii in range(point_inds_tracked.shape[0]):
        point_inds_tracked_tuple[ii] = tuple(np.int64(np.squeeze(point_inds_tracked[ii, 0, :])))
        cv2.circle(frame, point_inds_tracked_tuple[ii], dot_size, color_tuples[ii], -1)

    cv2.putText(frame, f'frame #: {iter_frame}/{numFrames_rough}-ish', org=(10, 20), fontFace=1, fontScale=1,
                color=(255, 255, 255), thickness=1)
    cv2.putText(frame, f'vid #: {vidNum_iter + 1}/{len(vidNums_toUse)}', org=(10, 40), fontFace=1, fontScale=1,
                color=(255, 255, 255), thickness=1)
    cv2.putText(frame, f'total frame #: {ind_concat + 1}/{numFrames_total_rough}-ish', org=(10, 60), fontFace=1,
                fontScale=1, color=(255, 255, 255), thickness=1)
    cv2.putText(frame, f'fps: {np.uint32(fps)}', org=(10, 80), fontFace=1, fontScale=1, color=(255, 255, 255),
                thickness=1)

    return frame
        

def visualize_progress(config, session, frame, point_inds_tracked_list, color_tuples, counters, out):
    remote = config['General']['remote']
    frame_labeled = create_frame(config, session, frame, point_inds_tracked_list, color_tuples, counters)

    if remote:
        out.write(frame_labeled)
    else:
        cv2.imshow('test', frame_labeled)


def visualize_points(config_filepath):
    config = helpers.load_config(config_filepath)
    general = config['General']
    video = config['Video']
    optic = config['Optic']

    save_pathFull = str(Path(config['Paths']['viz']) / f'{video["data_to_display"]}_demo.avi')
    color_tuples = helpers.load_data(config_filepath, 'color_tuples')

    demo_len = video['demo_len']
    vid_width = video['width']
    vid_height = video['height']
    Fs = video['Fs']

    if general['remote'] or video['save_demo']:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(f'saving to file {save_pathFull}')
        out = cv2.VideoWriter(save_pathFull, fourcc, Fs, (np.int64(vid_width), np.int64(vid_height)))
    else:
        out = None

    for session in general['sessions']:
        positions = helpers.load_nwb_ts(session['nwb'],'Optic Flow',video['data_to_display'])
        position_tuples = list(np.arange(positions.shape[0]))
        ind_concat = 0
        for vid_num in optic['vidNums_toUse']:
            vid = imageio.get_reader(session['videos'][vid_num], 'ffmpeg')
            for iter_frame in trange(demo_len):
                new_frame = vid.get_data(iter_frame)
                points = [positions[:,np.newaxis,:,iter_frame],position_tuples]
                counters = [iter_frame, vid_num, ind_concat, Fs]
                visualize_progress(config, session, new_frame, points, color_tuples, counters, out)

                k = cv2.waitKey(1) & 0xff
                if k == 27 : break
            ind_concat += 1

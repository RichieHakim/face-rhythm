import sys

import numpy as np
# if sys.platform == 'darwin':
#     import matplotlib
#     matplotlib.use('TKAgg')
from matplotlib import pyplot as plt
from scipy.stats import norm
import colorsys

import cv2
import imageio

from pathlib import Path
from tqdm.notebook import trange

from face_rhythm.util import helpers
from face_rhythm.visualize.make_custom_cmap import make_custom_cmap



def create_frame(config, session, frame, point_inds_tracked_list, color_tuples, counters, factor_num = 0):
    """
    creates a single frame of points overlayed on a video 
    returns that frame to be displayed or saved in a higher level function

    Args:
        config (dict): dictionary of config parameters
        session (dict): dictionary of session level data
        frame (cv2.image): current frame to display
        points_inds_tracked_list (list): list of two containers of points to display
        color_tuples (list): list of color tuples
        counters (list): list of counters that track various stats about the video

    Returns:
        frame (cv2.image): labeled and processed image with points
    """

    dot_size = config['Video']['dot_size']
    vidNums_toUse = config['Optic']['vidNums_toUse']
    numFrames_total_rough = session['frames_total']
    iter_frame, vidNum_iter, ind_concat, fps = counters
    point_inds_tracked, point_inds_tracked_tuple = point_inds_tracked_list
    numFrames_rough = session['vid_lens'][vidNum_iter]

    for ii in range(point_inds_tracked.shape[0]):
        point_inds_tracked_tuple[ii] = tuple(np.int64(np.squeeze(point_inds_tracked[ii, 0, :])))
        cv2.circle(frame, point_inds_tracked_tuple[ii], dot_size, color_tuples[ii], -1)

    cv2.putText(frame, f'frame #: {iter_frame}/~{numFrames_rough}', org=(10, 20), fontFace=1, fontScale=1,
                color=(255, 255, 255), thickness=1)
    cv2.putText(frame, f'vid #: {vidNum_iter + 1}/{len(vidNums_toUse)}', org=(10, 40), fontFace=1, fontScale=1,
                color=(255, 255, 255), thickness=1)
    cv2.putText(frame, f'total frame #: {ind_concat + 1}/~{numFrames_total_rough}', org=(10, 60), fontFace=1,
                fontScale=1, color=(255, 255, 255), thickness=1)
    cv2.putText(frame, f'fps: {np.uint32(fps)}', org=(10, 80), fontFace=1, fontScale=1, color=(255, 255, 255),
                thickness=1)
    if factor_num:
        cv2.putText(frame, f'factor: {factor_num}', org=(10, 100), fontFace=1, fontScale=1, color=(255, 255, 255),
                thickness=1)

    return frame
        

def visualize_progress(config, session, frame, point_inds_tracked_list, color_tuples, counters, out):
    """
    gets frame and then saves ior displays it

    Args:
        config (dict): dictionary of config parameters
        session (dict): dictionary of session level data
        frame (cv2.image): current frame to display
        points_inds_tracked_list (list): list of two containers of points to display
        color_tuples (list): list of color tuples
        counters (list): list of counters that track various stats about the video
        out (cv2.fileinterface): where to write the frames

    Returns:

    """

    frame_labeled = create_frame(config, session, frame, point_inds_tracked_list, color_tuples, counters)

    if config['General']['remote'] or (config['Video']['save_demo'] and counters[2] < config['Video']['demo_len']):
        out.write(frame_labeled)
    if not config['General']['remote']:
        cv2.imshow('Display Factors', frame_labeled)


def visualize_points(config_filepath):
    """
    loops over all sessions and creates a short demo video from each session

    Args:
        config_filepath (Path): path to current config file

    Returns:

    """

    config = helpers.load_config(config_filepath)
    general = config['General']
    video = config['Video']
    optic = config['Optic']

    demo_len = video['demo_len']
    vid_width = video['width']
    vid_height = video['height']
    Fs = video['Fs']

    for session in general['sessions']:
        color_tuples = helpers.load_nwb_ts(session['nwb'],'Optic Flow', 'color_tuples')
        save_pathFull = str(Path(video['demos']) / f'{session["name"]}_{video["data_to_display"]}_run{general['run_name']}_demo.avi')

        if general['remote'] or video['save_demo']:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            print(f'saving to file {save_pathFull}')
            out = cv2.VideoWriter(save_pathFull, fourcc, Fs, (np.int64(vid_width), np.int64(vid_height)))
        else:
            out = None

        positions = helpers.load_nwb_ts(session['nwb'],'Optic Flow',video['data_to_display'])
        position_tuples = list(np.arange(positions.shape[0]))
        ind_concat = 0
        for vid_num in optic['vidNums_toUse']:
            vid = imageio.get_reader(session['videos'][vid_num], 'ffmpeg')
            for iter_frame in trange(demo_len):
                new_frame = vid.get_data(iter_frame)
                absolute_ind = helpers.absolute_index(session, vid_num, iter_frame)
                points = [positions[:,np.newaxis,:,absolute_ind],position_tuples]
                counters = [iter_frame, vid_num, ind_concat, Fs]
                visualize_progress(config, session, new_frame, points, color_tuples, counters, out)

                k = cv2.waitKey(1) & 0xff
                if k == 27 : break

                ind_concat += 1

        if general['remote'] or video['save_demo']:
            out.release()


def visualize_factor(config_filepath):
    """
    creates videos of the points colored by their positional factor values

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """

    config = helpers.load_config(config_filepath)
    general = config['General']
    video = config['Video']
    optic = config['Optic']

    demo_len = video['demo_len']
    vid_width = video['width']
    vid_height = video['height']
    Fs = video['Fs']

    factor_category_name = video['factor_category_to_display']
    factor_name = video['factor_to_display']
    points_name = video['points_to_display']


    for session in general['sessions']:
        factor = helpers.load_nwb_ts(session['nwb'], factor_category_name, factor_name)
        factor_x, factor_y = np.array_split(factor, 2, axis=0)

        points = helpers.load_nwb_ts(session['nwb'],'Optic Flow', points_name)
        points_tuples = list(np.arange(points.shape[0]))
        if factor_category_name == 'PCA':
            rank = config['PCA']['n_factors_to_show']
        else:
            rank = factor.shape[1]

        for factor_iter in range(rank):
            save_path = str(Path(config['Paths']['viz']) / (factor_category_name + '__' 
            + factor_name + '__' + points_name + '__' + f'factor_{factor_iter+1}__run{general['run_name']}.avi'))

            if general['remote'] or video['save_demo']:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                print(f'saving to file {save_path}')
                out = cv2.VideoWriter(save_path, fourcc, Fs, (np.int64(vid_width), np.int64(vid_height)))
            else:
                out = None

            factor_toShow = factor_iter

            offset_toUse = np.min(factor[:,factor_toShow])
            scores = factor[:, factor_toShow] - offset_toUse
            # scores_norm = (scores_norm / np.max(scores_norm)) * (1/np.sqrt(2))
            scores_x, scores_y = np.array_split(scores,2)
            scores_complex = scores_x + scores_y*1j
            scores_angle = (np.angle(scores_complex, deg=True) -45)/45
            scores_mag = np.abs(scores_complex)
            scores_mag = (scores_mag/np.max(scores_mag))**(1/1) *1
            scores_mag[scores_mag > 1] = 1

            numColors = 256
            # cmap = matplotlib.cm.get_cmap('brg', numColors)
            cmap = make_custom_cmap()
            print(cmap.shape)


            color_tuples = list(np.arange(points.shape[0]))
            for ii in range(points.shape[0]):
                cmap_idx_toUse = np.int64(np.ceil(scores_angle[ii]*numColors/2 + numColors/2)) - 1
                # color_tuples[ii] = list(np.flip((np.array(cmap(cmap_idx_toUse)))[:3]) * scores_mag[ii] * 255*1)
                color_tuples[ii] = list(np.flip((np.array(cmap[cmap_idx_toUse]))) * scores_mag[ii] * 1*1)

            ind_concat = 0
            for vid_num in optic['vidNums_toUse']:
                vid = imageio.get_reader(session['videos'][vid_num], 'ffmpeg')
                # for iter_frame in trange(demo_len):
                for iter_frame, new_frame in enumerate(vid):
                    # new_frame = vid.get_data(iter_frame)
                    points_tracked = [points[:, np.newaxis, :, ind_concat], points_tuples]
                    counters = [iter_frame, vid_num, ind_concat, Fs]
                    visualize_progress(config, session, new_frame, points_tracked, color_tuples, counters, out)

                    k = cv2.waitKey(1) & 0xff
                    if k == 27: break

                    ind_concat += 1

                    if ind_concat >= demo_len:
                        break
                if ind_concat >= demo_len:
                    break

            if general['remote'] or video['save_demo']:
                out.release()

    cv2.destroyAllWindows()


class TemporalTrace(object):
    def __init__(self, factor_temporal, start, end, width, fps, fig_width, fig_height):
        self.start = start
        self.width = int(width)
        self.fps = int(fps)
        self.fig = plt.figure(figsize=(fig_width / 100, fig_height / 100), dpi=100)
        plt.plot(factor_temporal)
        self.line = plt.axvline(x=self.start)
        self.left_limit = self.start - self.width / 2
        self.right_limit = self.start + self.width / 2
        plt.xlim(self.left_limit, self.right_limit)
        ticks = np.arange(self.left_limit, self.right_limit + self.fps, self.fps)
        self.labels = ['{:.1f}'.format(x) for x in
                       np.arange(-self.width / self.fps / 2, self.width / self.fps / 2 + 1, 1)]
        plt.xticks(ticks, self.labels)
        plt.xlabel('time (s)')
        plt.yticks(())
        self.fig.canvas.draw()

    def update(self, nn):
        left_limit = self.left_limit + nn
        right_limit = self.right_limit + nn
        plt.xlim(left_limit, right_limit)
        ticks = np.arange(left_limit, right_limit + self.fps, self.fps)
        plt.xticks(ticks, self.labels)
        self.line.set_data([(self.start + nn)] * 100, np.linspace(0, 1, 100))
        self.fig.canvas.draw()


def fig_to_cv2_image(fig):
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def brighten_colors(cmap, alpha):
    cmap_hsv = np.array([colorsys.rgb_to_hsv(*[c / 255 for c in rgb]) for rgb in cmap])
    cmap_hsv[:, -1] *= alpha
    cmap_brighter = [[255 * c for c in colorsys.hsv_to_rgb(*hsv)] for hsv in cmap_hsv]
    return cmap_brighter


def minmax(data):
    return (data - min(data)) / (max(data) - min(data))


def face_with_trace(config_filepath):
    """
    creates videos of the points colored by their positional factor values

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """

    config = helpers.load_config(config_filepath)
    general = config['General']
    video = config['Video']
    optic = config['Optic']

    start_vid = video['start_vid']
    start_frame = video['start_frame']
    demo_len = video['demo_len']
    vid_width = video['width']
    vid_height = video['height']
    Fs = video['Fs']
    factors_to_show = video['factors_to_show']
    show_alpha = video['show_alpha']
    pulse_ind = video['pulse_test_index']

    factor_category_name = video['factor_category_to_display']
    factor_name = video['factor_to_display']
    points_name = video['points_to_display']

    for session in general['sessions']:
        factor = helpers.load_nwb_ts(session['nwb'], factor_category_name, factor_name)
        factor_temp = helpers.load_nwb_ts(session['nwb'], factor_category_name, 'factors_spectral_temporal_interp')
        factor_x, factor_y = np.array_split(factor, 2, axis=0)

        points = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', points_name)
        points_tuples = list(np.arange(points.shape[0]))
        if factor_category_name == 'PCA':
            rank = range(config['PCA']['n_factors_to_show'])
        elif not factors_to_show:
            rank = range(factor.shape[1])
        else:
            rank = [factor - 1 for factor in factors_to_show]

        for factor_iter in rank:
            save_path = str(Path(config['Paths']['viz']) / (factor_category_name + '__'
                                                            + factor_name + '__' + points_name + '__' + f'factor_temporal_{factor_iter + 1}_run{general['run_name']}.avi'))

            if general['remote'] or video['save_demo']:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                print(f'saving to file {save_path}')
                out = cv2.VideoWriter(save_path, fourcc, Fs, (np.int64(2 * vid_width), np.int64(vid_height)))
            else:
                out = None

            factor_toShow = factor_iter

            offset_toUse = np.min(factor[:, factor_toShow])
            scores = factor[:, factor_toShow] - offset_toUse
            # scores_norm = (scores_norm / np.max(scores_norm)) * (1/np.sqrt(2))
            scores_x, scores_y = np.array_split(scores, 2)
            scores_complex = scores_x + scores_y * 1j
            scores_angle = (np.angle(scores_complex, deg=True) - 45) / 45
            scores_mag = np.abs(scores_complex)
            scores_mag = (scores_mag / np.max(scores_mag)) ** (1 / 1) * 1
            scores_mag[scores_mag > 1] = 1

            numColors = 256
            # cmap = matplotlib.cm.get_cmap('brg', numColors)
            cmap = make_custom_cmap()

            color_tuples = list(np.arange(points.shape[0]))
            for ii in range(points.shape[0]):
                cmap_idx_toUse = np.int64(np.ceil(scores_angle[ii] * numColors / 2 + numColors / 2)) - 1
                # color_tuples[ii] = list(np.flip((np.array(cmap(cmap_idx_toUse)))[:3]) * scores_mag[ii] * 255*1)
                color_tuples[ii] = list(np.flip((np.array(cmap[cmap_idx_toUse]))) * scores_mag[ii] * 1 * 1)

            # ADDING TEMPORAL TRACE
            current_trace = factor_temp[:, factor_iter]
            alphas = minmax(current_trace) + 0.5
            if pulse_ind:
                dist = norm(loc=pulse_ind, scale=2)
                current_trace += dist.pdf(range(current_trace.shape[0]))

            vidLens_toUse = [session['vid_lens'][vidnum] for vidnum in optic['vidNums_toUse']]
            ind_concat = int(sum(vidLens_toUse[:start_vid])) + start_frame
            ind_loop = 0
            trace_plot = TemporalTrace(current_trace, ind_concat, current_trace.shape[0], Fs * 5, Fs, vid_width,
                                       vid_height)
            first_vid_ind = optic['vidNums_toUse'].index(start_vid)

            for vid_num in optic['vidNums_toUse'][first_vid_ind:]:
                vid = imageio.get_reader(session['videos'][vid_num], 'ffmpeg')
                for iter_frame in range(start_frame, int(session['vid_lens'][vid_num])):
                    new_frame = vid.get_data(iter_frame)
                    if pulse_ind and ind_concat == pulse_ind:
                        new_frame = np.zeros_like(new_frame)
                    points_tracked = [points[:, np.newaxis, :, ind_concat], points_tuples]
                    counters = [iter_frame, vid_num, ind_concat, Fs]

                    color_tuples_brightened = brighten_colors(color_tuples,
                                                              alphas[ind_concat]) if show_alpha else color_tuples
                    frame_labeled = create_frame(config, session, new_frame, points_tracked, color_tuples_brightened,
                                                 counters, factor_iter+1)
                    trace_im = fig_to_cv2_image(trace_plot.fig)

                    to_write = np.concatenate((frame_labeled, trace_im), axis=1)
                    if config['General']['remote'] or (
                            config['Video']['save_demo'] and ind_loop < config['Video']['demo_len']):
                        out.write(to_write)

                    k = cv2.waitKey(1) & 0xff
                    if k == 27: break

                    ind_concat += 1
                    ind_loop += 1
                    trace_plot.update(ind_loop)

                    if ind_loop >= demo_len or ind_concat >= current_trace.shape[0]:
                        break
                if ind_loop >= demo_len or ind_concat >= current_trace.shape[0]:
                    break

            if general['remote'] or video['save_demo']:
                out.release()
    cv2.destroyAllWindows()


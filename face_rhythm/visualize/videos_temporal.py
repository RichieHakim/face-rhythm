import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import cv2
import imageio

from pathlib import Path
from tqdm.notebook import trange

from face_rhythm.util import helpers
from face_rhythm.visualize.make_custom_cmap import make_custom_cmap
from face_rhythm.visualize import videos as frvideos

import ipdb

class TemporalTrace(object):
    def __init__(self, factor_temporal, start, end, width, fps, fig_width, fig_height):
        self.start = start
        self.width = int(width)
        self.fps = int(fps)
        self.fig = plt.figure(figsize=(fig_width/100, fig_height/100),dpi=100)
        plt.plot(factor_temporal[self.start:end])
        self.line = plt.axvline(x=self.start)
        self.left_limit = self.start-self.width/2
        self.right_limit = self.start+self.width/2
        plt.xlim(self.left_limit,self.right_limit)
        ticks = np.arange(self.left_limit,self.right_limit+self.fps,self.fps)
        self.labels = ['{:.1f}'.format(x) for x in np.arange(-self.width/self.fps/2,self.width/self.fps/2+1,1)]
        plt.xticks(ticks,self.labels)
        plt.xlabel('time (s)')
        plt.yticks(())
        self.fig.canvas.draw()
    
    def update(self, n):
        left_limit = self.left_limit + n
        right_limit = self.right_limit + n
        plt.xlim(left_limit,right_limit)
        ticks = np.arange(left_limit,right_limit+self.fps,self.fps)
        plt.xticks(ticks, self.labels)
        self.line.set_data([(self.start+n)]*100,np.linspace(0,1,100))
        self.fig.canvas.draw()
        
def fig_to_cv2_image(fig):
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

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

    factor_category_name = video['factor_category_to_display']
    factor_name = video['factor_to_display']
    points_name = video['points_to_display']


    for session in general['sessions']:
        factor = helpers.load_nwb_ts(session['nwb'], factor_category_name, factor_name)
        factor_temp = helpers.load_nwb_ts(session['nwb'], factor_category_name, 'factors_spectral_temporal_interp')
        factor_x, factor_y = np.array_split(factor, 2, axis=0)

        points = helpers.load_nwb_ts(session['nwb'],'Optic Flow', points_name)
        points_tuples = list(np.arange(points.shape[0]))
        if factor_category_name == 'PCA':
            rank = config['PCA']['n_factors_to_show']
        else:
            rank = factor.shape[1]

        for factor_iter in range(rank):
            save_path = str(Path(config['Paths']['viz']) / (factor_category_name + '__' 
            + factor_name + '__' + points_name + '__' + f'factor_temporal_{factor_iter+1}.avi'))

            if general['remote'] or video['save_demo']:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                print(f'saving to file {save_path}')
                out = cv2.VideoWriter(save_path, fourcc, Fs, (np.int64(2*vid_width), np.int64(vid_height)))
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
            
            # ADDING TEMPORAL TRACE
            current_trace = factor_temp[:,factor_iter]
            
            vidLens_toUse = [session['vid_lens'][vidnum] for vidnum in optic['vidNums_toUse']]
            ind_concat = int(sum(vidLens_toUse[:start_vid]))
            ind_loop = 0 
            trace_plot = TemporalTrace(current_trace,ind_concat,current_trace.shape[0],Fs*5,Fs,vid_width,vid_height)
            first_vid_ind = optic['vidNums_toUse'].index(start_vid)
            
            for vid_num in optic['vidNums_toUse'][first_vid_ind:]:
                vid = imageio.get_reader(session['videos'][vid_num], 'ffmpeg')
                for iter_frame in range(int(session['vid_lens'][vid_num])):
                    new_frame = vid.get_data(iter_frame)
                    points_tracked = [points[:, np.newaxis, :, ind_concat], points_tuples]
                    counters = [iter_frame, vid_num, ind_concat, Fs]
                    
                    frame_labeled = frvideos.create_frame(config, session, new_frame, points_tracked, color_tuples, counters)
                    trace_im = fig_to_cv2_image(trace_plot.fig)
                    
                    to_write = np.concatenate((frame_labeled,trace_im),axis=1)
                    if config['General']['remote'] or (config['Video']['save_demo'] and ind_loop < config['Video']['demo_len']):
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
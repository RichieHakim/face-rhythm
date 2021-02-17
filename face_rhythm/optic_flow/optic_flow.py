import time
import numpy as np
import copy
from face_rhythm.util import helpers

import cv2
import imageio

import multiprocessing
from multiprocessing import Pool, RLock, freeze_support
from functools import partial

from tqdm.notebook import tqdm

from pathlib import Path


def setup(config, pts_all):
    """
    initializes the arrays for optic flow analysis

    Parameters
    ----------
    config (dict): dictionary of config parameters
    pts_all (dict): dictionary of important points

    Returns
    -------
    pointInds_toUse
    pointInds_tracked
    pointInds_tracked_tuple
    displacements
    pts_spaced
    color_tuples
    """
    numFrames_total_rough = config['numFrames_total_rough']
    numVids = config['numVids']
    spacing = config['spacing']

    bbox_subframe_displacement = pts_all['bbox_subframe_displacement']
    pts_displacement = pts_all['pts_displacement']
    pts_x_displacement = pts_all['pts_x_displacement']
    pts_y_displacement = pts_all['pts_y_displacement']
    mask_frame_displacement = pts_all['mask_frame_displacement']

    ## Make point cloud
    pts_spaced = np.ones((np.int64(bbox_subframe_displacement[3] * bbox_subframe_displacement[2] / spacing),
                          2)) * np.nan  ## preallocation
    cc = 0  ## set idx counter

    # make spaced out points
    for ii in range(len(pts_x_displacement)):
        if (pts_x_displacement[ii] % spacing == 0) and (pts_y_displacement[ii] % spacing == 0):
            pts_spaced[cc, 0] = pts_x_displacement[ii]
            pts_spaced[cc, 1] = pts_y_displacement[ii]
            cc = cc + 1

    pts_spaced = np.expand_dims(pts_spaced, 1).astype('single')
    pts_spaced = np.delete(pts_spaced, np.where(np.isnan(pts_spaced[:, 0, 0])), axis=0)
    print(f'number of points: {pts_spaced.shape[0]}')

    ## Define random colors for points in cloud
    color_tuples = list(np.arange(len(pts_x_displacement)))
    for ii in range(len(pts_x_displacement)):
        color_tuples[ii] = (np.random.rand(1)[0] * 255, np.random.rand(1)[0] * 255, np.random.rand(1)[0] * 255)

    ## Preallocate output variables

    # I add a bunch of NaNs to the end because the openCV estimate is usually less than the actual number of frames
    displacements = np.ones((pts_spaced.shape[0], 2, np.uint64(
        numFrames_total_rough + numFrames_total_rough * 0.1 + (numVids * 1000)))) * np.nan

    ## Preset point tracking variables
    pointInds_toUse = copy.deepcopy(pts_spaced)
    pointInds_tracked = pointInds_toUse  ## set the first frame to have point locations be positions in the point cloud
    pointInds_tracked_tuple = list(np.arange(pointInds_toUse.shape[0]))

    return pointInds_toUse, pointInds_tracked, pointInds_tracked_tuple, displacements, pts_spaced, color_tuples


def visualize_progress(config, new_frame, pointInds_tracked, pointInds_tracked_tuple, color_tuples, counters, numFrames_rough, out, test_len):
    """
    plots a checkup

    Parameters
    ----------
    config (dict): dictionary of config parameters
    new_frame ():
    pointInds_tracked ():
    pointInds_tracked_tuple ():
    color_tuples ():
    counters ():

    Returns
    -------

    """
    dot_size = config['dot_size']
    vidNums_toUse = config['vidNums_toUse']
    numFrames_total_rough = config['numFrames_total_rough']
    remote = config['remote']

    iter_frame, vidNum_iter, ind_concat, fps = counters

    for ii in range(pointInds_tracked.shape[0]):
        pointInds_tracked_tuple[ii] = tuple(np.int64(np.squeeze(pointInds_tracked[ii,0,:])))
        cv2.circle(new_frame, pointInds_tracked_tuple[ii], dot_size, color_tuples[ii], -1)

    cv2.putText(new_frame, f'frame #: {iter_frame}/{numFrames_rough}-ish', org=(10, 20), fontFace=1, fontScale=1,
                color=(255, 255, 255), thickness=1)
    cv2.putText(new_frame, f'vid #: {vidNum_iter + 1}/{len(vidNums_toUse)}', org=(10, 40), fontFace=1, fontScale=1,
                color=(255, 255, 255), thickness=1)
    cv2.putText(new_frame, f'total frame #: {ind_concat + 1}/{numFrames_total_rough}-ish', org=(10, 60), fontFace=1,
                fontScale=1, color=(255, 255, 255), thickness=1)
    cv2.putText(new_frame, f'fps: {np.uint32(fps)}', org=(10, 80), fontFace=1, fontScale=1, color=(255, 255, 255),
                thickness=1)

    if remote and iter_frame < test_len:
        out.write(new_frame)
    else:
        cv2.imshow('test', new_frame)


def displacements_monothread(config, pointInds_toUse, pointInds_tracked, pointInds_tracked_tuple, displacements,
                             pts_spaced, color_tuples):
    """
    the workhorse of the optic flow
    Opens each video in the list of videos
    Iterates through the frames of the video
    Computes the optic flow between frames
    Saves this to displacements frame

    Parameters
    ----------
    config (dict): dictionary of config parameters
    pointInds_toUse ():
    pointInds_tracked ():
    pointInds_tracked_tuple ():
    displacements ():
    pts_spaced ():

    Returns
    -------
    displacements (): array of displacements
    numFrames_total (int): number of frames
    """

    ## Main loop to pull out displacements in each video
    ind_concat = 0
    fps = 0
    tic_fps = time.time()
    tic_all = time.time()

    vidNums_toUse = config['vidNums_toUse']
    path_vid_allFiles = config['path_vid_allFiles']
    numVids = config['numVids']
    showVideo_pref = config['showVideo_pref']
    fps_counterPeriod = config['fps_counterPeriod']
    printFPS_pref = config['printFPS_pref']

    remote = config['remote']
    Fs = config['vid_Fs']
    vid_width = config['vid_width']
    vid_height = config['vid_height']
    test_len = config['test_len']
    save_pathFull = str(Path(config['path_project']) / 'optic_test.avi')

    lk_names = [key for key in config.keys() if 'lk_' in key]
    lk_params = {k.split('lk_')[1]: (tuple(config[k]) if type(config[k]) is list else config[k]) \
                 for k in lk_names}

    # Define the codec and create VideoWriter object
    if showVideo_pref and remote:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(f'saving to file {save_pathFull}')
        out = cv2.VideoWriter(save_pathFull, fourcc, Fs, (np.int64(vid_width), np.int64(vid_height)))
    else:
        out = None

    for vidNum_iter in vidNums_toUse:
        vid = imageio.get_reader(path_vid_allFiles[vidNum_iter], 'ffmpeg')
        #     metadata = vid.get_meta_data()

        path_vid = path_vid_allFiles[vidNum_iter]  # get path of the current vid
        video = cv2.VideoCapture(path_vid)  # open the video object with openCV
        numFrames_rough = int(video.get(
            cv2.CAP_PROP_FRAME_COUNT))  # get frame count of this vid GENERALLY INACCURATE. OFF BY AROUND -25 frames

        frameToSet = 0
        frame = vid.get_data(0)  # Get a single frame to use as the first 'previous frame' in calculating optic flow
        new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        old_frame = new_frame_gray

        print(f'\n Calculating displacement field: video # {vidNum_iter + 1}/{numVids}')
        #     while True:
        for iter_frame, new_frame in enumerate(tqdm(vid, total=numFrames_rough)):
            new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            ##calculate optical flow
            pointInds_new, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame_gray, pointInds_toUse, None,
                **lk_params)  # Calculate displacement distance between STATIC/ANCHORED points and the calculated new points. Also note the excluded 'NextPts' parameter. Could be used for fancier tracking

            ## Calculate displacement and place into variable 'displacements' (changes in size every iter)         
            if iter_frame == 0:
                displacements[:, :, ind_concat] = np.zeros((pts_spaced.shape[0], 2))
            else:
                displacements[:, :, ind_concat] = np.single(np.squeeze((
                    pointInds_new - pointInds_toUse)))  # this is the important variable. Simply the difference in the estimate

            old_frame = new_frame_gray  # make current frame the 'old_frame' for the next iteration

            ## below is just for visualization. Nothing calculated is maintained
            if showVideo_pref:
                pointInds_tracked = pointInds_tracked + (
                        pointInds_new - pointInds_toUse)  # calculate integrated position
                pointInds_tracked = pointInds_tracked - (
                        pointInds_tracked - pointInds_toUse) * 0.01  # multiplied constant is the relaxation term. this is just for display purposes. Relaxation term chosen during cleanup will be real
                counters = [iter_frame, vidNum_iter, ind_concat, fps]
                visualize_progress(config, new_frame, pointInds_tracked, pointInds_tracked_tuple, color_tuples, counters, numFrames_rough, out, test_len)
                if remote and iter_frame == test_len:
                    out.release()
                
                k = cv2.waitKey(1) & 0xff
                if k == 27: break

            ind_concat = ind_concat + 1

            if ind_concat % fps_counterPeriod == 0:
                elapsed = time.time() - tic_fps
                fps = fps_counterPeriod / elapsed
                if printFPS_pref:
                    print(fps)
                tic_fps = time.time()

        ## Calculate how long calculation took
        elapsed = time.time() - tic_all
        if elapsed < 60:
            print(f'time elapsed: {np.uint32(elapsed)} seconds. Capture rate: {round(ind_concat / elapsed, 3)} fps')
        else:
            print(f'time elapsed: {round(elapsed / 60, 3)} minutes. Capture rate: {round(ind_concat / elapsed, 3)} fps')

    numFrames_total = ind_concat - 1
    cv2.destroyAllWindows()

    displacements = displacements[:, :, ~np.isnan(displacements[0, 0, :])]

    return displacements, numFrames_total

def displacements_recursive(config, pointInds_toUse, pointInds_tracked, pointInds_tracked_tuple, displacements,
                             pts_spaced, color_tuples, relaxation_factor):
    
    """
    the workhorse of the optic flow
    Opens each video in the list of videos
    Iterates through the frames of the video
    Computes the optic flow between frames
    Saves this to displacements frame

    Parameters
    ----------
    config (dict): dictionary of config parameters
    pointInds_toUse ():
    pointInds_tracked ():
    pointInds_tracked_tuple ():
    displacements ():
    pts_spaced ():

    Returns
    -------
    displacements (): array of displacements
    numFrames_total (int): number of frames
    """

    ## Main loop to pull out displacements in each video
    ind_concat = 0
    fps = 0
    tic_fps = time.time()
    tic_all = time.time()

    vidNums_toUse = config['vidNums_toUse']
    path_vid_allFiles = config['path_vid_allFiles']
    numVids = config['numVids']
    showVideo_pref = config['showVideo_pref']
    fps_counterPeriod = config['fps_counterPeriod']
    printFPS_pref = config['printFPS_pref']

    remote = config['remote']
    Fs = config['vid_Fs']
    vid_width = config['vid_width']
    vid_height = config['vid_height']
    test_len = config['test_len']
    save_pathFull = str(Path(config['path_project']) / 'optic_test.avi')

    lk_names = [key for key in config.keys() if 'lk_' in key]
    lk_params = {k.split('lk_')[1]: (tuple(config[k]) if type(config[k]) is list else config[k]) \
                 for k in lk_names}

    # Define the codec and create VideoWriter object
    if showVideo_pref and remote:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(f'saving to file {save_pathFull}')
        out = cv2.VideoWriter(save_pathFull, fourcc, Fs, (np.int64(vid_width), np.int64(vid_height)))
    else:
        out = None

    for vidNum_iter in vidNums_toUse:
        vid = imageio.get_reader(path_vid_allFiles[vidNum_iter],  'ffmpeg')
    #     metadata = vid.get_meta_data()
            
        path_vid = path_vid_allFiles[vidNum_iter]  # get path of the current vid
        video = cv2.VideoCapture(path_vid)  # open the video object with openCV
        numFrames_rough = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # get frame count of this vid GENERALLY INACCURATE. OFF BY AROUND -25 frames

        frameToSet = 0
        frame = vid.get_data(0) # Get a single frame to use as the first 'previous frame' in calculating optic flow
        new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        old_frame = new_frame_gray
        
        
        print(f'\n Calculating displacement field: video # {vidNum_iter+1}/{numVids}')
        pointInds_old = pointInds_toUse
        for iter_frame , new_frame in enumerate(vid):
            new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            ##calculate optical flow
    #         pointInds_new, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame_gray, pointInds_toUse, None, **lk_params)  # Calculate displacement distance between STATIC/ANCHORED points and the calculated new points. Also note the excluded 'NextPts' parameter. Could be used for fancier tracking
            pointInds_new, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame_gray, np.single(pointInds_old), None, **lk_params)  # Calculate displacement distance between STATIC/ANCHORED points and the calculated new points. Also note the excluded 'NextPts' parameter. Could be used for fancier tracking

            ## Calculate displacement and place into variable 'displacements' (changes in size every iter)         
            if iter_frame ==0:
                displacements[:,:,ind_concat] = np.zeros((pts_spaced.shape[0] ,2))
            else:
    #             displacements[:,:,ind_concat] =  np.single(np.squeeze((pointInds_new - pointInds_toUse)))  # this is the important variable. Simply the difference in the estimate
                displacements[:,:,ind_concat] =  np.squeeze((pointInds_new - pointInds_old))  # this is the important variable. Simply the difference in the estimate
            
            old_frame = new_frame_gray  # make current frame the 'old_frame' for the next iteration
            
    #             pointInds_tracked = pointInds_tracked + (pointInds_new - pointInds_toUse)  # calculate integrated position
            pointInds_tracked = pointInds_tracked + displacements[:,:,ind_concat][:,None,:]  # calculate integrated position
            pointInds_tracked = pointInds_tracked - (pointInds_tracked -pointInds_toUse)*relaxation_factor  # multiplied constant is the relaxation term
            pointInds_old = pointInds_tracked
            
            ## below is just for visualization. Nothing calculated is maintained
            if showVideo_pref:
                counters = [iter_frame, vidNum_iter, ind_concat, fps]
                visualize_progress(config, new_frame, pointInds_tracked, pointInds_tracked_tuple, color_tuples, counters, numFrames_rough, out, test_len)
                if remote and iter_frame == test_len:
                    out.release()
                
                k = cv2.waitKey(1) & 0xff
                if k == 27: break

            ind_concat = ind_concat + 1
            

            if ind_concat % fps_counterPeriod == 0:
                elapsed = time.time() - tic_fps
                fps = fps_counterPeriod / elapsed
                if printFPS_pref:
                    print(fps)
                tic_fps = time.time()

        ## Calculate how long calculation took
        elapsed = time.time() - tic_all
        if elapsed < 60:
            print(f'time elapsed: {np.uint32(elapsed)} seconds. Capture rate: {round(ind_concat / elapsed, 3)} fps')
        else:
            print(f'time elapsed: {round(elapsed / 60, 3)} minutes. Capture rate: {round(ind_concat / elapsed, 3)} fps')

    numFrames_total = ind_concat - 1
    cv2.destroyAllWindows()

    displacements = displacements[:, :, ~np.isnan(displacements[0, 0, :])]

    return displacements, numFrames_total #, positions_tracked

def analyze_video(vidNum_iter, config, pointInds_toUse, pts_spaced):  # function needed for multiprocessing
    """
    computes optic flow for a single video within the multithread command
    similar to displacements monothread
    eventually refactor / recombine with displacements monothread

    Parameters
    ----------
    vidNum_iter ():
    config (dict): dictionary of config parameters
    pointInds_toUse ():
    displacements ():
    pts_spaced ():

    Returns
    -------
    displacements (): array of displacements
    """

    numVids = config['numVids']
    path_vid_allFiles = config['path_vid_allFiles']
    lk_names = [key for key in config.keys() if 'lk_' in key]
    lk_params = {k.split('lk_')[1]: (tuple(config[k]) if type(config[k]) is list else config[k]) \
                 for k in lk_names}

    vid = imageio.get_reader(path_vid_allFiles[vidNum_iter], 'ffmpeg')
    #     metadata = vid.get_meta_data()

    path_vid = path_vid_allFiles[vidNum_iter]  # get path of the current vid
    video = cv2.VideoCapture(path_vid)  # open the video object with openCV
    numFrames = int(video.get(
        cv2.CAP_PROP_FRAME_COUNT))  # get frame count of this vid GENERALLY INACCURATE. OFF BY AROUND -25 frames

    frameToSet = 0
    frame = vid.get_data(frameToSet)  # Get a single frame to use as the first 'previous frame' in calculating optic flow
    new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    old_frame = new_frame_gray

    displacements_tmp = np.zeros((pts_spaced.shape[0], 2, np.uint64(numFrames + (numVids * 1000)))) * np.nan

    print(' ', end='', flush=True)
    text = "progresser #{}".format(vidNum_iter)
    print(f'\n Calculating displacement field: video # {vidNum_iter + 1}/{numVids}')

    for iter_frame, new_frame in enumerate(tqdm(vid, total=numFrames, desc=text, position=vidNum_iter)):
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        ##calculate optical flow
        pointInds_new, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame_gray, pointInds_toUse, None,
                                                                **lk_params)  # Calculate displacement distance between STATIC/ANCHORED points and the calculated new points. Also note the excluded 'NextPts' parameter. Could be used for fancier tracking

        ## Calculate displacement and place into variable 'displacements' (changes in size every iter)
        if iter_frame == 0:
            displacements_tmp[:, :, iter_frame] = np.zeros((pts_spaced.shape[0], 2))
        else:
            displacements_tmp[:, :, iter_frame] = np.single(np.squeeze((pointInds_new - pointInds_toUse)))  # this is the important variable. Simply the difference in the estimate

        old_frame = new_frame_gray  # make current frame the 'old_frame' for the next iteration

    return displacements_tmp


def analyze_trial(trial_num, trial_inds, config, pointInds_toUse, pts_spaced):  # function needed for multiprocessing
    """
    computes optic flow for a single video within the multithread command
    similar to displacements monothread
    eventually refactor / recombine with displacements monothread

    Parameters
    ----------
    vidNum_iter ():
    config (dict): dictionary of config parameters
    pointInds_toUse ():
    displacements ():
    pts_spaced ():

    Returns
    -------
    displacements (): array of displacements
    """

    vidNum_iter = 0
    numVids = config['numVids']
    path_vid_allFiles = config['path_vid_allFiles']
    current_trial = trial_inds[trial_num].tolist()
    lk_names = [key for key in config.keys() if 'lk_' in key]
    lk_params = {k.split('lk_')[1]: (tuple(config[k]) if type(config[k]) is list else config[k]) \
                 for k in lk_names}

    vid = imageio.get_reader(path_vid_allFiles[vidNum_iter], 'ffmpeg')
    #     metadata = vid.get_meta_data()

    path_vid = path_vid_allFiles[vidNum_iter]  # get path of the current vid
    video = cv2.VideoCapture(path_vid)  # open the video object with openCV
    numFrames = trial_inds.shape[1]

    frameToSet = current_trial[0]
    frame = vid.get_data(frameToSet)  # Get a single frame to use as the first 'previous frame' in calculating optic flow
    new_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    old_frame = new_frame_gray

    displacements_tmp = np.zeros((pts_spaced.shape[0], 2, np.uint64(numFrames))) * np.nan

    print(' ', end='', flush=True)
    text = "progresser #{}".format(trial_num)
    print(f'\n Calculating displacement field: video # {trial_num + 1}/{trial_inds.shape[0]}')


    for iter_frame,current_frame in enumerate(tqdm(current_trial, total=numFrames, desc=text, position=trial_num)):
        new_frame = vid.get_data(current_frame)
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        ##calculate optical flow
        pointInds_new, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame_gray, pointInds_toUse, None,
                                                                **lk_params)  # Calculate displacement distance between STATIC/ANCHORED points and the calculated new points. Also note the excluded 'NextPts' parameter. Could be used for fancier tracking

        ## Calculate displacement and place into variable 'displacements' (changes in size every iter)
        if iter_frame == 0:
            displacements_tmp[:, :, iter_frame] = np.zeros((pts_spaced.shape[0], 2))
        else:
            displacements_tmp[:, :, iter_frame] = np.single(np.squeeze((pointInds_new - pointInds_toUse)))  # this is the important variable. Simply the difference in the estimate

        old_frame = new_frame_gray  # make current frame the 'old_frame' for the next iteration

    return displacements_tmp


def displacements_multithread(config, pointInds_toUse, displacements, pts_spaced):
    """
    wrapper for multithreaded optic flow computation
    operates on multiple videos at once

    Parameters
    ----------
    config (dict): dictionary of config parameters
    pointInds_toUse ():
    displacements ():
    pts_spaced ():

    Returns
    -------
    displacements (): array of displacements
    numFrames_total (int): number of frames
    """

    numVids = config['numVids']
    cv2.setNumThreads(0)
    freeze_support()
    tqdm.set_lock(RLock())
    p = Pool(multiprocessing.cpu_count(),initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    displacements_list = p.map(
        partial(analyze_video, config=config, pointInds_toUse=pointInds_toUse, pts_spaced=pts_spaced),
        list(np.arange(numVids)))

    ## all of the below called for safety.
    p.close()
    p.terminate()
    p.join()

    cv2.destroyAllWindows()

    for ii in range(len(displacements_list)):
        #     displacements[:,:,ii] = test_disp[ii]
        if ii == 0:
            displacements = displacements_list[ii]
        else:
            displacements = np.concatenate((displacements, displacements_list[ii]), axis=2)

    displacements = displacements[:, :, ~np.isnan(displacements[0, 0, :])]
    numFrames_total = displacements.shape[2]

    return displacements, numFrames_total


def displacements_trial_separated(config, pointInds_toUse, displacements, pts_spaced):
    """
    wrapper for multithreaded optic flow computation
    operates on multiple videos at once

    Parameters
    ----------
    config (dict): dictionary of config parameters
    pointInds_toUse ():
    displacements ():
    pts_spaced ():

    Returns
    -------
    displacements (): array of displacements
    numFrames_total (int): number of frames
    """

    trial_inds = np.load(config['trial_inds'])[:config['num_trials'],...]

    if config['optic_multithread']:
        cv2.setNumThreads(0)
        freeze_support()
        tqdm.set_lock(RLock())
        p = Pool(multiprocessing.cpu_count(), initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        displacements_trials = p.map(
            partial(analyze_trial, trial_inds=trial_inds, config=config, pointInds_toUse=pointInds_toUse, pts_spaced=pts_spaced),
            list(range(trial_inds.shape[0])))
        p.close()
        p.terminate()
        p.join()
    else:
        displacements_trials = []
        for i, _ in enumerate(trial_inds):
            displacements_tmp = analyze_trial(i, trial_inds, config, pointInds_toUse, pts_spaced)
            displacements_trials.append(displacements_tmp)

    displacements = np.stack(displacements_trials)
    displacements = displacements[..., ~np.isnan(displacements[0, 0, 0, :])]
    numFrames_total = displacements.shape[2]

    return displacements, numFrames_total


def optic_workflow(config_filepath):
    """
    sequences the steps of the optic flow computation

    Parameters
    ----------
    config_filepath (Path): path to the config file

    Returns
    -------

    """

    print(f'== Beginning optic flow computation ==')
    tic_all = time.time()

    config = helpers.load_config(config_filepath)
    pts_all = helpers.load_h5(config_filepath, 'path_pts_all')

    tic = time.time()
    pointInds_toUse, pointInds_tracked, pointInds_tracked_tuple, displacements, pts_spaced, color_tuples = setup(config,
                                                                                                                 pts_all)
    helpers.print_time('Optic Flow Set Up', time.time() - tic)

    tic = time.time()
    if config['trial_inds']:
        displacements_trials, numFrames_total = displacements_trial_separated(config, pointInds_toUse, displacements, pts_spaced)
    elif config['optic_multithread']:
        displacements, numFrames_total = displacements_multithread(config, pointInds_toUse, displacements, pts_spaced)
    elif config['optic_recursive']:
        displacements, numFrames_total = displacements_recursive(config, pointInds_toUse, pointInds_tracked,
                                                            pointInds_tracked_tuple, displacements, pts_spaced, color_tuples, config['optic_recursive_relaxation_factor'])
    else:
        displacements, numFrames_total = displacements_monothread(config, pointInds_toUse, pointInds_tracked,
                                                                  pointInds_tracked_tuple, displacements, pts_spaced, color_tuples)

    helpers.print_time('Displacements computed', time.time() - tic)

    tic = time.time()
    config['numFrames_total'] = numFrames_total
    helpers.save_config(config, config_filepath)

    helpers.save_data(config_filepath, 'pointInds_toUse', pointInds_toUse)
    helpers.create_nwb_group(config_filepath, 'Optic Flow')
    if config['trial_inds']:
        helpers.create_nwb_ts(config_filepath, 'Optic Flow', 'displacements_trials', displacements_trials)
    else:
        helpers.create_nwb_ts(config_filepath, 'Optic Flow', 'displacements', displacements)
    helpers.save_data(config_filepath, 'color_tuples', color_tuples)
    helpers.print_time('Data Saved', time.time() - tic)

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'Average frames per second: {round(numFrames_total / (time.time() - tic_all), 2)} fps')
    print(f'== End Optic Flow Computation ==')

import time
import numpy as np
import copy
import gc

from face_rhythm.util import helpers
from face_rhythm.visualize import videos

import cv2
import imageio

import multiprocessing
from multiprocessing import Pool, RLock, freeze_support
from functools import partial

from tqdm.notebook import tqdm

from pathlib import Path


def setup(config, session, pts_all):
    """
    initializes the arrays for optic flow analysis

    Args:
        config (dict): dictionary of config parameters
        session (dict): dictionary of session parameters
        pts_all (dict): dictionary of important points

    Returns:
        pointInds_toUse (np.ndarray): points indices to use for displacement calculations
        pointInds_tracked (np.ndarray): points for plotting
        pointInds_tracked_tuple (list of tuples): points for plotting
        displacements (np.ndarray): array of displacements
        pts_spaced (np.ndarray): spaced out points (similar to pointInds_toUse)
        color_tuples (list of tuples): colors for plotting
        positions_recursive (np.ndarray): recursively updated point locations
    """
    optic = config['Optic']
    general = config['General']

    numFrames_total_rough = session['frames_total']
    numVids = session['num_vids']
    spacing = optic['spacing']

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
    positions_recursive = np.ones((pts_spaced.shape[0], 2, np.uint64(
        numFrames_total_rough + numFrames_total_rough * 0.1 + (numVids * 1000)))) * np.nan

    ## Preset point tracking variables
    pointInds_toUse = copy.deepcopy(pts_spaced)
    pointInds_tracked = pointInds_toUse  ## set the first frame to have point locations be positions in the point cloud
    pointInds_tracked_tuple = list(np.arange(pointInds_toUse.shape[0]))

    if config['Video']['frames_to_ignore_pref']:
        frames_to_ignore = np.load(session['frames_to_ignore'])
        print(frames_to_ignore)
    else:
        frames_to_ignore = None

    return pointInds_toUse, pointInds_tracked, pointInds_tracked_tuple, displacements, pts_spaced, color_tuples , positions_recursive, frames_to_ignore



def displacements_monothread(config, pointInds_toUse, pointInds_tracked, pointInds_tracked_tuple, displacements,
                             pts_spaced, color_tuples, session):
    """
    the workhorse of the optic flow
    Opens each video in the list of videos
    Iterates through the frames of the video
    Computes the optic flow between frames
    Saves this to displacements frame

    Args:
        config (dict): dictionary of config parameters
        pointInds_toUse (np.ndarray): points indices to use for displacement calculations
        pointInds_tracked (np.ndarray): points for plotting
        pointInds_tracked_tuple (list of tuples): points for plotting
        displacements (np.ndarray): array of displacements
        pts_spaced (np.ndarray): spaced out point locations
        color_tuples (list of tuples): colors for plotting
        session (dict): dict of session parameters

    Returns:
        displacements (np.ndarray): array of displacements
        numFrames_total (int): number of frames
    """

    ## Main loop to pull out displacements in each video
    ind_concat = 0
    fps = 0
    tic_fps = time.time()
    tic_all = time.time()

    optic = config['Optic']
    video = config['Video']

    vidNums_toUse = optic['vidNums_toUse']
    showVideo_pref = optic['showVideo_pref']
    fps_counterPeriod = video['fps_counterPeriod']
    printFPS_pref = video['printFPS_pref']
    remote = config['General']['remote']
    save_vid = video['save_demo']

    Fs = video['Fs']
    vid_width = video['width']
    vid_height = video['height']
    test_len = video['demo_len']
    save_pathFull = str(Path(config['Paths']['viz']) / 'optic_test.avi')

    numVids = session['num_vids']
    path_vid_allFiles = session['videos']
    lk_names = [key for key in optic.keys() if 'lk_' in key]
    lk_params = {k.split('lk_')[1]: (tuple(optic[k]) if type(optic[k]) is list else optic[k]) \
                 for k in lk_names}

    # Define the codec and create VideoWriter object
    if showVideo_pref and (save_vid or remote):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(f'saving to file {save_pathFull}')
        out = cv2.VideoWriter(save_pathFull, fourcc, Fs, (np.int64(vid_width), np.int64(vid_height)))
    else:
        out = None
    vid_lens = []
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
                pointInds = [pointInds_tracked, pointInds_tracked_tuple]
                counters = [iter_frame, vidNum_iter, ind_concat, fps]
                if (remote and iter_frame < test_len) or not remote:
                    videos.visualize_progress(config, session, new_frame, pointInds, color_tuples, counters, out)

                if (save_vid or remote) and iter_frame == test_len:
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
        vid_lens.append(iter_frame+1)
        ## Calculate how long calculation took
        elapsed = time.time() - tic_all
        helpers.print_time('video time elapsed:', elapsed)
        print(f'Capture rate: {round(ind_concat / elapsed, 3)} fps')

    numFrames_total = ind_concat - 1
    cv2.destroyAllWindows()

    displacements = displacements[:, :, ~np.isnan(displacements[0, 0, :])]

    return displacements, numFrames_total, vid_lens



def displacements_recursive(config, pointInds_toUse, pointInds_tracked, pointInds_tracked_tuple, positions_recursive,
                            pts_spaced, color_tuples, relaxation_factor, session, frames_to_ignore):
    """
    the workhorse of the optic flow
    Opens each video in the list of videos
    Iterates through the frames of the video
    Computes the optic flow between frames
    Saves this to displacements frame

    Args:
        config (dict): dictionary of config parameters
        pointInds_toUse (np.ndarray): points indices to use for displacement calculations
        pointInds_tracked_tuple (list of tuples): points for plotting
        positions_recursive(np.ndarray): recursively updated point locations
        pts_spaced (np.ndarray): spaced out points locations
        color_tuples (list of tuples): colors for plotting
        relaxation_factor (float): relaxation factor between frames
        session (dict): dict of session parameters

    Returns:
        displacements (np.ndarray): array of displacements
        numFrames_total (int): number of frames
        positions_recursive (np.ndarray): recursively updated point locations
    """

    ## Main loop to pull out displacements in each video
    ind_concat = 0
    fps = 0
    tic_fps = time.time()
    tic_all = time.time()

    optic = config['Optic']
    video = config['Video']

    vidNums_toUse = optic['vidNums_toUse']
    showVideo_pref = optic['showVideo_pref']
    fps_counterPeriod = video['fps_counterPeriod']
    printFPS_pref = video['printFPS_pref']
    remote = config['General']['remote']
    save_vid = video['save_demo']


    Fs = video['Fs']
    vid_width = video['width']
    vid_height = video['height']
    test_len = video['demo_len']
    save_pathFull = str(Path(config['Paths']['viz']) / 'optic_test.avi')

    numVids = session['num_vids']
    path_vid_allFiles = session['videos']
    
    lk_names = [key for key in optic['lk'].keys()]
    lk_params = {k: (tuple(optic['lk'][k]) if type(optic['lk'][k]) is list else optic['lk'][k]) \
                 for k in lk_names}
    

    # Define the codec and create VideoWriter object
    if showVideo_pref and (save_vid or remote):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(f'saving to file {save_pathFull}')
        out = cv2.VideoWriter(save_pathFull, fourcc, Fs, (np.int64(vid_width), np.int64(vid_height)))
    else:
        out = None

    vid_lens = []
    pointInds_old = pointInds_toUse
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

        print(f'\n Calculating displacement field: video # {vidNum_iter+1}/{numVids}')
        for iter_frame , new_frame in enumerate(tqdm(vid, total=numFrames_rough)):

            new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            ##calculate optical flow
            #         pointInds_new, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame_gray, pointInds_toUse, None, **lk_params)  # Calculate displacement distance between STATIC/ANCHORED points and the calculated new points. Also note the excluded 'NextPts' parameter. Could be used for fancier tracking
            pointInds_new, status, error = cv2.calcOpticalFlowPyrLK(old_frame, new_frame_gray, np.single(pointInds_old),
                                                                    None,
                                                                    **lk_params)  # Calculate displacement distance between STATIC/ANCHORED points and the calculated new points. Also note the excluded 'NextPts' parameter. Could be used for fancier tracking

            # diff =  np.squeeze((pointInds_new - pointInds_old))  # this is the important variable. Simply the difference in the estimate
            
            old_frame = new_frame_gray  # make current frame the 'old_frame' for the next iteration
            
            if frames_to_ignore is None:
                #             pointInds_tracked = pointInds_tracked + (pointInds_new - pointInds_toUse)  # calculate integrated position
                # pointInds_tracked = pointInds_tracked + diff[:,None,:]  # calculate integrated position
                pointInds_tracked = pointInds_new - (pointInds_new -pointInds_toUse)*relaxation_factor  # multiplied constant is the relaxation term
            elif frames_to_ignore[iter_frame]==0:
                pointInds_tracked = pointInds_new - (pointInds_new -pointInds_toUse)*relaxation_factor  # multiplied constant is the relaxation term
            else:
                pointInds_tracked = pointInds_old

            ## Calculate displacement and place into variable 'displacements' (changes in size every iter)         
            if ind_concat ==0:
                positions_recursive[:,:,ind_concat] = np.zeros((pts_spaced.shape[0] ,2))
            else:
    #             displacements[:,:,ind_concat] =  np.single(np.squeeze((pointInds_new - pointInds_toUse)))  # this is the important variable. Simply the difference in the estimate
                positions_recursive[:,:,ind_concat] =  np.squeeze((pointInds_tracked))  # this is the important variable. Simply the difference in the estimate

            pointInds_old = pointInds_tracked

            ## below is just for visualization. Nothing calculated is maintained
            if showVideo_pref:

                pointInds = [pointInds_tracked, pointInds_tracked_tuple]
                counters = [iter_frame, vidNum_iter, ind_concat, fps]
                if (remote and iter_frame < test_len) or not remote:
                    videos.visualize_progress(config, session, new_frame, pointInds, color_tuples, counters, out)

                if (save_vid or remote) and iter_frame == test_len:
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
        vid_lens.append(iter_frame+1)

        ## Calculate how long calculation took
        elapsed = time.time() - tic_all
        helpers.print_time('video time elapsed:', elapsed)
        print(f'Capture rate: {round(ind_concat / elapsed, 3)} fps')

    numFrames_total = ind_concat
    cv2.destroyAllWindows()

    positions_recursive = positions_recursive[:, :, ~np.isnan(positions_recursive[0, 0, :])]
    displacements = np.zeros_like(positions_recursive)
    displacements[:,:,1:] = np.diff(positions_recursive , axis=2)

    return displacements, numFrames_total , positions_recursive, vid_lens


def analyze_video(vidNum_iter, config, pointInds_toUse, pts_spaced, session):  # function needed for multiprocessing
    """
    computes optic flow for a single video within the multithread command
    similar to displacements monothread
    eventually refactor / recombine with displacements monothread

    Args:
        vidNum_iter (int): current video of analysis
        config (dict): dictionary of config parameters
        pointInds_toUse (np.ndarray): points indices to use for displacement calculations
        displacements (np.ndarray): displacement of each point for each time step
        pts_spaced (np.ndarray): spaced out points (similar to pointInds_toUse)

    Returns:
        displacements_tmp (np.ndarray): array of displacements
    """

    optic = config['Optic']

    numVids = session['num_vids']
    path_vid_allFiles = session['videos']
    lk_names = [key for key in optic.keys() if 'lk_' in key]
    lk_params = {k.split('lk_')[1]: (tuple(optic[k]) if type(optic[k]) is list else optic[k]) \
                 for k in lk_names}

    vid = imageio.get_reader(path_vid_allFiles[vidNum_iter], 'ffmpeg')
    #     metadata = vid.get_meta_data()

    path_vid = path_vid_allFiles[vidNum_iter]  # get path of the current vid
    video = cv2.VideoCapture(path_vid)  # open the video object with openCV
    numFrames = int(video.get(
        cv2.CAP_PROP_FRAME_COUNT))  # get frame count of this vid GENERALLY INACCURATE. OFF BY AROUND -25 frames

    frameToSet = 0
    frame = vid.get_data(
        frameToSet)  # Get a single frame to use as the first 'previous frame' in calculating optic flow
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
            displacements_tmp[:, :, iter_frame] = np.single(np.squeeze((
                    pointInds_new - pointInds_toUse)))  # this is the important variable. Simply the difference in the estimate

        old_frame = new_frame_gray  # make current frame the 'old_frame' for the next iteration

    return displacements_tmp


def displacements_multithread(config, pointInds_toUse, displacements, pts_spaced, session):
    """
    wrapper for multithreaded optic flow computation
    operates on multiple videos at once

    Args:
        config (dict): dictionary of config parameters
        pointInds_toUse (np.ndarray): points indices to use for displacement calculations
        displacements (np.ndarray): displacement of each point for each time step
        pts_spaced (np.ndarray): spaced out point locations
        session (dict): dictionary of session level information

    Returns:
        displacements (np.ndarray): array of displacements
        numFrames_total (int): number of frames
    """

    numVids = session['num_vids']
    cv2.setNumThreads(0)
    freeze_support()
    tqdm.set_lock(RLock())
    p = Pool(multiprocessing.cpu_count(), initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
    displacements_list = p.map(
        partial(analyze_video, config=config, pointInds_toUse=pointInds_toUse, pts_spaced=pts_spaced, session=session),
        list(np.arange(numVids)))

    ## all of the below called for safety.
    p.close()
    p.terminate()
    p.join()

    cv2.destroyAllWindows()
    vid_lens = []
    for ii in range(len(displacements_list)):
        vid_lens.append(displacements_list[ii].shape[-1])
        if ii == 0:
            displacements = displacements_list[ii]
        else:
            displacements = np.concatenate((displacements, displacements_list[ii]), axis=2)

    displacements = displacements[:, :, ~np.isnan(displacements[0, 0, :])]
    numFrames_total = displacements.shape[-1]

    return displacements, numFrames_total, vid_lens



def optic_workflow(config_filepath):
    """
    sequences the steps of the optic flow computation

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """

    print(f'== Beginning optic flow computation ==')
    tic_all = time.time()

    config = helpers.load_config(config_filepath)
    optic = config['Optic']
    general = config['General']
    video = config['Video']

    if optic['recursive'] and optic['multithread']:
        raise NameError("Incompatible option combination:  If optic_recursive==True, optic_multithread MUST ==False \n\
    The recursive calculation is done serially, so it is not possible to parallelize it.")
    try:
        for session in general['sessions']:
            tic_session = time.time()
            tic = tic_session
            pts_all = helpers.get_pts(session['nwb'])

            pointInds_toUse, pointInds_tracked, pointInds_tracked_tuple, displacements, pts_spaced, color_tuples, positions_recursive, frames_to_ignore = setup(config, session, pts_all)
            helpers.print_time('Optic Flow Set Up', time.time() - tic)

            tic = time.time()

            if optic['multithread']:
                displacements, numFrames_total, vid_lens = displacements_multithread(config, pointInds_toUse, displacements, pts_spaced, session)
            elif optic['recursive']:
                displacements, numFrames_total , positions_recursive, vid_lens = displacements_recursive(config, pointInds_toUse, pointInds_tracked,
                                                                         pointInds_tracked_tuple, positions_recursive, pts_spaced,
                                                                         color_tuples,
                                                                         optic['recursive_relaxation_factor'],
                                                                         session,
                                                                         frames_to_ignore)
            else:
                displacements, numFrames_total, vid_lens = displacements_monothread(config, pointInds_toUse, pointInds_tracked,
                                                                          pointInds_tracked_tuple, displacements, pts_spaced,
                                                                          color_tuples, session)


            helpers.print_time('Displacements computed', time.time() - tic)
            session['numFrames_total'] = numFrames_total
            session['vid_lens_true'] = vid_lens
            session['vid_locs'] = [sum(vid_lens[:i+1]) for i in range(len(vid_lens))]
            optic['num_dots'] = pointInds_toUse.shape[0]
            helpers.save_config(config, config_filepath)

            helpers.create_nwb_group(session['nwb'], 'Optic Flow')
            helpers.create_nwb_ts(session['nwb'], 'Optic Flow', 'displacements', displacements, video['Fs'])
            helpers.create_nwb_ts(session['nwb'], 'Optic Flow', 'positions_recursive', positions_recursive,video['Fs'])
            helpers.create_nwb_ts(session['nwb'], 'Optic Flow', 'color_tuples', np.array(color_tuples),video['Fs'])
            helpers.create_nwb_ts(session['nwb'], 'Optic Flow', 'pointInds_toUse', pointInds_toUse, video['Fs'])

            helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)
            print(f'Total number of frames: {numFrames_total} frames')
            print(f'Average frames per second: {round(numFrames_total / (time.time() - tic_session), 2)} fps')

            del pointInds_tracked, pointInds_tracked_tuple, displacements, pts_spaced, positions_recursive, color_tuples, pointInds_toUse
    except KeyboardInterrupt:
        print('"Shutdown requested...exiting"')
        del pointInds_tracked, pointInds_tracked_tuple, displacements, pts_spaced, positions_recursive, color_tuples, pointInds_toUse

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End Optic Flow Computation ==')

    gc.collect()

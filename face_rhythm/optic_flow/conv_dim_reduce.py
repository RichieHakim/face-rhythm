import numpy as np
import sklearn.decomposition
import gc

import cv2
import imageio
import copy
import time
from tqdm import tqdm, trange

from matplotlib import pyplot as plt

from face_rhythm.util_old import helpers

def cosine_kernel_2D(center=(5,5), image_size=(11,11), width=5):
    """
    Generate a 2D cosine kernel
    RH 2021
    
    Args:
        center (tuple):  
            The mean position (X, Y) - where high value expected. 0-indexed. Make second value 0 to make 1D
        image_size (tuple): 
            The total image size (width, height). Make second value 0 to make 1D
        width (scalar): 
            The full width of one cycle of the cosine
    
    Return:
        k_cos (np.ndarray): 
            2D or 1D array of the cosine kernel
    """
    x, y = np.meshgrid(range(image_size[1]), range(image_size[0]))  # note dim 1:X and dim 2:Y
    dist = np.sqrt((y - int(center[1])) ** 2 + (x - int(center[0])) ** 2)
    dist_scaled = (dist/(width/2))*np.pi
    dist_scaled[np.abs(dist_scaled > np.pi)] = np.pi
    k_cos = (np.cos(dist_scaled) + 1)/2
    return k_cos


def create_kernel(config_filepath, point_idxs):
    """
    creates convolutional kernel
    RH 2021

    Args:
        config_filepath (Path): path to the config file
        point_idxs (np.ndarray): point indices

    Returns:
        cos_kernel (np.ndarray): cosine kernel
        cos_kernel_mean (np.ndarray): mean of cosine kernel
    """

    config = helpers.load_config(config_filepath)
    cdr = config['CDR']
    video = config['Video']

    width_cos_kernel = cdr['width_cosKernel']
    num_dots = cdr['num_dots']
    vid_height = video['height']
    vid_width = video['width']

    point_idxs_squeeze = np.fliplr(point_idxs.squeeze()) # flip so that the first index is the y-axis

    k_width = copy.copy(width_cos_kernel)
    k_center = k_width//2

    k_cos = cosine_kernel_2D(center=(k_center, k_center), image_size=(k_width, k_width), width=k_width)
    cos_kernel = np.zeros((vid_height, vid_width, num_dots))
    cos_kernel_mean = np.zeros(num_dots)
    for ii in tqdm(range(num_dots),desc="creating kernel"):
        # The following code block deals with clipping the edges
        fi = point_idxs_squeeze[ii] - np.floor(k_width/2) # 'first index'
        fis = -np.minimum(fi, 0) # 'first index shift'
        fi_im = fi + fis # 'first index image'
        fi_k = fis # 'first index kernel'
        li = point_idxs_squeeze[ii] + np.ceil(k_width/2) # 'last index'
        li_im = np.minimum(li, np.array([vid_height, vid_width])) # 'last index image'
        lis = -(li - li_im) # 'last index shift'
        li_k = k_width + lis # 'last index kernel'
        fi_im, fi_k, li_im, li_k = np.int64(fi_im), np.int64(fi_k), np.int64(li_im), np.int64(li_k)

        cos_kernel[fi_im[0]:li_im[0], fi_im[1]:li_im[1], ii] = k_cos[fi_k[0]:li_k[0], fi_k[1]:li_k[1]]
        
        tmp = copy.deepcopy(cos_kernel[:, :, ii])
        tmp[tmp == 0] = np.nan
        cos_kernel_mean[ii] = np.nanmean(tmp)
    return cos_kernel, cos_kernel_mean


def space_points(config_filepath, pts_all):
    """
    spaces out the points

    Args:
        config_filepath (Path): path to the config file
        pts_all (dict): dict of point arrays

    Returns:
        pts_spaced_convDR (np.ndarray): spaced out point array
    """

    config = helpers.load_config(config_filepath)
    cdr = config['CDR']

    spacing = cdr['spacing']

    bbox_subframe_displacement = pts_all['bbox_subframe_displacement']
    pts_x_displacement = pts_all['pts_x_displacement']
    pts_y_displacement = pts_all['pts_y_displacement']

    pts_spaced_convDR = np.ones((np.int64(bbox_subframe_displacement[3] * bbox_subframe_displacement[2] / spacing),
                                 2)) * np.nan  ## preallocation
    cc = 0  ## set idx counter
    # make spaced out points
    for ii in range(len(pts_x_displacement)):
        if (pts_x_displacement[ii] % spacing == 0) and (pts_y_displacement[ii] % spacing == 0):
            pts_spaced_convDR[cc, 0] = pts_x_displacement[ii]
            pts_spaced_convDR[cc, 1] = pts_y_displacement[ii]
            cc = cc + 1

    pts_spaced_convDR = np.expand_dims(pts_spaced_convDR, 1).astype('single')
    pts_spaced_convDR = np.delete(pts_spaced_convDR, np.where(np.isnan(pts_spaced_convDR[:, 0, 0])), axis=0)
    return pts_spaced_convDR


def points_show(config_filepath, session, pts_all, pts_spaced_convDR, cosKernel):
    """
    shows the points with the cosKernel overlayed

    Args:
        config_filepath (Path): path to the config file
        session (dict): current session dictionary
        pts_all (dict): dict of point arrays
        pts_spaced_convDR (np.ndarray): array of spaced points
        cosKernel (np.ndarray): cosine kernel

    Returns:

    """

    config = helpers.load_config(config_filepath)
    cdr = config['CDR']
    vidNum_toUse = cdr['vidNum']
    frameNum_toUse = cdr['frameNum']
    dot_size = cdr['dot_size']
    kernel_pixel = cdr['kernel_pixel']
    path_vid_allFiles = session['videos']

    kernel_example = np.zeros_like(cosKernel[...,:3])
    kernel_example[...,2] = cosKernel[...,kernel_pixel]
    alpha = cdr['kernel_alpha']

    color_tuples = list(np.arange(len(pts_spaced_convDR)))
    for ii in range(len(pts_spaced_convDR)):
        color_tuples[ii] = tuple([int(np.random.rand(1)[0] * 255), int(np.random.rand(1)[0] * 255), int(np.random.rand(1)[0] * 255)])
    
    # color_tuples = helpers.load_data(config_filepath, 'color_tuples')

    vid = imageio.get_reader(path_vid_allFiles[vidNum_toUse], 'ffmpeg')
    frame = vid.get_data(
        frameNum_toUse)  # Get a single frame to use as the first 'previous frame' in calculating optic flow
    pointInds_tuple = list(np.arange(pts_spaced_convDR.shape[0]))
    for ii in range(pts_spaced_convDR.shape[0]):
        pointInds_tuple[ii] = tuple(np.squeeze(pts_spaced_convDR[ii, 0, :]).astype('int64'))
        cv2.circle(frame, pointInds_tuple[ii], dot_size, color_tuples[ii], -1)
    plt.imshow(cv2.cvtColor(np.float32((frame*(1-alpha)+255*kernel_example*alpha)/255), cv2.COLOR_BGR2RGB))
    plt.show()



def makeConvDR(ii, input_traces, cos_kernel, cos_kernel_mean, pca, rank_reduced, dots_new):
    """
    performs the convolutional dimensionality reduction
    called within the multithreading

    Args:
        ii ():
        input_traces ():
        cos_kernel ():
        cos_kernel_mean ():
        pca ():
        rank_reduced ():
        dots_new ():

    Returns:
        positions_convDR_meanSub ():

    """

    influence_weightings = cos_kernel[int(dots_new[ii][0][1]), int(dots_new[ii][0][0]), :]

    idx_nonZero = np.array(np.where(influence_weightings != 0))[0, :]

    displacements_preConvDR_x = input_traces[idx_nonZero, 0, :] * influence_weightings[idx_nonZero][:, None]
    displacements_preConvDR_x = displacements_preConvDR_x - np.mean(displacements_preConvDR_x, axis=1)[:, None]
    displacements_preConvDR_y = input_traces[idx_nonZero, 1, :] * influence_weightings[idx_nonZero][:, None]
    displacements_preConvDR_y = displacements_preConvDR_y - np.mean(displacements_preConvDR_y, axis=1)[:, None]
    pca.fit(displacements_preConvDR_x)
    output_PCA_loadings0 = pca.components_.T
    pca.fit(displacements_preConvDR_y)
    output_PCA_loadings1 = pca.components_.T

    output_PCA_scores0 = np.dot(displacements_preConvDR_x, output_PCA_loadings0)
    output_PCA_scores1 = np.dot(displacements_preConvDR_y, output_PCA_loadings1)
    positions_convDR_meanSub = np.zeros((2, input_traces.shape[2]))
    positions_convDR_meanSub[0, :] = np.mean(
        np.dot(output_PCA_loadings0[:, :rank_reduced], output_PCA_scores0[:, :rank_reduced].T), axis=1) / \
                                     cos_kernel_mean[ii]
    positions_convDR_meanSub[1, :] = np.mean(
        np.dot(output_PCA_loadings1[:, :rank_reduced], output_PCA_scores1[:, :rank_reduced].T), axis=1) / \
                                     cos_kernel_mean[ii]
    return positions_convDR_meanSub


def compute_influence(config_filepath, pointInds_toUse, pts_spaced_convDR, cosKernel, cosKernel_mean, positions_new_sansOutliers):
    """
    performs single-threaded convolutional dimensionality reduction

    Args:
        config_filepath (Path): path to current config file
        pointInds_toUse (np.ndarray): original point indices
        pts_spaced_convDR (np.ndarray): point locations after dimensionality reduction
        cosKernel (np.ndarray): cosine kernel
        cosKernel_mean (np.ndarray): mean of the cosine kernel
        positions_new_sansOutliers (np.ndarray): integrated positions

    Returns:
        positions_convDR_meanSub (np.ndarray): positions after dim red with mean substracted
        positions_convDR_absolute (np.ndarray): absolute positions after dim red
    """

    config = helpers.load_config(config_filepath)
    num_components = config['CDR']['num_components']
    
    input_traces = np.float32(positions_new_sansOutliers)
    # num_components = 3
    rank_reduced = num_components

    dots_old = pointInds_toUse
    dots_new = pts_spaced_convDR

    pca = sklearn.decomposition.PCA(n_components=num_components)

    positions_convDR_meanSub = np.zeros((dots_new.shape[0] , 2 , input_traces.shape[2]))
    output_PCA_loadings = np.zeros((dots_new.shape[0] , 2 , input_traces.shape[2] , num_components))
    output_PCA_scores = list(np.zeros(dots_new.shape[0]))

    for ii in trange(dots_new.shape[0] , mininterval=1):
        influence_weightings = cosKernel[int(dots_new[ii][0][1]) , int(dots_new[ii][0][0]) , :]
        
        idx_nonZero = np.array(np.where(influence_weightings !=0))[0,:]

        displacements_preConvDR_x = input_traces[idx_nonZero , 0 , :] * influence_weightings[idx_nonZero][:,None]
        displacements_preConvDR_x = displacements_preConvDR_x - np.mean(displacements_preConvDR_x , axis=1)[:,None]
        displacements_preConvDR_y = input_traces[idx_nonZero , 1 , :] * influence_weightings[idx_nonZero][:,None]
        displacements_preConvDR_y = displacements_preConvDR_y - np.mean(displacements_preConvDR_y , axis=1)[:,None]
        pca.fit(displacements_preConvDR_x)
        output_PCA_loadings[ii,0,:,:] = pca.components_.T
        pca.fit(displacements_preConvDR_y)
        output_PCA_loadings[ii,1,:,:] = pca.components_.T
        
        output_PCA_scores[ii] = np.zeros((2,displacements_preConvDR_y.shape[0] , num_components))
        output_PCA_scores[ii][0,:,:] = np.dot( displacements_preConvDR_x  ,  output_PCA_loadings[ii,0,:,:] )
        output_PCA_scores[ii][1,:,:] = np.dot( displacements_preConvDR_y  ,  output_PCA_loadings[ii,1,:,:] )
        positions_convDR_meanSub[ii,0,:] = np.mean(np.dot( output_PCA_loadings[ii,0,:,:rank_reduced] , output_PCA_scores[ii][0,:,:rank_reduced].T ) , axis=1) / cosKernel_mean[ii]
        positions_convDR_meanSub[ii,1,:] = np.mean(np.dot( output_PCA_loadings[ii,1,:,:rank_reduced] , output_PCA_scores[ii][1,:,:rank_reduced].T ) , axis=1) / cosKernel_mean[ii]

    positions_convDR_absolute = (positions_convDR_meanSub + np.squeeze(pts_spaced_convDR)[:, :, None])
    return positions_convDR_meanSub, positions_convDR_absolute


def conv_dim_reduce_workflow(config_filepath):
    """
    sequences the steps for performing convolutional dimensionality reduction on the points

    Args:
        config_filepath (Path): path to the current config file

    Returns:

    """

    print(f'== Beginning convolutional dimensionality reduction ==')
    tic_all = time.time()

    config = helpers.load_config(config_filepath)
    general = config['General']
    video = config['Video']

    for session in general['sessions']:
        pointInds_toUse = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', 'pointInds_toUse')
        pts_all = helpers.get_pts(session['nwb'])

        # let's make new dots with wider spacing
        tic = time.time()
        pts_spaced_convDR = space_points(config_filepath, pts_all)
        print(f'number of points: {pts_spaced_convDR.shape[0]}')
        helpers.print_time('Points spaced out', time.time() - tic)
        
        # first let's make the convolutional kernel. I like the cosine kernel because it goes to zero.
        tic = time.time()
        cosKernel, cosKernel_mean = create_kernel(config_filepath, pointInds_toUse)
        helpers.print_time('Kernel created', time.time() - tic)

        tic_session = time.time()
        if config['CDR']['display_points']:
            points_show(config_filepath, session, pts_all, pts_spaced_convDR, cosKernel)
        positions_new_sansOutliers = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', 'positions_cleanup')
        positions_convDR_meanSub, positions_convDR_absolute = compute_influence(config_filepath, pointInds_toUse, pts_spaced_convDR,
                                                                 cosKernel, cosKernel_mean, positions_new_sansOutliers)

        helpers.create_nwb_ts(session['nwb'], 'Optic Flow', 'positions_convDR_meanSub', positions_convDR_meanSub, video['Fs'])
        helpers.create_nwb_ts(session['nwb'], 'Optic Flow', 'positions_convDR_absolute', positions_convDR_absolute, video['Fs'])
        helpers.create_nwb_ts(session['nwb'], 'Optic Flow', 'pts_spaced_convDR', pts_spaced_convDR, video['Fs'])
        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)

        del positions_new_sansOutliers, positions_convDR_meanSub, positions_convDR_absolute, cosKernel, cosKernel_mean, pts_spaced_convDR, pts_all, pointInds_toUse

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End convolutional dimensionality reduction ==')

    gc.collect()
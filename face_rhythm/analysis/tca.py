import sys
import time
import imageio

import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import scipy
import scipy.signal
import tensorly as tl
import tensorly.decomposition
import sklearn.decomposition
import sklearn.manifold

from tqdm.notebook import tqdm

from face_rhythm.util import helpers


def tca(config_filepath, positions):
    """
    computes the tca of the provided dataframe

    Parameters
    ----------
    config_filepath (Path): path to the config file
    positions ():

    Returns
    -------
    factors_np ():

    """
    
    config = helpers.load_config(config_filepath)
    pref_useGPU = config['tca_pref_useGPU']
    device = config['tca_device']
    rank = config['tca_rank']
    init = config['tca_init']
    verbosity = config['tca_verbosity']
    n_iters = config['tca_n_iters']

    input_dimRed_meanSub = helpers.load_data(config_filepath,'path_input_dimRed_meanSub')
    
    tl.set_backend('pytorch')
    
    ## Prepare the input tensor
    input_tensor = tl.tensor(positions, dtype=tl.float32, device=device, requires_grad=False)

    print(f'Size of input (spectrogram): {input_tensor.shape}')

    print(f'{round(sys.getsizeof(input_dimRed_meanSub)/1000000000,3)} GB')
    
    ### Fit TCA model
    ## If the input is small, set init='svd'
    weights, factors = tensorly.decomposition.non_negative_parafac(input_tensor, init=init, tol=1e-05, n_iter_max=n_iters, rank=rank, verbose=verbosity)

    ## make numpy version of tensorly output

    factors_toUse = factors


    if pref_useGPU:
        factors_np = list(np.arange(len(factors_toUse)))
        for ii in range(len(factors_toUse)):
    #         factors_np[ii] = tl.tensor(factors[ii] , dtype=tl.float32 , device='cpu')
            factors_np[ii] = factors_toUse[ii].cpu().clone().detach().numpy()
    else:
        factors_np = []
        for ii in range(len(factors_toUse)):
            factors_np.append(np.array(factors_toUse[ii]))
            
    return factors_np


def plot_factors(config_filepath, factors_np):
    """
    plots the positional factors for visualization / analysis

    Parameters
    ----------
    config_filepath (Path): path to the config file
    factors_np ():

    Returns
    -------

    """

    factors_toUse = factors_np
    modelRank = factors_toUse[0].shape[1]
    ## just for plotting in case 
#     if 'Fs' not in globals():
#         Fs = 120
    config = helpers.load_config(config_filepath)
    Fs = config['vid_Fs']

    plt.figure()
    # plt.plot(np.arange(factors_toUse.factors(4)[0][2].shape[0])/Fs , factors_toUse.factors(4)[0][2])
    factors_temporal = scipy.stats.zscore(factors_toUse[2][:,:] , axis=0)
    factors_temporal = factors_toUse[2][:,:]
    # factors_temporal = scipy.stats.zscore(factors_temporal_reconstructed , axis=0)
    # plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,:])
    plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,])
    # plt.plot(factors_temporal[:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('time (s)')
    plt.ylabel('a.u.')


    plt.figure()
    plt.plot(factors_toUse[1][:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('x vs. y')
    plt.ylabel('a.u.')

    plt.figure()
    plt.plot(factors_toUse[0][:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('pixel number')
    plt.ylabel('a.u.')


    plt.figure()
    plt.imshow(np.single(np.corrcoef(factors_toUse[2][:,:].T)))

    # input_dimRed = factors_toUse[2][:,:]
    # # input_dimRed_meanSub = 
    # pca = sk.decomposition.PCA(n_components=modelRank-2)
    # # pca = sk.decomposition.FactorAnalysis(n_components=3)
    # pca.fit(np.single(input_dimRed).transpose())
    # output_PCA = pca.components_.transpose()
    # # scores_points = np.dot(ensemble.factors(4)[0][2] , output_PCA)

    # plt.figure()
    # plt.plot(output_PCA)
    
    
def factor_videos(config_filepath, factors_np, positions_convDR_absolute):
    """
    creates videos of the points colored by their positional factor values

    Parameters
    ----------
    config_filepath (Path): path to the config file
    factors_np ():
    positions_convDR_absolute ():

    Returns
    -------

    """

    config = helpers.load_config(config_filepath)
    Fs = config['vid_Fs']
    vid_width = config['vid_width']
    vid_height = config['vid_height']
    numFrames_allFiles = config['numFrames_allFiles']
    path_vid_allFiles = config['path_vid_allFiles']

    # Display video of factors

    factors_toShow = np.arange(factors_np[0].shape[1])  # zero-indexed
    # factors_toShow = [3]  # zero-indexed

    for factor_iter in factors_toShow:

        # vidNums_toUse = range(numVids) ## note zero indexing!
        vidNums_toUse = config['vidNums_toUse'] ## note zero indexing!

        if type(vidNums_toUse) == int:
            vidNums_toUse = np.array([vidNums_toUse])

        dot_size = 2

        printFPS_pref = 0
        fps_counterPeriod = 10 ## number of frames to do a tic toc over

    #     modelRank_toUse = 5
        factor_toShow = factor_iter+1
        save_pref = config['tca_vid_save']

        # save_dir = "F:\\RH_Local\\Rich data\\camera data"
        save_dir = config['tca_vid_dir']
        save_fileName = f'factor {factor_toShow}'
        # save_pathFull = f'{save_dir}\\{save_fileName}.avi'
        save_pathFull = f'{save_dir}/{save_fileName}.avi'
        config[f'path_{save_fileName}'] = save_pathFull
        helpers.save_config(config, config_filepath)

        # ensemble_toUse = ensemble
        ensemble_toUse = factors_np
        positions_toUse = positions_convDR_absolute

        factor_toShow = factor_toShow-1
        # input_scores = ensemble_toUse.factors(modelRank_toUse)[0][0]
        input_scores = np.single(ensemble_toUse[0])

        range_toUse = np.ceil(np.max(input_scores[:,factor_toShow]) - np.min(input_scores[:,factor_toShow])) + 1
        offset_toUse = np.min(input_scores[:,factor_toShow])
        scores_norm = input_scores[:,factor_toShow] - offset_toUse
        scores_norm = (scores_norm / np.max(scores_norm)) *1000
        cmap = matplotlib.cm.get_cmap('hot', 1000)
        # cmap_viridis(np.arange(range_toUse))

        colormap_tuples =  list(np.arange(positions_toUse.shape[0]))
        for ii in range(positions_toUse.shape[0]):
            colormap_tuples[ii] = list(np.flip((np.array(cmap(np.int64(scores_norm[ii]))) *255)[:3]))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if save_pref:
            print(f'saving to file {save_pathFull}')
            out = cv2.VideoWriter(save_pathFull, fourcc, Fs, (np.int64(vid_width), np.int64(vid_height)))


        ## Main loop to pull out displacements in each video   
        ind_concat = int(np.hstack([0 , np.cumsum(numFrames_allFiles)])[vidNums_toUse[0]])

        fps = 0
        tic_fps = time.time()
        for iter_vid , vidNum_iter in enumerate(vidNums_toUse):
            path_vid = path_vid_allFiles[vidNum_iter]
            vid = imageio.get_reader(path_vid,  'ffmpeg')

    #         numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            numFrames = 600

    #         frameToSet = 0
    #         video.set(1,frameToSet)

            for iter_frame , new_frame in enumerate(vid):
                    
    #             ind_currentVid = np.int64(video.get(cv2.CAP_PROP_POS_FRAMES))
                if iter_frame >= numFrames:
                    break
    #             ok, new_frame = video.read()

                for ii in range(positions_toUse.shape[0]):
                    pointInds_tracked_tuple = tuple(np.int64(np.squeeze(positions_toUse[ii,:,ind_concat])))
                    cv2.circle(new_frame,pointInds_tracked_tuple, dot_size, colormap_tuples[ii], -1)
                if save_pref:
                    out.write(new_frame)


                cv2.putText(new_frame, f'frame #: {iter_frame}/{numFrames}', org=(10,20), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
                cv2.putText(new_frame, f'total frame #: {ind_concat+1}/{positions_toUse.shape[2]}', org=(10,60), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
                cv2.putText(new_frame, f'fps: {np.uint32(fps)}', org=(10,80), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
                cv2.putText(new_frame, f'factor num: {factor_iter+1} / {np.max(factors_toShow)+1}', org=(10,100), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
                cv2.imshow('test',new_frame)


                k = cv2.waitKey(1) & 0xff
                if k == 27 : break

                ind_concat = ind_concat+1


                if ind_concat%fps_counterPeriod==0:
                    elapsed = time.time() - tic_fps
                    fps = fps_counterPeriod/elapsed
                    if printFPS_pref:
                        print(fps)
                    tic_fps = time.time()


    out.release()
    #video.release()
    cv2.destroyAllWindows()
    
    
def plot_factors_full(config_filepath, factors_np, freqs_Sxx, Sxx_allPixels_normFactor):
    """
    plots the full set of factors

    Parameters
    ----------
    config_filepath (Path): path to the config file
    factors_np ():
    freqs_Sxx ():
    Sxx_allPixels_normFactor  ():

    Returns
    -------

    """

    config = helpers.load_config(config_filepath)
    Fs = config['vid_Fs']

    factors_toUse = factors_np
    modelRank = factors_toUse[0].shape[1]

    plt.figure()
    # plt.plot(np.arange(factors_toUse.factors(4)[0][2].shape[0])/Fs , factors_toUse.factors(4)[0][2])
    # factors_temporal = scipy.stats.zscore(factors_toUse[2][:,:] , axis=0)
    factors_temporal = factors_toUse[2][:,:]
    # factors_temporal = scipy.stats.zscore(factors_temporal_reconstructed , axis=0)
    # plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,:])
    plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,])
    # plt.plot(factors_temporal[:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('time (s)')
    plt.ylabel('a.u.')

    plt.figure()
    # plt.plot(np.arange(factors_toUse.factors(4)[0][2].shape[0])/Fs , factors_toUse.factors(4)[0][2])
    # factors_temporal = scipy.stats.zscore(factors_toUse[2][:,:] , axis=0)
    factors_temporal = factors_toUse[2][:,:] * (np.mean(Sxx_allPixels_normFactor , axis=1)[:,None] **(2/5))
    # factors_temporal = scipy.stats.zscore(factors_temporal_reconstructed , axis=0)
    # plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,:])
    plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,:])
    # plt.plot(factors_temporal[:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('time (s)')
    plt.ylabel('a.u.')

    plt.figure()
    plt.plot(freqs_Sxx , (factors_toUse[1][:,:]))
    # plt.plot(freqXaxis , (factors_toUse[1][:,:]))
    # plt.plot(f , (factors_toUse[1][:,:]))
    # plt.plot((factors_toUse[1][:,:]))
    plt.legend(np.arange(modelRank)+1)
    plt.xscale('log')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('a.u.')
    # plt.xscale('log')

    plt.figure()
    plt.plot(factors_toUse[3][:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('x vs. y')
    plt.ylabel('a.u.')

    plt.figure()
    plt.plot(factors_toUse[0][:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('pixel number')
    plt.ylabel('a.u.')


    plt.figure()
    plt.imshow(np.single(np.corrcoef(factors_toUse[2][:,:].T)))

    input_dimRed = factors_toUse[2][:,:]
    # input_dimRed_meanSub = 
    pca = sklearn.decomposition.PCA(n_components=modelRank-2)
    # pca = sk.decomposition.FactorAnalysis(n_components=3)
    pca.fit(np.single(input_dimRed).transpose())
    output_PCA = pca.components_.transpose()
    # scores_points = np.dot(ensemble.factors(4)[0][2] , output_PCA)

    plt.figure()
    plt.plot(output_PCA[:,5])

    config['modelRank'] = modelRank
    helpers.save_config(config, config_filepath)

    return factors_temporal
    
    
def correlations(config_filepath, factors_np):
    """
    finds correlations between the factors

    Parameters
    ----------
    config_filepath (Path): path to the config file
    factors_np ():

    Returns
    -------

    """

    config = helpers.load_config(config_filepath)
    modelRank = config['modelRank']

    input_factors = factors_np
    factors_xcorr = np.zeros((input_factors[2].shape[0] , input_factors[2].shape[1] , input_factors[2].shape[1]))
    for ii in tqdm(range(input_factors[2].shape[1]),desc="correlations"):
        for jj in range(input_factors[2].shape[1]):
            factors_xcorr[:,ii,jj] = scipy.signal.correlate(input_factors[2][:,ii] , input_factors[2][:,jj] , mode='same')
        
    for ii in range(factors_xcorr.shape[1]): 
        plt.figure()
        plt.plot(np.squeeze(factors_xcorr[:,ii,:]))
        plt.legend(np.arange(modelRank)+1)  
        
    return factors_xcorr

def more_factors_videos(config_filepath, factors_np, positions_convDR_absolute):
    """
    creates videos of points colored by a variety of factors

    Parameters
    ----------
    config_filepath (Path): path to the config file
    factors_np ():
    positions_convDR_absolute ():

    Returns
    -------

    """

    config = helpers.load_config(config_filepath)
    Fs = config['vid_Fs']
    vid_width = config['vid_width']
    vid_height = config['vid_height']
    numFrames_allFiles = config['numFrames_allFiles']
    path_vid_allFiles = config['path_vid_allFiles']

    # Display video of factors
    factors_toShow = np.arange(factors_np[0].shape[1])  # zero-indexed
    # factors_toShow = [3]  # zero-indexed

    for factor_iter in factors_toShow:

        # vidNums_toUse = range(numVids) ## note zero indexing!
        vidNums_toUse = config['vidNums_toUse'] ## note zero indexing!

        if type(vidNums_toUse) == int:
            vidNums_toUse = np.array([vidNums_toUse])

        dot_size = 2

        printFPS_pref = 0
        fps_counterPeriod = 10 ## number of frames to do a tic toc over

    #     modelRank_toUse = 5
        factor_toShow = factor_iter+1
        save_pref = config['tca_vid_save']

        # save_dir = "F:\\RH_Local\\Rich data\\camera data"
        save_dir = config['tca_vid_dir']
        
        save_fileName = f'factor {factor_toShow}'
        # save_pathFull = f'{save_dir}\\{save_fileName}.avi'
        save_pathFull = f'{save_dir}/{save_fileName}.avi'
        config[f'path_{save_fileName}'] = save_pathFull
        helpers.save_config(config, config_filepath)

        # ensemble_toUse = ensemble
        ensemble_toUse = factors_np
        positions_toUse = positions_convDR_absolute

        factor_toShow = factor_toShow-1
        # input_scores = ensemble_toUse.factors(modelRank_toUse)[0][0]
        input_scores = np.single(ensemble_toUse[0])

        range_toUse = np.ceil(np.max(input_scores[:,factor_toShow]) - np.min(input_scores[:,factor_toShow])) + 1
        offset_toUse = np.min(input_scores[:,factor_toShow])
        scores_norm = input_scores[:,factor_toShow] - offset_toUse
        scores_norm = (scores_norm / np.max(scores_norm)) *1000
        cmap = matplotlib.cm.get_cmap('hot', 1000)
        # cmap_viridis(np.arange(range_toUse))

        colormap_tuples =  list(np.arange(positions_toUse.shape[0]))
        for ii in range(positions_toUse.shape[0]):
            colormap_tuples[ii] = list(np.flip((np.array(cmap(np.int64(scores_norm[ii]))) *255)[:3]))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if save_pref:
            print(f'saving to file {save_pathFull}')
            out = cv2.VideoWriter(save_pathFull, fourcc, Fs, (np.int64(vid_width), np.int64(vid_height)))


        ## Main loop to pull out displacements in each video   
        ind_concat = int(np.hstack([0 , np.cumsum(numFrames_allFiles)])[vidNums_toUse[0]])

        fps = 0
        tic_fps = time.time()
        for iter_vid , vidNum_iter in enumerate(vidNums_toUse):
            path_vid = path_vid_allFiles[vidNum_iter]
            vid = imageio.get_reader(path_vid,  'ffmpeg')

    #         numFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            numFrames = 600

    #         frameToSet = 0
    #         video.set(1,frameToSet)

            for iter_frame , new_frame in enumerate(vid):

    #             ind_currentVid = np.int64(video.get(cv2.CAP_PROP_POS_FRAMES))
                if iter_frame >= numFrames:
                    break
    #             ok, new_frame = video.read()

                for ii in range(positions_toUse.shape[0]):
                    pointInds_tracked_tuple = tuple(np.int64(np.squeeze(positions_toUse[ii,:,ind_concat])))
                    cv2.circle(new_frame,pointInds_tracked_tuple, dot_size, colormap_tuples[ii], -1)
                if save_pref:
                    out.write(new_frame)

    #             Sxx_frameNum = round( ind_currentVid / (positions_toUse.shape[2] / Sxx_allPixels.shape[2]) ,1)
                cv2.putText(new_frame, f'frame #: {iter_frame}/{numFrames}', org=(10,20), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
    #             cv2.putText(new_frame, f'frame #: {Sxx_frameNum}', org=(10,20), fontFace=1, fontScale=1, color=(255,255,255), thickness=2)
    #             cv2.putText(new_frame, f'vid #: {iter+1}/{len(vidNums_toUse)}', org=(10,40), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
                cv2.putText(new_frame, f'total frame #: {ind_concat+1}/{positions_toUse.shape[2]}', org=(10,60), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
                cv2.putText(new_frame, f'fps: {np.uint32(fps)}', org=(10,80), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
                cv2.putText(new_frame, f'factor num: {factor_iter+1} / {np.max(factors_toShow)+1}', org=(10,100), fontFace=1, fontScale=1, color=(255,255,255), thickness=1)
                cv2.imshow('test',new_frame)


                k = cv2.waitKey(1) & 0xff
                if k == 27 : break

                ind_concat = ind_concat+1


                if ind_concat%fps_counterPeriod==0:
                    elapsed = time.time() - tic_fps
                    fps = fps_counterPeriod/elapsed
                    if printFPS_pref:
                        print(fps)
                    tic_fps = time.time()


    out.release()
    #video.release()
    cv2.destroyAllWindows()


def factor_tsne(factors):
    """
    creates and plots tsne of the factors

    Parameters
    ----------
    factors ():

    Returns
    -------

    """

    print("Computing t-SNE embedding")
    tsne = sklearn.manifold.TSNE(n_components=2, init='pca',
                         random_state=0, perplexity=200)
    X_tsne = tsne.fit_transform(factors)
    print("Finished computing t-SNE embedding")

    factor_toCMap = 8  # 1 indexed

    plt.figure(figsize=(5, 5))
    # plt.plot(X_tsne[:,0] , X_tsne[:,1] , linewidth=0.05)
    # plt.scatter(X_tsne[:,0] , X_tsne[:,1], 'r.' , markersize=0.6)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=1.5, c=factors[:, factor_toCMap - 1], cmap='jet')


def positional_tca_workflow(config_filepath):
    """
    sequences the steps for tca of the positions of the optic flow data

    Parameters
    ----------
    config_filepath (Path): path to the config file

    Returns
    -------

    """

    print(f'== Beginning Positional TCA Workflow ==')
    tic_all = time.time()
    config = helpers.load_config(config_filepath)
    
    positions_convDR_meanSub = helpers.load_data(config_filepath, 'path_positions_convDR_meanSub')
    positions_convDR_absolute = helpers.load_data(config_filepath, 'path_positions_convDR_absolute')
    
    factors_np_positional = tca(config_filepath, positions_convDR_meanSub)

    plot_factors(config_filepath, factors_np_positional)
    if config['tca_vid_display']:
        factor_videos(config_filepath, factors_np_positional, positions_convDR_absolute)

    helpers.save_data(config_filepath, 'factors_np_positional', factors_np_positional)
    
    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End Positional TCA ==')


def full_tca_workflow(config_filepath, data_key):
    """
    sequences the steps for tca of the spectral decomposition of the optic flow data

    Parameters
    ----------
    config_filepath (Path): path to the config file

    Returns
    -------

    """

    print(f'== Beginning Full TCA Workflow ==')
    tic_all = time.time()
    config = helpers.load_config(config_filepath)
    
    Sxx_allPixels_norm = helpers.load_data(config_filepath, 'path_Sxx_allPixels_norm')
    positions_convDR_absolute = helpers.load_data(config_filepath, data_key)
    freqs_Sxx = helpers.load_data(config_filepath, 'path_freqs_Sxx')
    Sxx_allPixels_normFactor = helpers.load_data(config_filepath, 'path_Sxx_allPixels_normFactor')

    tic = time.time()
    factors_np = tca(config_filepath, Sxx_allPixels_norm[:,:,:,:])
    helpers.print_time('Decomposition completed', time.time() - tic)

    factors_temporal = plot_factors_full(config_filepath, factors_np, freqs_Sxx, Sxx_allPixels_normFactor)
    factors_xcorr = correlations(config_filepath, factors_np)

    if config['tca_vid_display']:
        more_factors_videos(config_filepath, factors_np, positions_convDR_absolute)

    factor_tsne(factors_temporal)

    helpers.save_data(config_filepath, 'factors_np', factors_np)
    helpers.save_data(config_filepath, 'factors_xcorr', factors_xcorr)
    helpers.save_data(config_filepath, 'factors_temporal', factors_temporal)
    
    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End Full TCA ==')
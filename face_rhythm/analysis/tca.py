import sys
import time
import imageio

import cv2
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

import scipy
import scipy.signal
import scipy.interpolate
import tensorly as tl
import tensorly.decomposition
import sklearn.decomposition
import sklearn.manifold

from tqdm.notebook import tqdm
from pynwb import NWBHDF5IO

from face_rhythm.util import helpers

FACTOR_NAMES = {'positional': ['points','temporal'],
                'frequential': ['points','frequential','temporal','cartesian']}


def tca(config_filepath, input_array):
    """
    computes the tca of the provided dataframe

    Parameters
    ----------
    config_filepath (Path): path to the config file
    input_array ():

    Returns
    -------
    factors_np ():

    """
    
    config = helpers.load_config(config_filepath)
    tca = config['TCA']
    pref_useGPU = tca['pref_useGPU']
    device = tca['device']
    rank = tca['rank']
    init = tca['init']
    tol = tca['tolerance']
    verbosity = tca['verbosity']
    n_iters = tca['n_iters']
    pref_concat_cartesian_dim = tca['pref_concat_cartesian_dim']
    
    tl.set_backend('pytorch')
    
    ## Prepare the input tensor
    if pref_concat_cartesian_dim:
        if config['General']['trials']:
            input_tensor = tl.tensor(np.concatenate((input_array[...,0] , input_array[...,1]) , axis=1), dtype=tl.float32, device=device, requires_grad=False)
        else:
            input_tensor = tl.tensor(np.concatenate((input_array[...,0] , input_array[...,1]) , axis=0), dtype=tl.float32, device=device, requires_grad=False)
    else:
        input_tensor = tl.tensor(input_array, dtype=tl.float32, device=device, requires_grad=False)

    print(f'Size of input (spectrogram): {input_tensor.shape}')

    
    ### Fit TCA model
    ## If the input is small, set init='svd'
    weights, factors = tensorly.decomposition.non_negative_parafac(input_tensor, init=init, tol=tol, n_iter_max=n_iters, rank=rank, verbose=verbosity)

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
    modelRank = factors_toUse[-2].shape[1]
    ## just for plotting in case 
#     if 'Fs' not in globals():
#         Fs = 120
    config = helpers.load_config(config_filepath)
    Fs = config['Video']['Fs']

    plt.figure()
    # plt.plot(np.arange(factors_toUse.factors(4)[0][2].shape[0])/Fs , factors_toUse.factors(4)[0][2])
    factors_temporal = scipy.stats.zscore(factors_toUse[-1][:,:] , axis=0)
    factors_temporal = factors_toUse[-1][:,:]
    # factors_temporal = scipy.stats.zscore(factors_temporal_reconstructed , axis=0)
    # plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,:])
    plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,])
    # plt.plot(factors_temporal[:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('time (s)')
    plt.ylabel('a.u.')


    plt.figure()
    plt.plot(factors_toUse[-2][:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('x vs. y')
    plt.ylabel('a.u.')

    plt.figure()
    plt.plot(factors_toUse[-3][:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('pixel number')
    plt.ylabel('a.u.')


    plt.figure()
    plt.imshow(np.single(np.corrcoef(factors_toUse[-1][:,:].T)))

    # input_dimRed = factors_toUse[2][:,:]
    # # input_dimRed_meanSub = 
    # pca = sk.decomposition.PCA(n_components=modelRank-2)
    # # pca = sk.decomposition.FactorAnalysis(n_components=3)
    # pca.fit(np.single(input_dimRed).transpose())
    # output_PCA = pca.components_.transpose()
    # # scores_points = np.dot(ensemble.factors(4)[0][2] , output_PCA)

    # plt.figure()
    # plt.plot(output_PCA)

def plot_trial_factor(trial_factor):
    modelRank = trial_factor.shape[1]
    plt.figure()
    plt.plot(trial_factor)
    plt.legend(np.arange(modelRank) + 1)
    plt.xlabel('trial number')
    plt.ylabel('a.u.')

    
    
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
    Fs = config['Video']['Fs']
    # This bit is just to offset the indexing due to the loss of the last dimension in the case of concatenating the cartesian dimension
    if config['TCA']['pref_concat_cartesian_dim']:
        ind_offset = 1
    else:
        ind_offset = 0

    factors_toUse = factors_np
    modelRank = factors_toUse[0].shape[1]

    plt.figure()
    # plt.plot(np.arange(factors_toUse.factors(4)[0][2].shape[0])/Fs , factors_toUse.factors(4)[0][2])
    # factors_temporal = scipy.stats.zscore(factors_toUse[2][:,:] , axis=0)
    factors_temporal = factors_toUse[-2+ind_offset][:,:]
    # factors_temporal = scipy.stats.zscore(factors_temporal_reconstructed , axis=0)
    # plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,:])
    plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,])
    # plt.plot(factors_temporal[:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('time (s)')
    plt.ylabel('a.u.')

    plt.figure()
    plt.plot(freqs_Sxx , (factors_toUse[-3+ind_offset][:,:]))
    # plt.plot(freqXaxis , (factors_toUse[1][:,:]))
    # plt.plot(f , (factors_toUse[1][:,:]))
    # plt.plot((factors_toUse[1][:,:]))
    plt.legend(np.arange(modelRank)+1)
    plt.xscale('log')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('a.u.')
    # plt.xscale('log')

    plt.figure()
    plt.plot(factors_toUse[-1+ind_offset][:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('x vs. y')
    plt.ylabel('a.u.')

    plt.figure()
    plt.plot(factors_toUse[-4+ind_offset][:,:])
    plt.legend(np.arange(modelRank)+1)
    plt.xlabel('pixel number')
    plt.ylabel('a.u.')

    config['modelRank'] = modelRank
    helpers.save_config(config, config_filepath)


def save_factors(nwb_path, factors_all, ftype, factors_temporal_interp = None, trials = False):
    factor_names = FACTOR_NAMES[ftype]
    factor_names = (['trials'] + factor_names) if trials else factor_names
    for i, factor in enumerate(factors_all):
        helpers.create_nwb_ts(nwb_path, 'TCA', f'factors_{ftype}_{factor_names[i]}', factor, 1.0)
    if factors_temporal_interp is not None:
        helpers.create_nwb_ts(nwb_path, 'TCA', f'factors_{ftype}_temporal_interp', factors_temporal_interp, 1.0)


def load_factors(nwb_path, stem):
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        tca_data = nwbfile.processing['Face Rhythm']['TCA']
        return [tca_data[ts].data[()] for ts in tca_data.time_series if stem in ts]


def trial_reshape_positional(positions, trial_inds):
    reshaped = np.zeros((trial_inds.shape[0], *positions.shape[:-1], trial_inds.shape[1]))
    for i, trial_ind in enumerate(trial_inds):
        reshaped[i, ...] = positions[..., trial_ind]
    return reshaped


def downsample_trial_inds(trial_inds, len_original, len_cqt):
    idx_cqt_originalSamples = np.round(np.linspace(0, len_original, len_cqt))
    trial_idx_cqt = np.ones((trial_inds.shape[0], 1000)) * np.nan
    for ii in range(trial_inds.shape[0]):
        retained = np.where((idx_cqt_originalSamples > trial_inds[ii, 0]) * (idx_cqt_originalSamples < trial_inds[ii, -1]))[0]
        trial_idx_cqt[ii, :retained.shape[0]] = retained
    to_keep = ~np.any(np.isnan(trial_idx_cqt),axis=0)
    return trial_idx_cqt[:,to_keep].astype(int)


def trial_reshape_frequential(positions, spectrum, trial_inds):
    trial_inds = downsample_trial_inds(trial_inds,positions.shape[-1], spectrum.shape[-2])
    reshaped = np.zeros((trial_inds.shape[0], *spectrum.shape[:-2], trial_inds.shape[1], spectrum.shape[-1]))
    for i, trial_ind in enumerate(trial_inds):
        reshaped[i, ...] = spectrum[..., trial_ind,:]
    return reshaped


def positional_tca_workflow(config_filepath, data_key):
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
    general = config['General']

    for session in general['sessions']:
        tic_session = time.time()

        positions_convDR_meanSub = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', data_key)

        if general['trials']:
            trial_inds = np.load(session['trial_inds'])
            positions_convDR_meanSub = trial_reshape_positional(positions_convDR_meanSub, trial_inds)

        factors_np_positional = tca(config_filepath, positions_convDR_meanSub.transpose(0,2,1))

        # plot_factors(config_filepath, factors_np_positional)
        # if general['trials']:
        #     plot_trial_factor(factors_np_positional[0])

        helpers.create_nwb_group(session['nwb'], 'TCA')
        save_factors(session['nwb'], factors_np_positional, 'positional', trials=general['trials'])

        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End Positional TCA ==')

def interpolate_temporal_factor(y_input , numFrames):
    """
    Interpolates the temporal component from the frequential TCA step from CQT time steps into
    camera time steps. This allows for a 1 to 1 sync of the temporal component with the camera frames.
    This step assumes non-negativity and rectifies the output to be >0.

    Parameters
    ----------
    y_input: This should be the temporal factor matrix [N,M] where N: factors, M: time steps
    numFrames: This should be the number of frames from the original camera time series
    ----------

    Returns
    ----------
    y_new: This will be the interpolated y_input
    ----------
    """

    x_old = np.linspace(0 , y_input.shape[0] , num=y_input.shape[0] , endpoint=True)
    x_new = np.linspace(0 , y_input.shape[0] , num=numFrames, endpoint=True)

    f_interp = scipy.interpolate.interp1d(x_old, y_input, kind='cubic',axis=0)
    y_new = f_interp(x_new)
    y_new[y_new <=0] = 0 # assumes non-negativity

    return y_new


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
    general = config['General']

    freqs_Sxx = helpers.load_data(config_filepath, 'freqs_Sxx')
    for session in general['sessions']:
        positions_toUse = helpers.load_nwb_ts(session['nwb'],'Optic Flow', data_key)
        Sxx_allPixels_norm = helpers.load_nwb_ts(session['nwb'], 'CQT','Sxx_allPixels_norm')
        Sxx_allPixels_normFactor = helpers.load_nwb_ts(session['nwb'], 'CQT','Sxx_allPixels_normFactor')
        if general['trials']:
            trial_inds = np.load(session['trial_inds'])
            Sxx_allPixels_norm = trial_reshape_frequential(positions_toUse, Sxx_allPixels_norm, trial_inds)

        tic = time.time()
        factors_np = tca(config_filepath, Sxx_allPixels_norm)
        helpers.print_time('Decomposition completed', time.time() - tic)

        # plot_factors_full(config_filepath, factors_np, freqs_Sxx, Sxx_allPixels_normFactor)

        if general['trials']:
            plot_trial_factor(factors_np[0])

        interp_dim = session['trial_len'] if general['trials'] else session['numFrames_total']
        if config['TCA']['pref_concat_cartesian_dim']:
            factors_temporal_interp = interpolate_temporal_factor(factors_np[-1], interp_dim)
        else:
            factors_temporal_interp = interpolate_temporal_factor(factors_np[-2], interp_dim)
        helpers.create_nwb_group(session['nwb'], 'TCA')
        save_factors(session['nwb'], factors_np, 'frequential', factors_temporal_interp, trials=general['trials'])


        helpers.print_time('total elapsed time', time.time() - tic_all)
        print(f'== End Full TCA ==')

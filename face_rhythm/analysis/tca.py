import time
import numpy as np
import torch.cuda
import gc

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
                'spectral': ['points','spectral','temporal']}



def tca(config_filepath, input_array, pref_non_negative):
    """
    computes the tca of the provided dataframe

    Args:
        config_filepath (Path): path to the config file
        input_array (np.ndarray): tensor array

    Returns:
        factors_np (list): list of factors
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
    
    tl.set_backend('pytorch')
    
    ## Prepare the input tensor
    if config['General']['trials']:
        input_tensor = tl.tensor(np.concatenate((input_array[...,0] , input_array[...,1]) , axis=1), dtype=tl.float32, device=device, requires_grad=False)
    else:
        input_tensor = tl.tensor(np.concatenate((input_array[...,0] , input_array[...,1]) , axis=0), dtype=tl.float32, device=device, requires_grad=False)


    print(f'Size of input: {input_tensor.shape}')

    
    ### Fit TCA model
    ## If the input is small, set init='svd'
    if pref_non_negative:
        weights, factors = tensorly.decomposition.non_negative_parafac(input_tensor, init=init, tol=tol, n_iter_max=n_iters, rank=rank, verbose=verbosity)
    else:
        weights, factors = tensorly.decomposition.parafac(input_tensor, init=init, tol=tol, n_iter_max=n_iters, rank=rank, verbose=verbosity)

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


def save_factors(nwb_path, factors_all, ftype, factors_temporal_interp = None, trials = False):
    """
    load factors from nwb file

    Args:
        nwb_path (str): path to nwb file
        factors_all (list): list of factors
        ftype (str): factor type
        factors_temporal_interp (np.ndarray): interpolated temporal factors (if they've been generated)
        trials (bool): whether or not we're using trials

    Returns:

    """
    factor_names = FACTOR_NAMES[ftype].copy()
    if trials:
        factor_names = ['trials'] + factor_names
    for i, factor in enumerate(factors_all):
        helpers.create_nwb_ts(nwb_path, 'TCA', f'factors_{ftype}_{factor_names[i]}', factor, 1.0)
    if factors_temporal_interp is not None:
        helpers.create_nwb_ts(nwb_path, 'TCA', f'factors_{ftype}_temporal_interp', factors_temporal_interp, 1.0)


def load_factors(nwb_path, stem):
    """
    load factors from nwb file

    Args:
        nwb_path (str): path to nwb file
        stem (str): some substring of the time series to collect

    Returns:
        factors (list): list of factors matching given stem
    """
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        tca_data = nwbfile.processing['Face Rhythm']['TCA']
        factors = [tca_data[ts].data[()] for ts in tca_data.time_series if stem in ts]
        return factors


def trial_reshape_positional(positions, trial_inds):
    """
    reshapes the positional data

    Args:
        positions (np.ndarray): original positions array
        trial_inds (np.ndarray): array of the trial indices

    Returns:
        reshaped (np.ndarray): reshaped positions array
    """
    reshaped = np.zeros((trial_inds.shape[0], *positions.shape[:-1], trial_inds.shape[1]))
    for i, trial_ind in enumerate(trial_inds):
        reshaped[i, ...] = positions[..., trial_ind]
    return reshaped


def downsample_trial_inds(trial_inds, len_original, len_cqt):
    """
    downsamples the trial indices given a new cqt length

    Args:
        trial_inds (np.ndarray): array of the trial indices
        len_original (int): length of the original trial
        len_cqt (int): length of the trial after cqt

    Returns:
        downsampled (np.ndarray): downsampled trial indices
    """

    idx_cqt_originalSamples = np.round(np.linspace(0, len_original, len_cqt))
    trial_idx_cqt = np.ones((trial_inds.shape[0], 1000)) * np.nan
    for ii in range(trial_inds.shape[0]):
        retained = np.where((idx_cqt_originalSamples > trial_inds[ii, 0]) * (idx_cqt_originalSamples < trial_inds[ii, -1]))[0]
        trial_idx_cqt[ii, :retained.shape[0]] = retained
    to_keep = ~np.any(np.isnan(trial_idx_cqt),axis=0)
    downsampled = trial_idx_cqt[:,to_keep].astype(int)
    return downsampled


def trial_reshape_spectral(positions, spectrum, trial_inds):
    """
    reshapes the spectral data if the data is trial type

    Args:
        positions (np.ndarray): point positions
        spectrum (np.ndarray): spectrogram array
        trial_inds (np.ndarray): array of the trial indices

    Returns:
        reshaped (np.ndarray): reshaped spectrogram array
    """

    trial_inds = downsample_trial_inds(trial_inds,positions.shape[-1], spectrum.shape[-2])
    reshaped = np.zeros((trial_inds.shape[0], *spectrum.shape[:-2], trial_inds.shape[1], spectrum.shape[-1]))
    for i, trial_ind in enumerate(trial_inds):
        reshaped[i, ...] = spectrum[..., trial_ind,:]
    return reshaped


def set_device(config_filepath):
    config = helpers.load_config(config_filepath)
    if config['TCA']['pref_useGPU']:
        cuda_device_number = torch.cuda.current_device()
        print(f"using CUDA device: 'cuda:{cuda_device_number}'")
        config['TCA']['device'] = f'cuda:{cuda_device_number}'
    else:
        print(f"using CPU")
        config['TCA']['device'] = 'cpu'
    helpers.save_config(config, config_filepath)


def positional_tca_workflow(config_filepath, data_key):
    """
    sequences the steps for tca of the positions of the optic flow data

    Args:
        config_filepath (Path): path to the config file
        data_key (str): name of the positions on which to perform the tca

    Returns:

    """

    print(f'== Beginning Positional TCA Workflow ==')
    tic_all = time.time()
    set_device(config_filepath)
    config = helpers.load_config(config_filepath)
    general = config['General']

    for session in general['sessions']:
        tic_session = time.time()

        positions_convDR_meanSub = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', data_key)

        if general['trials']:
            trial_inds = np.load(session['trial_inds']).astype(int)
            positions_convDR_meanSub = trial_reshape_positional(positions_convDR_meanSub, trial_inds)
            positions_convDR_meanSub = positions_convDR_meanSub.transpose(0, 1, 3, 2)
        else:
            positions_convDR_meanSub = positions_convDR_meanSub.transpose(0, 2, 1)

        factors_np_positional = tca(config_filepath, positions_convDR_meanSub, 0)

        helpers.create_nwb_group(session['nwb'], 'TCA')
        save_factors(session['nwb'], factors_np_positional, 'positional', trials=general['trials'])

        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)

        del positions_convDR_meanSub, factors_np_positional

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End Positional TCA ==')

    gc.collect()


def interpolate_temporal_factor(y_input , numFrames):
    """
    Interpolates the temporal component from the frequential TCA step from CQT time steps into
    camera time steps. This allows for a 1 to 1 sync of the temporal component with the camera frames.
    This step assumes non-negativity and rectifies the output to be >0.

    Args:
        y_input (np.ndarray): This should be the temporal factor matrix [N,M] where N: factors, M: time steps
        numFrames (int): This should be the number of frames from the original camera time series

    Returns:
        y_new (np.ndarray): This will be the interpolated y_input

    """

    x_old = np.linspace(0 , y_input.shape[0] , num=y_input.shape[0] , endpoint=True)
    x_new = np.linspace(0 , y_input.shape[0] , num=numFrames, endpoint=True)

    f_interp = scipy.interpolate.interp1d(x_old, y_input, kind='cubic',axis=0)
    y_new = f_interp(x_new)
    y_new[y_new <=0] = 0 # assumes non-negativity

    return y_new


def full_tca_workflow(config_filepath, data_key_positionsTraceForInterpolation='positions_cleanup'):
    """
    sequences the steps for tca of the spectral decomposition of the optic flow data

    Args:
        config_filepath (Path): path to the config file
        data_key (str): name of the positions to use

    Returns:

    """

    print(f'== Beginning Full TCA Workflow ==')
    tic_all = time.time()
    set_device(config_filepath)
    config = helpers.load_config(config_filepath)
    general = config['General']

    for session in general['sessions']:
        positions_toUse = helpers.load_nwb_ts(session['nwb'],'Optic Flow', data_key_positionsTraceForInterpolation)
        Sxx_allPixels_norm = helpers.load_nwb_ts(session['nwb'], 'CQT','Sxx_allPixels_norm')
        if general['trials']:
            trial_inds = np.load(session['trial_inds'])
            Sxx_allPixels_norm = trial_reshape_spectral(positions_toUse, Sxx_allPixels_norm, trial_inds)

        tic = time.time()
        factors_np = tca(config_filepath, Sxx_allPixels_norm , 1)
        helpers.print_time('Decomposition completed', time.time() - tic)

        interp_dim = session['trial_len'] if general['trials'] else session['numFrames_total']
        factors_temporal_interp = interpolate_temporal_factor(factors_np[-1], interp_dim)

        helpers.create_nwb_group(session['nwb'], 'TCA')
        save_factors(session['nwb'], factors_np, 'spectral', factors_temporal_interp, trials=general['trials'])

        helpers.print_time('total elapsed time', time.time() - tic_all)
        print(f'== End Full TCA ==')

        del positions_toUse, Sxx_allPixels_norm, factors_np, factors_temporal_interp

    gc.collect()

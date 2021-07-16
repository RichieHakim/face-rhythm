from matplotlib import pyplot as plt
import numpy as np

from face_rhythm.util import helpers
from face_rhythm.analysis import tca

import pynwb

import pdb
FACTOR_UNITS = {'positional':['point number','time (s)','trials'],
                'spectral':['point number', 'frequency (Hz)','time (binned)','time (s)','trials',]}

def plot_pca_diagnostics(config_filepath):
    """
    displays some pca diagnostics like explained variance

    Args:
        output_PCA (np.ndarray): pca components
        pca (sklearn.PCA): pca object
        scores_points (np.ndarray): projected scores onto points

    Returns:

    """
    config = helpers.load_config(config_filepath)
    n_factors = config['PCA']['n_factors_to_show']
    legend = np.arange(n_factors) + 1
    Fs = config['Video']['Fs']
    for session in config['General']['sessions']:
        factors_temporal = helpers.load_nwb_ts(session['nwb'], 'PCA','factors_temporal')
        factors_points = helpers.load_nwb_ts(session['nwb'], 'PCA', 'factors_points')
        explained_variance = helpers.load_nwb_ts(session['nwb'], 'PCA', 'explained_variance')

        plt.figure()
        plt.plot(np.arange(factors_temporal[:,:n_factors].shape[0]) / Fs, factors_temporal[:,:n_factors])
        plt.legend(legend)
        plt.xlabel('time (s)')
        plt.ylabel('a.u.')
        plt.figure()
        plt.plot(factors_points[:,:n_factors])
        plt.legend(legend)
        plt.xlabel('point number')
        plt.ylabel('a.u.')
        plt.figure()
        plt.plot(np.cumsum(explained_variance))
        plt.xlabel('factor number')
        plt.ylabel('variance explained')



def plot_tca_factors(config_filepath):
    """
    plots the tca factors for visualization / analysis

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """
    config = helpers.load_config(config_filepath)
    Fs = config['Video']['Fs']
    ftype = config['TCA']['ftype']
    factor_units = FACTOR_UNITS[ftype].copy()
    if not config['General']['trials']:
        factor_units.remove('trials')
    for session in config['General']['sessions']:
        factors = tca.load_factors(session['nwb'], ftype)
        model_rank = factors[0].shape[1]
        legend = np.arange(model_rank) + 1
        for i, factor in enumerate(factor_units):
            plt.figure()
            if 'time' in factor:
                plt.plot(np.arange(factors[i].shape[0]) / Fs,factors[i])
            elif 'freq' in  factor:
                freqs_Sxx = helpers.load_nwb_ts(session['nwb'],'CQT','freqs_Sxx_toUse')
                plt.plot(freqs_Sxx, factors[i])
                plt.xscale('log')
            else:
                plt.plot(factors[i])
            plt.legend(legend)
            plt.xlabel(factor_units[i])
            plt.ylabel('a.u.')
        plt.figure()
        plt.imshow(np.single(np.corrcoef(factors[-1][:, :].T)))
        plt.colorbar()


def plot_cqt(config_filepath, Sxx_toUse, positions_toUse, xy_toUse='x', dot_toUse=0):
    """
    displays a cqt generated spectrogram for one dot

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """
    config = helpers.load_config(config_filepath)
    # for session in config['General']['sessions']:
        # Sxx_allPixels_norm = helpers.load_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_norm')
        # Sxx_allPixels_normFactor = helpers.load_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_normFactor')

        # plt.figure()
        # plt.imshow(Sxx_allPixels_norm[config['CQT']['pixelNum_toUse'], :, :, 0], aspect='auto', cmap='hot', origin='lower')

        # plt.figure()
        # plt.plot(Sxx_allPixels_normFactor)

    if xy_toUse == 'x':
        xy_toUse = 1
    else:
        xy_toUse = 0
    
    config = helpers.load_config(config_filepath)
    nwb_path = config['General']['sessions'][0]['nwb']
    for session in config['General']['sessions']:

        hop_length      = config['CQT']['hop_length']
        fmin_rough      = config['CQT']['fmin_rough']
        fmax_rough      = config['CQT']['fmax_rough']
        fmin            = config['CQT']['fmin']
        sampling_rate   = config['CQT']['sampling_rate']
        n_bins          = config['CQT']['n_bins']
        bins_per_octave = config['CQT']['bins_per_octave']
        filter_scale    = config['CQT']['filter_scale']
        gamma           = config['CQT']['gamma']
        spectrogram_exponent = config['CQT']['spectrogram_exponent']
        normalization_factor = config['CQT']['normalization_factor']

        with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
            nwbfile = io.read()
            Sxx = np.array(nwbfile.processing['Face Rhythm']['CQT'][Sxx_toUse].data[dot_toUse, :,:, xy_toUse])
            positions = np.array(nwbfile.processing['Face Rhythm']['Optic Flow'][positions_toUse].data[dot_toUse, xy_toUse, :])
            freqs_Sxx = np.array(nwbfile.processing['Face Rhythm']['CQT']['freqs_Sxx_toUse'].data)
            Sxx_xAxis = np.array(nwbfile.processing['Face Rhythm']['CQT']['Sxx_xAxis'].data)
            # Sxx_xAxis = config['CQT']['Sxx_xAxis']
            
        fig, axs = plt.subplots(2, sharex=True, figsize=(8,8))
        axs[0].plot(np.arange(len(positions))/sampling_rate , positions)
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('positional displacement (pixels)')
                
        axs[1].imshow(
            Sxx, 
            extent= [Sxx_xAxis[0], Sxx_xAxis[-1],
                    0, freqs_Sxx.shape[0]],
            aspect='auto',
            origin='lower',
            cmap='hot')
        
        ticks_toUse = np.arange(0,len(freqs_Sxx) ,3)    
        axs[1].set_yticks(ticks_toUse)
        axs[1].set_yticklabels(np.round(freqs_Sxx[ticks_toUse],3))
        axs[1].set_ylabel('frequency of filter (Hz)')
        axs[1].set_xlabel('time (s)')
        axs[1].set_title('filters (real component)')

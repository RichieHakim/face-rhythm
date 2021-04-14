from matplotlib import pyplot as plt
import numpy as np

from face_rhythm.util import helpers
from face_rhythm.analysis import tca

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
                freqs_Sxx = helpers.load_nwb_ts(session['nwb'],'CQT','freqs_Sxx')
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


def plot_cqt(config_filepath):
    """
    displays a cqt generated spectrogram for one pixel

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """
    config = helpers.load_config(config_filepath)
    for session in config['General']['sessions']:
        Sxx_allPixels_norm = helpers.load_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_norm')
        Sxx_allPixels_normFactor = helpers.load_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_normFactor')

        plt.figure()
        plt.imshow(Sxx_allPixels_norm[config['CQT']['pixelNum_toUse'], :, :, 0], aspect='auto', cmap='hot', origin='lower')

        plt.figure()
        plt.plot(Sxx_allPixels_normFactor)

from matplotlib import pyplot as plt
import numpy as np

from face_rhythm.util import helpers
from face_rhythm.analysis import tca


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
    for session in config['General']['sessions']:
        scores_points = helpers.load_nwb_ts(session['nwb'], 'PCA','scores_points')
        explained_variance = helpers.load_nwb_ts(session['nwb'], 'PCA', 'explained_variance')
        output_PCA = helpers.load_nwb_ts(session['nwb'], 'PCA', 'pc_components')

        plt.figure()
        plt.plot(output_PCA[:,:3])
        plt.figure()
        plt.plot(explained_variance)
        plt.figure()
        plt.plot(output_PCA[:,0] , output_PCA[:,1]  , linewidth=.1)
        plt.figure()
        plt.plot(scores_points[:,:3])


def plot_positional_tca_factors(config_filepath):
    """
    plots the tca factors for visualization / analysis

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """
    config = helpers.load_config(config_filepath)
    Fs = config['Video']['Fs']
    factor_units = ['time (s)','x vs. y','pixel number']
    for session in config['General']['sessions']:
        factors = tca.load_factors(session['nwb'], 'positional')
        model_rank = factors[0].shape[1]
        legend = np.arange(model_rank) + 1
        for i in range(-3,0):
            plt.figure()
            if i == -3:
                plt.plot(np.arange(factors[i].shape[0]) / Fs,factors[i])
            else:
                plt.plot(factors[i])
            plt.legend(legend)
            plt.xlabel(factor_units[i])
            plt.ylabel('a.u.')
        plt.figure()
        plt.imshow(np.single(np.corrcoef(factors[-1][:, :].T)))
        if config['General']['trials']:
            plot_trial_factor(factors[0])


def plot_trial_factor(trial_factor):
    """
    plots the trial factors for visualization / analysis

    Args:
        trial_factor (np.ndarray): plot the trial dimension if it exists

    Returns:

    """
    modelRank = trial_factor.shape[1]
    plt.figure()
    plt.plot(trial_factor)
    plt.legend(np.arange(modelRank) + 1)
    plt.xlabel('trial number')
    plt.ylabel('a.u.')


def plot_factors_full(config_filepath, factors_np, freqs_Sxx):
    """
    plots the full set of factors

    Args:
        config_filepath (Path): path to the config file
        factors_np (list): list of factors
        freqs_Sxx (np.ndarray): frequency array

    Returns:

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
    factors_temporal = factors_toUse[-2 + ind_offset][:, :]
    # factors_temporal = scipy.stats.zscore(factors_temporal_reconstructed , axis=0)
    # plt.plot(np.arange(factors_temporal.shape[0])/Fs, factors_temporal[:,:])
    plt.plot(np.arange(factors_temporal.shape[0]) / Fs, factors_temporal[:, ])
    # plt.plot(factors_temporal[:,:])
    plt.legend(np.arange(modelRank) + 1)
    plt.xlabel('time (s)')
    plt.ylabel('a.u.')

    plt.figure()
    plt.plot(freqs_Sxx, (factors_toUse[-3 + ind_offset][:, :]))
    # plt.plot(freqXaxis , (factors_toUse[1][:,:]))
    # plt.plot(f , (factors_toUse[1][:,:]))
    # plt.plot((factors_toUse[1][:,:]))
    plt.legend(np.arange(modelRank) + 1)
    plt.xscale('log')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('a.u.')
    # plt.xscale('log')

    plt.figure()
    plt.plot(factors_toUse[-1 + ind_offset][:, :])
    plt.legend(np.arange(modelRank) + 1)
    plt.xlabel('x vs. y')
    plt.ylabel('a.u.')

    plt.figure()
    plt.plot(factors_toUse[-4 + ind_offset][:, :])
    plt.legend(np.arange(modelRank) + 1)
    plt.xlabel('pixel number')
    plt.ylabel('a.u.')

    config['modelRank'] = modelRank
    helpers.save_config(config, config_filepath)
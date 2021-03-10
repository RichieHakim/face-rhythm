import numpy as np
from face_rhythm.util import helpers
import time
import scipy.signal
from matplotlib import pyplot as plt

def clean_displacements(config_filepath, displacements):
    """
    cleans and integrates a set of displacements according to a set of parameters

    Parameters
    ----------
    config_filepath (Path): path to the config file
    displacements (np.ndarray): array of displacements

    Returns
    -------
    positions_new_sansOutliers (np.ndarray): positions
    positions_new_absolute_sansOutliers (np.ndarray): absolute positions
    """

    config = helpers.load_config(config_filepath)
    clean = config['Clean']

    outlier_threshold_positions = clean['outlier_threshold_positions']
    outlier_threshold_displacements = clean['outlier_threshold_displacements']
    framesHalted_beforeOutlier = clean['framesHalted_beforeOutlier']
    framesHalted_afterOutlier = clean['framesHalted_afterOutlier']
    relaxation_factor = clean['relaxation_factor']

    pointInds_toUse = helpers.load_data(config_filepath, 'pointInds_toUse')

    ## Remove flagrant outliers from displacements
    displacements_simpleOutliersRemoved = displacements * (np.abs(displacements) < outlier_threshold_displacements)

    ## Make a convolutional kernel for extending the outlier trace
    kernel = np.zeros(np.max(np.array([framesHalted_beforeOutlier, framesHalted_afterOutlier])) * 2 + 1)
    kernel_center = int(np.ceil(len(kernel) / 2))
    kernel[kernel_center - framesHalted_beforeOutlier: kernel_center] = 1;
    kernel[kernel_center: kernel_center + framesHalted_afterOutlier] = 1

    ## Make integrated position traces from the displacement traces
    positions_new = np.zeros_like(displacements)  # preallocation
    for ii in range(displacements_simpleOutliersRemoved.shape[2]):
        if ii == 0:
            tmp = np.squeeze(pointInds_toUse) * 0
        else:
            tmp = positions_new[:, :, ii - 1] + displacements_simpleOutliersRemoved[:, :, ii]  # heres the integration
        positions_new[:, :, ii] = tmp - (tmp) * relaxation_factor  # and the relaxation

    ## Define outliers, then extend the outlier trace to include the outlier kernel (before and after a threshold event)
    tic = time.time()
    positions_tracked_outliers = (np.abs(positions_new) > outlier_threshold_positions)
    positions_tracked_outliers_extended = np.apply_along_axis(lambda m: scipy.signal.convolve(m, kernel, mode='same'),
                                                              axis=2, arr=positions_tracked_outliers)
    positions_tracked_outliers_extended = positions_tracked_outliers_extended > 0

    ## Make outlier timepoints zero in 'displacements'
    displacements_sansOutliers = displacements_simpleOutliersRemoved * (~positions_tracked_outliers_extended)

    ## Make a new integrated position traces array the displacement traces, but now with the outliers set to zero
    positions_new_sansOutliers = np.zeros_like(displacements_sansOutliers)
    for ii in range(displacements_sansOutliers.shape[2]):
        if ii == 0:
            tmp = np.squeeze(pointInds_toUse) * 0
        else:
            tmp = positions_new_sansOutliers[:, :, ii - 1] + displacements_sansOutliers[:, :, ii]
        positions_new_sansOutliers[:, :, ii] = tmp - (tmp) * relaxation_factor

    positions_new_absolute_sansOutliers = positions_new_sansOutliers + np.squeeze(pointInds_toUse)[:, :, None]

    return positions_new_sansOutliers, positions_new_absolute_sansOutliers


def clean_workflow(config_filepath):
    """
    sequences all steps for cleaning the optic flow data

    Parameters
    ----------
    config_filepath (Path): path to the config file

    Returns
    -------

    """

    print(f'== Beginning outlier removal ==')
    tic_all = time.time()

    config = helpers.load_config(config_filepath)
    general = config['General']
    video = config['Video']

    for session in general['sessions']:
        tic_session = time.time()
        displacements = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', 'displacements')
        positions_new_sansOutliers, positions_new_absolute_sansOutliers = clean_displacements(config_filepath, displacements)

        helpers.create_nwb_ts(session['nwb'], 'Optic Flow', 'positions', positions_new_sansOutliers, video['Fs'])
        helpers.create_nwb_ts(session['nwb'], 'Optic Flow', 'positions_absolute', positions_new_absolute_sansOutliers,
                              video['Fs'])
        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End outlier removal ==')


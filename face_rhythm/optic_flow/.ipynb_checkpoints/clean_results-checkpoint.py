import numpy as np
from face_rhythm.util import helpers
import time
import scipy.signal
from matplotlib import pyplot as plt


def clean_workflow(config_filepath):
    print(f'== Beginning outlier removal ==')
    tic_all = time.time()
    
    tic = time.time()
    config = helpers.load_config(config_filepath)

    outlier_threshold_positions = config['outlier_threshold_positions']
    outlier_threshold_displacements = config['outlier_threshold_displacements']
    framesHalted_beforeOutlier = config['framesHalted_beforeOutlier']
    framesHalted_afterOutlier = config['framesHalted_afterOutlier']

    displacements = helpers.load_data(config_filepath, 'path_displacements')
    pointInds_toUse = helpers.load_data(config_filepath, 'path_pointInds_toUse')
    print(f'Files Loaded. Elapsed time: {round(time.time() - tic , 1)} seconds')

    relaxation_factor = 0.01 # This is the speed at which the integrated position exponentially relaxes back to its anchored position

    ## Remove flagrant outliers from displacements
    tic = time.time()
    displacements_simpleOutliersRemoved = displacements * (np.abs(displacements) < outlier_threshold_displacements)
    print(f'Flagrant outliers removed. Elapsed time: {round(time.time() - tic , 1)} seconds')

    ## Make a convolutional kernel for extending the outlier trace
    kernel = np.zeros(np.max(np.array([framesHalted_beforeOutlier , framesHalted_afterOutlier])) * 2 + 1)
    kernel_center = int(np.ceil(len(kernel)/2))
    kernel[kernel_center - framesHalted_beforeOutlier : kernel_center] = 1; kernel[kernel_center : kernel_center + framesHalted_afterOutlier] = 1

    ## Make integrated position traces from the displacement traces
    tic = time.time()
    positions_new = np.zeros_like(displacements)  # preallocation
    for ii in range(displacements_simpleOutliersRemoved.shape[2]):
        if ii==0:
            tmp = np.squeeze(pointInds_toUse)*0
        else: 
            tmp = positions_new[:,:,ii-1] + displacements_simpleOutliersRemoved[:,:,ii]  # heres the integration
        positions_new[:,:,ii] = tmp - (tmp)*relaxation_factor  # and the relaxation
        
    print(f'Made integrated position traces. Elapsed time: {round(time.time() - tic , 1)} seconds')

    ## Define outliers, then extend the outlier trace to include the outlier kernel (before and after a threshold event)
    tic = time.time()
    positions_tracked_outliers = (np.abs(positions_new) > outlier_threshold_positions)
    positions_tracked_outliers_extended = np.apply_along_axis(lambda m: scipy.signal.convolve(m , kernel, mode='same'), axis=2, arr=positions_tracked_outliers)
    positions_tracked_outliers_extended = positions_tracked_outliers_extended > 0
    print(f'Made extended outliers trace. Elapsed time: {round(time.time() - tic , 1)} seconds')

    ## Make outlier timepoints zero in 'displacements'
    tic = time.time()
    displacements_sansOutliers = displacements_simpleOutliersRemoved * (~positions_tracked_outliers_extended)
    print(f'All outliers removed. Elapsed time: {round(time.time() - tic , 1)} seconds')

    ## Make a new integrated position traces array the displacement traces, but now with the outliers set to zero
    tic = time.time()
    positions_new_sansOutliers = np.zeros_like(displacements_sansOutliers)
    for ii in range(displacements_sansOutliers.shape[2]):
        if ii==0:
            tmp = np.squeeze(pointInds_toUse)*0
        else: 
            tmp = positions_new_sansOutliers[:,:,ii-1] + displacements_sansOutliers[:,:,ii]
        positions_new_sansOutliers[:,:,ii] = tmp - (tmp)*relaxation_factor
    print(f'Made a new integrated position. Elapsed time: {round(time.time() - tic , 1)} seconds')
    
    tic = time.time()
    positions_new_absolute_sansOutliers = positions_new_sansOutliers + np.squeeze(pointInds_toUse)[:, :, None]
    print(f'Final absolute position trace. Elapsed time: {round(time.time() - tic , 1)} seconds')
    
    pixelNum_toUse = 300
    plt.figure()
    plt.plot(positions_new_sansOutliers[pixelNum_toUse, 0, :])
    plt.show()
    
    tic = time.time()
    helpers.save_data(config_filepath, 'positions', positions_new_sansOutliers)
    helpers.save_data(config_filepath, 'positions_absolute', positions_new_absolute_sansOutliers)
    print(f'Files saved. Elapsed time: {round(time.time() - tic , 1)} seconds')
    
    toc = time.time() - tic_all
    print(f'total elapsed time: {round(toc/60,2)} minutes')
    print(f'== End outlier removal ==')
import numpy as np

def clean_workflow(displacements):
    outlier_threshold_positions = 40 ## in pixels. If position goes past this, short time window before and including outlier timepoint has displacement set to 0 
    outlier_threshold_displacements = 6 ## in pixels. If displacement goes past this, displacement set to 0 at those time points
    framesHalted_beforeOutlier = 30 # in frames. best to make even
    framesHalted_afterOutlier = 10 # in frames. best to make even

    relaxation_factor = 0.01 # This is the speed at which the integrated position exponentially relaxes back to its anchored position

    print(f'== Beginning outlier removal ==')
    tic_all = time.time()

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
        print(f'Made first integrated position traces. Elapsed time: {round(time.time() - tic , 1)} seconds')


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
   
    return positions_new_sansOutliers
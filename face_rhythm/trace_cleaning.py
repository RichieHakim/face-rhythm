from typing import Union
from pathlib import Path

import numpy as np

from .util import FR_Module
from . import h5_handling

class Trace_cleaner(FR_Module):
    """
    Class for cleaning traces of tracked points.
    This class finds frames where there are violations in the 
     displacement or velocity of the tracked points. 
     Displacements are frozen in place for a specified number 
     of frames before and after the violation.
    RH 2022
    """
    def __init__(
        self,
        
        path_traces: Union[str, Path]=None,
        field_traces: str='points_tracked',

        traces: Union[np.ndarray, dict]=None,
        point_positions: np.ndarray=None,

        thresh_displacement: float=25,
        thresh_velocity: float=3,
        framesHalted_before: int=25,
        framesHalted_after: int=25,

        verbose: int=1,
    ):
        super().__init__()

        ## Assertions
        ### Assert that either traces or path_traces is specified
        assert traces is not None or path_traces is not None, "FR ERROR: traces or path_traces must be specified"
        ### Assert that if path traces is specified, that field_traces is also specified
        if path_traces is not None:
            assert field_traces is not None, "FR ERROR: If path_traces is specified, field_traces must also be specified"
        ### Assert that if path_traces is specified, that it is a string or a pathlib.Path object
        if path_traces is not None:
            assert isinstance(path_traces, str) or isinstance(path_traces, Path), "FR ERROR: path_traces must be a string or pathlib.Path object"
        ### Assert that if field_traces is specified, that it is a string
        if field_traces is not None:
            assert isinstance(field_traces, str), "FR ERROR: field_traces must be a string"

        ## Set variables
        self._path_traces = str(path_traces)
        self._field_traces = str(field_traces)
        self.thresh_position = float(thresh_displacement)
        self.thresh_velocity = float(thresh_velocity)
        self.framesHalted_before = int(framesHalted_before)
        self.framesHalted_after = int(framesHalted_after)
        self._verbose = int(verbose)

        ## Load traces
        self.traces = h5_handling.simple_load(path=self._path_traces, verbose=(self._verbose > 1))[self._field_traces] if traces is None else traces
        ## If traces is a numpy array, convert it to a dictionary
        if isinstance(self.traces, np.ndarray):
            self.traces = {"1": self.traces}

        ## Assert that traces is a dictionary of numpy arrays
        assert all([isinstance(val, np.ndarray) for val in self.traces.values()]), "FR ERROR: traces must be a dictionary of numpy arrays"

        ## Find outlier frames
        ### Iterate over each trace
        ### For each trace:
        ### 1. Compute the displacements trace. Which = trace - points_0
        ### 2. Compute the velocity trace. Which = diff(displacements)
        ### 3. Convert each each (y,x) pair to a magnitude (L2 norm)
        ### 4. Find outliers as frames above the thresholds for displacement and velocity
        ### 6. Compute the new trace as the sum of the displacements trace and the points_0 trace
        ### 7. Store the new trace
        self.outlier_frames = {}
        self.traces_cleaned = {}
        # for key, trace in self.traces.items():
            







def clean_displacements(config_filepath, displacements, pointInds_toUse):
    """
    cleans and integrates a set of displacements according to a set of parameters

    Args:
        config_filepath (Path): path to the config file
        displacements (np.ndarray): array of displacements
        pointInds_toUse (np.ndarray): array of point indices

    Returns:
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

    ## Remove flagrant outliers from displacements
    print('removing flagrant outliers')
    displacements_simpleOutliersRemoved = displacements * (np.abs(displacements) < outlier_threshold_displacements)
    del displacements;    gc.collect()

    ## Make integrated position traces from the displacement traces
    print('making integrated position traces')
    positions_new = np.zeros_like(displacements_simpleOutliersRemoved)  # preallocation
    for ii in range(displacements_simpleOutliersRemoved.shape[2]):
        if ii == 0:
            tmp = np.squeeze(pointInds_toUse) * 0
        else:
            tmp = positions_new[:, :, ii - 1] + displacements_simpleOutliersRemoved[:, :, ii]  # heres the integration
        positions_new[:, :, ii] = tmp - (tmp) * relaxation_factor  # and the relaxation
        
    ## Make a convolutional kernel for extending the outlier trace
    kernel = np.zeros(np.max(np.array([framesHalted_beforeOutlier, framesHalted_afterOutlier])) * 2 + 1)
    kernel_center = int(np.ceil(len(kernel) / 2))
    kernel[kernel_center - (framesHalted_beforeOutlier+1): kernel_center] = 1;
    kernel[kernel_center: kernel_center + framesHalted_afterOutlier] = 1

    ## Define outliers, then extend the outlier trace to include the outlier kernel (before and after a threshold event)
    tic = time.time()
    print('defining outliers')
    positions_new_abs = np.abs(positions_new)
    del positions_new;    gc.collect()
    positions_tracked_outliers = (positions_new_abs > outlier_threshold_positions)
    del positions_new_abs;    gc.collect()
    print('extending outliers')
    positions_tracked_outliers_extended = np.apply_along_axis(lambda m: scipy.signal.convolve(m, kernel, mode='same'),
                                                              axis=2, arr=positions_tracked_outliers)
    del positions_tracked_outliers;    gc.collect()
    positions_tracked_outliers_extended = positions_tracked_outliers_extended > 0

    ## Make outlier timepoints zero in 'displacements'
    print('making outlier displacements zero')
    displacements_sansOutliers = displacements_simpleOutliersRemoved * (~positions_tracked_outliers_extended)
    del positions_tracked_outliers_extended, displacements_simpleOutliersRemoved;    gc.collect()

    ## Make a new integrated position traces array the displacement traces, but now with the outliers set to zero
    print('making new integrated position traces')
    positions_new_sansOutliers = np.zeros_like(displacements_sansOutliers)
    for ii in range(displacements_sansOutliers.shape[2]):
        if ii == 0:
            tmp = np.squeeze(pointInds_toUse) * 0
        else:
            tmp = positions_new_sansOutliers[:, :, ii - 1] + displacements_sansOutliers[:, :, ii]
        positions_new_sansOutliers[:, :, ii] = tmp - (tmp) * relaxation_factor

    return positions_new_sansOutliers
from typing import Union
from pathlib import Path

import numpy as np

from .util import FR_Module
from . import h5_handling

class SpectralAnalayzer(FR_Module):
    """
    A class for generating normalized spectrograms for point
     displacement traces.
    The input data can either be the PointAnalyzer.h5 output
     file, or a dictionary containing a similar structure:
        {
            'points_tracked': {
                '1': traces of shape(n_frames, n_points, 2),
                '2': traces,
                ...
            },
            'point_positions': np.ndarray of shape(n_points, 2),
        }
    RH 2022
    """
    def __init__(
        self,

        path_traces: Union[str, Path]=None,

        dict_PointTracker: dict=None,

        verbose: int=1,
    ):
        super().__init__()

        ## Assertions
        ### Assert that either path_traces or dict_PointTracker is specified
        assert path_traces is not None or dict_PointTracker is not None, "FR ERROR: path_traces or dict_PointTracker must be specified"
        ### Assert that if path_traces is specified, that it is a string or a pathlib.Path object
        if path_traces is not None:
            assert isinstance(path_traces, str) or isinstance(path_traces, Path), "FR ERROR: path_traces must be a string or pathlib.Path object"
        ### Assert that if dict_PointTracker is specified, that it is a dictionary
        if dict_PointTracker is not None:
            assert isinstance(dict_PointTracker, dict), "FR ERROR: dict_PointTracker must be a dictionary"
            ### Assert that dict_PointTracker has the correct keys
            assert 'points_tracked' in dict_PointTracker.keys(), "FR ERROR: dict_PointTracker must have the keys 'points_tracked' and 'point_positions'"
            assert 'point_positions' in dict_PointTracker.keys(), "FR ERROR: dict_PointTracker must have the keys 'points_tracked' and 'point_positions'"
            ### Assert that dict_PointTracker['points_tracked'] is a dictionary
            assert isinstance(dict_PointTracker['points_tracked'], dict), "FR ERROR: dict_PointTracker['points_tracked'] must be a dictionary"
            ### Assert that dict_PointTracker['point_positions'] is a numpy array
            assert isinstance(dict_PointTracker['point_positions'], np.ndarray), "FR ERROR: dict_PointTracker['point_positions'] must be a numpy array"
            ### Assert that dict_PointTracker['point_positions'] is of shape (n_points, 2)
            assert dict_PointTracker['point_positions'].shape[1] == 2, "FR ERROR: dict_PointTracker['point_positions'] must be of shape (n_points, 2)"
            ### Assert that dict_PointTracker['points_tracked'] contains only numpy arrays
            assert all([isinstance(dict_PointTracker['points_tracked'][key], np.ndarray) for key in dict_PointTracker['points_tracked'].keys()]), "FR ERROR: dict_PointTracker['points_tracked'] must contain only numpy arrays"
            ### Assert that dict_PointTracker['points_tracked'] contains only numpy arrays of shape (n_frames, n_points, 2)
            assert all([dict_PointTracker['points_tracked'][key].shape[2] == 2 for key in dict_PointTracker['points_tracked'].keys()]), "FR ERROR: dict_PointTracker['points_tracked'] must contain only numpy arrays of shape (n_frames, n_points, 2)"
            assert all([dict_PointTracker['points_tracked'][key].shape[1] == dict_PointTracker['point_positions'].shape[0] for key in dict_PointTracker['points_tracked'].keys()]), "FR ERROR: dict_PointTracker['points_tracked'] must contain only numpy arrays of shape (n_frames, n_points, 2)"

        ## Set attributes
        self._path_traces = path_traces
        


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
            



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
            



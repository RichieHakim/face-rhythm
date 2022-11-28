from typing import Union
import time

import numpy as np
from tqdm import tqdm
import decord
import cv2
import torch
import scipy.sparse
import tensorly as tl

from .util import FR_Module
from .video_playback import FrameVisualizer

## Define TCA class as a subclass of utils.FR_Module
class TCA(FR_Module):
    """
    Class for performing Tensor Component Analysis (TCA).
    RH 2022
    """
    def __init__(
        self,
        verbose: Union[bool, int]=1,
    ):
        """
        Initialize the TCA object.

        Args:
            verbose (bool or int):
                Whether to print progress messages.
                0: No messages
                1: Warnings
                2: Info
        """
        ## Imports
        super().__init__()

        ## Set variables
        self.verbose = int(verbose)

    def fit(
        self,
        data: dict,
        names_dimensions: list=['xy', 'points', 'frequency', 'time'],
        dims_to_concatenate: list=[('xy', 'points')],
        method: str='non_negative_parafac_hals',
        params_method: dict={},
        verbose: Union[bool, int]=1,
    ):
        """
        Fit the TCA model to the data.

        Args:
            data (dict of np.ndarray):
                Dictionary of data arrays.
                Each array should have the same shape.
            names_dimensions (list of str):
                Optional. If None, then the names of the dimensions will
                 be set to ['dim0', 'dim1', ...].
                Names of the dimensions of the data arrays.
                Should be in the same order as the dimensions of the data.
            dims_to_concatenate (list of tuples of str):
                Optional. If None, then no dimensions will be concatenated.
                List of tuples of dimension names to concatenate.
                Example tuple: ('dim1', 'dim2'). Here 'dim1' will be
                 concatenated along 'dim2'. The resulting array will have
                 one less dimension than the original arrays, and 'dim2'
                 will be longer (len(dim2_new) = len(dim2_old) * len(dim1)).
                The dimensions will be concatenated in the order they are
                 specified.
        """
        ## Set attributes
        self.names_dimensions = names_dimensions
        self.dims_to_concatenate = dims_to_concatenate
        self.method = method
        self.params_method = params_method
        self._verbose = int(verbose)

        ## Assertions
        ### Assert that for each argument, the type matches the expected type
        ### Assert that the data is a dictionary, and that all the values are numpy arrays, and that all arrays have the same number of dimensions
        assert isinstance(data, dict), 'data must be a dictionary'
        assert all([isinstance(v, np.ndarray) for v in data.values()]), 'all values in data must be numpy arrays'
        assert len(set([v.ndim for v in data.values()])) == 1, 'all arrays in data must have the same number of dimensions'
        ### Assert that the names of the dimensions are unique strings, and that the number of names matches the number of dimensions in the data
        assert all([isinstance(n, str) for n in names_dimensions]), 'names of dimensions must be strings'
        assert len(names_dimensions) == len(set(names_dimensions)), 'names of dimensions must be unique'
        assert len(names_dimensions) == len(data[list(data.keys())[0]].shape), 'number of names of dimensions must match the number of dimensions in the data'
        ### Assert that the dimensions to concatenate are tuples of strings, and that the strings are in the list of names of dimensions
        assert all([isinstance(d, tuple) for d in dims_to_concatenate]), 'dimensions to concatenate must be tuples'
        assert all([all([isinstance(n, str) for n in d]) for d in dims_to_concatenate]), 'dimensions to concatenate must be tuples of strings'
        assert all([all([n in names_dimensions for n in d]) for d in dims_to_concatenate]), 'dimensions to concatenate must be tuples of strings that are in the list of names of dimensions'
        ### Assert that the method is a string
        assert isinstance(method, str), 'method must be a string'
        ### Assert that the parameters for the method are a dictionary
        assert isinstance(params_method, dict), 'parameters for method must be a dictionary'

        ## Concatenate dimensions
        ### Concatenate dimensions
        for dims in dims_to_concatenate:
            data = self._concatenate_dimensions(data, dims)
        ### Update names of dimensions
        names_dimensions = self._update_names_dimensions(names_dimensions, dims_to_concatenate)
        

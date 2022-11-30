from typing import Union
import time
from functools import partial
import gc

import numpy as np
from tqdm import tqdm
import decord
import cv2
import torch
import scipy.sparse
import tensorly as tl
import tensorly.decomposition
import einops

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
        self._verbose = int(verbose)

    def rearrange_data(
        self,
        data: dict,

        names_dims_array: list=['xy', 'points', 'frequency', 'time'],
        names_dims_array_concat: list=[('xy', 'points')],

        name_dim_dictElements: str='trials',

        method_handling_dictElements: str='concatenate',
        name_dim_concat_dictElements: str='time',

        idx_window_dictElements: list=None,
        name_dim_window_dictElements: str='time',

        # DEVICE: str='cpu',
    ):
        """
        Rearrange the data dictionary into a format that can be used 
         for TCA.
        This function allows for the data to be rearranged in the
         following ways:
            - Concatenate dimensions of the data arrays
            - Concatenate elements of the data dictionary along an
               array dimension
            - Stack elements of the data dictionary into a new
               array dimension
            - Treat dictionary elements independently
            - Select a window of data to use from each array

        Args:
            data (dict of np.ndarray):
                Dictionary of data arrays.

            names_dims_array (list of str):
                Names of the dimensions of the data arrays.
                Typically these are ['xy', 'points', 'frequency', 'time'].

            names_dims_array_concat (list of tuples of str):
                List of tuples of dimension names to concatenate.
                Example tuple: ('dim1', 'dim2'). Here 'dim1' will be
                 concatenated along 'dim2'. The resulting array will have
                 one less dimension than the original arrays, and 'dim2'
                 will be longer (len(dim2_new) = len(dim2_old) * len(dim1)).
                The dimensions will be concatenated in the order they are
                 specified.
                New dimension names will be joined with underscore:
                    example: ('dim1', 'dim2') -> 'dim1_dim2'

            name_dim_dictElements (str):
                What the different elements of the dictionary correspond
                 to. Typically this is 'trials', but it could be 'videos'.

            method_handling_dictElements (str):
                How to handle the different elements of the dictionary.
                'concatenate': Concatenate the different elements of the
                    dictionary along a specified dimension. Specify the
                    dimension to concatenate along using the 
                    name_dim_concat_dictElements argument.
                    This will result in data being a single array with
                    the same number of dimensions as the original data.
                'stack': Stack the different elements of the dictionary
                    along a new dimension. Will be new first dimension.
                    This will result in data being a single array with
                    one more dimension than the original data.
                'separate': Keep the different elements of the dictionary
                    separate. This will result in each array being a
                    separate tensor. And getting decomposed separately.
            name_dim_concat_dictElements (str):
                Name of the dimension to concatenate the dictElements
                 along. Only used if method_handling_dictElements is 'concatenate'.

            idx_window_dictElements (list of 2-tuples of int):
                Indices of the start and end of the window to use for
                 each dictEntry. Indices are inclusive.
                If None then the entire array in dictionary entry will
                 be used.
            name_dim_window_dictElements (str):
                Name of the dimension to use for the window.
                Only used if idx_window_dictElements is not None.
            
            # DEVICE (str):
            #     Device to use for the tensor. Typically 'cpu' or 'cuda'.
        """
        ## Assertions
        ### Check that the names_dims_array is a list of strings
        assert isinstance(names_dims_array, list), "names_dims_array must be a list of strings."
        assert all([isinstance(name, str) for name in names_dims_array]), "names_dims_array must be a list of strings."
        ### Check that the name_dim_dictElements is a string
        assert isinstance(name_dim_dictElements, str), "name_dim_dictElements must be a string."
        ### Check that the method_handling_dictElements is a string and is valid
        assert isinstance(method_handling_dictElements, str), "method_handling_dictElements must be a string."
        assert method_handling_dictElements in ['concatenate', 'stack', 'separate'], "method_handling_dictElements must be one of 'concatenate', 'stack', or 'separate'."
        ### Check that the name_dim_concat_dictElements is a string
        assert isinstance(name_dim_concat_dictElements, str), "name_dim_concat_dictElements must be a string."
        # ### Check that the idx_window_dictElements is a list of 2-tuples of ints
        # assert isinstance(idx_window_dictElements, list), "idx_window_dictElements must be a list of 2-tuples of ints."
        # assert all([isinstance(idx, tuple) for idx in idx_window_dictElements]), "idx_window_dictElements must be a list of 2-tuples of ints."
        # assert all([len(idx) == 2 for idx in idx_window_dictElements]), "idx_window_dictElements must be a list of 2-tuples of ints."
        # assert all([all([isinstance(idx_i, int) for idx_i in idx]) for idx in idx_window_dictElements]), "idx_window_dictElements must be a list of 2-tuples of ints."
        # ### Check that the name_dim_window_dictElements is a string
        # assert isinstance(name_dim_window_dictElements, str), "name_dim_window_dictElements must be a string."

        ## Set variables
        self._names_dims_array = names_dims_array
        self._names_dims_array_concat = names_dims_array_concat
        self._name_dim_dictElements = name_dim_dictElements
        self._method_handling_dictElements = method_handling_dictElements
        self._name_dim_dictElements_concat = name_dim_concat_dictElements
        self._idx_window_dictElements = idx_window_dictElements
        self._name_dim_window_dictElements = name_dim_window_dictElements
        # self.DEVICE = torch.device(DEVICE)

        ## Make a function for concatenating the array dimensions
        def concatenate_array_dimensions(data):
            """
            Concatenate dimensions of the data arrays.
            """
            names_dims_array_new = self._names_dims_array.copy()
            ## Use einops to concatenate dimensions
            for dims in self._names_dims_array_concat:
                dims_in = names_dims_array_new
                
                dims_out = [d for d in dims_in if d not in dims[0]]
                dims_out[dims_out.index(dims[1])] = f'({dims[0]} {dims[1]})'

                pattern = f"{''.join([d + ' ' for d in dims_in])} -> {''.join([d + ' ' for d in dims_out])}"
                data_out = einops.rearrange(data, pattern)
                
                names_dims_array_new = dims_out
            return data_out
        cat = concatenate_array_dimensions

        ## Find new names for the dimensions. Just take some code from the above function.
        self._names_dims_array_new = self._names_dims_array.copy()
        for dims in self._names_dims_array_concat:
            dims_in = self._names_dims_array_new
            dims_out = [d for d in dims_in if d not in dims[0]]
            dims_out[dims_out.index(dims[1])] = f'({dims[0]} {dims[1]})'            
            self._names_dims_array_new = dims_out
        print(f"Preparing new names for the concatenated array dimensions. From {self._names_dims_array} to {self._names_dims_array_new}.") if self._verbose > 1 else None

        
        ## Make a function for windowing the data
        def window_data(data):
            """
            Window the data. Assume numpy input.
            """
            if self._idx_window_dictElements is None:
                # print(data.shape)
                return data
            else:
                ## Get the indices of the window
                idx_window = [self._idx_window_dictElements[0], self._idx_window_dictElements[1] + 1]
                ## Window the data
                data_out = data.take(indices=range(*idx_window), axis=data.shape.index(self._name_dim_window_dictElements))
                return data_out
        win = window_data

        ## Rearrange the dict elements
        print(f"Rearranging the dict elements using method '{self._method_handling_dictElements}'.") if self._verbose > 1 else None
        if self._method_handling_dictElements == 'concatenate':
            ### Concatenate the different elements of the dictionary
            ###  along the specified dimension"
            print(f"Concatenating the different elements of the dictionary along the dimension '{self._name_dim_dictElements_concat}', corresponding to array axis: {self._names_dims_array_new.index(self._name_dim_dictElements_concat)}") if self._verbose > 1 else None
            data_out = {'0': np.concatenate(
                [cat(win(data[key])) for key in data.keys()],
                axis=self._names_dims_array_new.index(self._name_dim_dictElements_concat),
            )}
            self._names_dims_array_new[self._names_dims_array_new.index(self._name_dim_dictElements_concat)] = '(' + self._name_dim_dictElements_concat + ' ' + self._name_dim_dictElements + ')'
            self._name_dim_dictElements_new = '0'
            print(f"New names for the array dimensions: {self._names_dims_array_new}") if self._verbose > 1 else None
            print(f"New name for the dict dimension: '{self._name_dim_dictElements_new}'") if self._verbose > 1 else None
        elif self._method_handling_dictElements == 'stack':
            ### Stack the different elements of the dictionary
            ###  along the specified dimension
            data_out = {'0': np.stack(
                [cat(win(data[key])) for key in data.keys()],
                axis=0,
            )}
            self._names_dims_array_new.insert(0, self._name_dim_dictElements)
            self._name_dim_dictElements_new = '0'
        elif self._method_handling_dictElements == 'separate':
            ### Separate the different elements of the dictionary
            ###  into different arrays
            data_out = {key: cat(win(data[key])) for key in data.keys()}
            self._name_dim_dictElements_new = self._name_dim_dictElements
        
        ## Set the data
        self.data = data_out
        ## Set the names of the dimensions of the data arrays
        self.names_dims_array = self._names_dims_array_new
        ## Set the name of the dimension of the dictionary elements
        self.name_dim_dictElements = self._name_dim_dictElements_new
            

    def fit(
        self,
        data: dict=None,
        method: str='CP_NN_HALS',
        params_method: dict={
            'rank': 6, 
            'n_iter_max': 100, 
            'init': 'svd', 
            'svd': 'truncated_svd', 
            'tol': 1e-07, 
            'sparsity_coefficients': None, 
            'fixed_modes': None, 
            'nn_modes': 'all', 
            'exact': False, 
            'verbose': False, 
            'cvg_criterion': 'abs_rec_error', 
        },
        backend: str='pytorch',
        DEVICE: str='cpu',
        verbose: Union[bool, int]=1,
    ):
        """
        Fit the TCA model to the data.

        Args:
            data (dict of np.ndarray):
                Dictionary of data arrays.
                Each array should have the same shape.

        """
        ## Assert that method is valid
        assert isinstance(method, str), f"Argument 'method' must be a string."
        assert method in (valid_methods:=['CP_NN_HALS', 'CP', 'RandomizedCP', 'ConstrainedCP',]), f"Method '{method}' is not valid. Valid methods are: {valid_methods}"
        ## Assert that backend is valid
        assert isinstance(backend, str), f"Argument 'backend' must be a string."
        assert backend in (valid_backends:=['pytorch', 'numpy',]), f"Backend '{backend}' is not valid. Valid backends are: {valid_backends}"

        ## Set attributes
        self.method = method
        self._backend = backend
        self.params_method = params_method
        self.data = data if data is not None else self.data
        self._DEVICE = torch.device(DEVICE)
        self._verbose = int(verbose)

        print(f"Using device: {self._DEVICE}") if self._verbose > 1 else None
        print(f"Using method: {tl.decomposition.__dict__[method]}") if self._verbose > 1 else None

        # ## Make sure data is not complex if method is CP_NN_HALS
        # def abs_if_needed(data):
        #     if self.backend == 'pytorch':
        #         abs = torch.abs
        #     elif self.backend == 'numpy':
        #         abs = np.abs
        #     if self.method == 'CP_NN_HALS':
        #         return abs(data)
        #     else:
        #         return data
        # def prep_array(data):
        #     if self.backend == 'pytorch':
        #         ## Check if data is complex
        #         if np.iscomplexobj(data):
        #         return torch.as_tensor(data).to(self._DEVICE)

        ## Run the TCA model
        tl.set_backend('pytorch')
        print(f"Running the TCA model with method '{self.method}'.") if self._verbose > 1 else None
        self._model = tl.decomposition.__dict__[method](**self.params_method)
        # cp_all = [self._model.fit_transform(abs_if_needed(torch.as_tensor(d, device=self._DEVICE))) for d in self.data.values()]
        cp_all = [self._model.fit_transform(torch.as_tensor(d, device=self._DEVICE)) for d in self.data.values()]
        self.factors = [{key: cp.factors[ii].cpu().numpy() for ii, key in enumerate(self.names_dims_array)} for cp in cp_all]

        ## Clean up
        self._cleanup()


    def _cleanup(self):
        """
        Clear the CUDA cache and garbage collect.
        """
        if 'cuda' in self._DEVICE.type:
            for ii in range(5):
                torch.cuda.empty_cache()
                time.sleep(0.1)
                gc.collect()
                time.sleep(0.1)
        else:
            [gc.collect() for ii in range(5)]
            
            


    def _check_inputs(self, data):
        """
        Check the inputs for type and value.
        data is passed in because it is large and not set as
         an attribute until after this function is called.

        Args:
            data (dict of np.ndarray):
                Dictionary of data arrays.
                Each element of the dictionary should be a numpy
                 array. Number of dimensions should be the same
                 as the number of names in self.names_dims_array.
        """
        ## Assertions
        ### Assert that for each argument, the type matches the expected type
        ### Assert that the data is a dictionary, and that all the values are numpy arrays, and that all arrays have the same number of dimensions
        assert isinstance(data, dict), 'data must be a dictionary'
        assert all([isinstance(v, np.ndarray) for v in data.values()]), 'all values in data must be numpy arrays'
        assert len(set([v.ndim for v in data.values()])) == 1, 'all arrays in data must have the same number of dimensions'
        ### Assert that the names of the dimensions are unique strings, and that the number of names matches the number of dimensions in the data
        assert all([isinstance(n, str) for n in self.names_dimensions]), 'names of dimensions must be strings'
        assert len(self.names_dimensions) == len(set(self.names_dimensions)), 'names of dimensions must be unique'
        assert len(self.names_dimensions) == len(data[list(data.keys())[0]].shape), 'number of names of dimensions must match the number of dimensions in the data'
        ### Assert that the dimensions to concatenate are tuples of strings, and that the strings are in the list of names of dimensions
        assert all([isinstance(d, tuple) for d in self.dims_to_concatenate]), 'dimensions to concatenate must be tuples'
        assert all([all([isinstance(n, str) for n in d]) for d in self.dims_to_concatenate]), 'dimensions to concatenate must be tuples of strings'
        assert all([all([n in self.names_dimensions for n in d]) for d in self.dims_to_concatenate]), 'dimensions to concatenate must be tuples of strings that are in the list of names of dimensions'
        ### Assert that the method is a string
        assert isinstance(self.method, str), 'method must be a string'
        ### Assert that the parameters for the method are a dictionary
        assert isinstance(self.params_method, dict), 'parameters for method must be a dictionary'


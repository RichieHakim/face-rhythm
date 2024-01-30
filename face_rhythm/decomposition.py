from typing import Union
import time
from functools import partial
import gc
import copy
import re

import numpy as np
from tqdm import tqdm
import torch
import tensorly as tl
import einops

from . import util
from . import helpers

## Define TCA class as a subclass of util.FR_Module
class TCA(util.FR_Module):
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

        ## For FR_Module compatibility
        self.config = {
            'verbose': self._verbose,
        }
        self.run_info = {}
        self.run_data = {}

    def rearrange_data(
        self,
        data: dict,

        names_dims_array: list=['xy', 'points', 'frequency', 'time'],
        names_dims_concat_array: list=[['xy', 'points']],

        concat_complexDim: bool=True,
        name_dim_concat_complexDim: str='time',

        name_dim_dictElements: str='trials',
        method_handling_dictElements: str='concatenate',
        name_dim_concat_dictElements: str='time',

        idx_windows: list=None,
        name_dim_array_window: str='time',

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

            names_dims_concat_array (list of 2-element lists of str):
                List of 2-element lists of dimension names to concatenate.
                Example tuple: ('dim1', 'dim2'). Here 'dim1' will be
                 concatenated along 'dim2'. The resulting array will have
                 one less dimension than the original arrays, and 'dim2'
                 will be longer (len(dim2_new) = len(dim2_old) * len(dim1)).
                The dimensions will be concatenated in the order they are
                 specified.
                New dimension names will be joined with underscore:
                    example: ('dim1', 'dim2') -> 'dim1_dim2'

            concat_complexDim (str):
                Whether to concatenate the complex dimension.
            name_dim_concat_complexDim (str):
                Name of the array dimension to concatenate the complex
                 dimension along. Typically this should be 'time'.

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

            idx_windows (list of 2-tuples of int):
                Indices of the start and end of the window to use for
                 each dictEntry. Indices are inclusive.
                If None then the entire array in dictionary entry will
                 be used.
            name_dim_array_window (str):
                Name of the dimension to use for the window.
                Only used if idx_windows is not None.
            
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
        # ### Check that the idx_windows is a list of 2-tuples of ints
        # assert isinstance(idx_windows, list), "idx_windows must be a list of 2-tuples of ints."
        # assert all([isinstance(idx, tuple) for idx in idx_windows]), "idx_windows must be a list of 2-tuples of ints."
        # assert all([len(idx) == 2 for idx in idx_windows]), "idx_windows must be a list of 2-tuples of ints."
        # assert all([all([isinstance(idx_i, int) for idx_i in idx]) for idx in idx_windows]), "idx_windows must be a list of 2-tuples of ints."
        # ### Check that the name_dim_array_window is a string
        # assert isinstance(name_dim_array_window, str), "name_dim_array_window must be a string."

        ## Set variables
        self._names_dims_array = names_dims_array
        self._names_dims_concat_array = names_dims_concat_array
        self._name_dim_dictElements = name_dim_dictElements
        self._method_handling_dictElements = method_handling_dictElements
        self._name_dim_concat_dictElements = name_dim_concat_dictElements
        self._idx_windows = idx_windows
        self._name_dim_array_window = name_dim_array_window
        self._concat_complexDim = concat_complexDim
        self._name_dim_concat_complexDim = name_dim_concat_complexDim
        # self.DEVICE = torch.device(DEVICE)

        self.num_dictElements = len(data)
        self.shapes_dictElements = [data[key].shape for key in data.keys()]
        self._names_dictElements = list(data.keys())

        ## Check if data are complex
        isComplex = np.iscomplexobj(data[self._names_dictElements[0]])
        ## If not complex, assert that concat_complexDim is False
        if not isComplex:
            assert not self._concat_complexDim, "FR ERROR: Data are not complex, so concat_complexDim must be False." 

        ## Make a function for concatenating the array dimensions
        def concatenate_array_dimensions(data):
            """
            Concatenate dimensions of the data arrays.
            """
            names_dims_array_new = self._names_dims_array.copy()
            ## Use einops to concatenate dimensions
            for dims in self._names_dims_concat_array:
                dims_in = names_dims_array_new
                
                dims_out = [d for d in dims_in if d not in dims[0]]
                dims_out[dims_out.index(dims[1])] = f'({dims[0]} {dims[1]})'

            ## If we are concatenating the complex dimension
            if self._concat_complexDim:
                data = np.stack([data.real, data.imag], axis=-1)
                dims_in.append('complex')
                dims_out[dims_out.index(self._name_dim_concat_complexDim)] = f'({self._name_dim_concat_complexDim} complex)'

            pattern = f"{''.join([d + ' ' for d in dims_in])} -> {''.join([d + ' ' for d in dims_out])}"
            data_out = einops.rearrange(data, pattern)
            
            names_dims_array_new = dims_out
            return data_out
        cat = concatenate_array_dimensions

            
        ## Find new names for the dimensions. Just take some code from the above function.
        self._names_dims_array_new = self._names_dims_array.copy()
        for dims in self._names_dims_concat_array:
            dims_in = self._names_dims_array_new
            dims_out = [d for d in dims_in if d not in dims[0]]
            dims_out[dims_out.index(dims[1])] = f'({dims[0]} {dims[1]})'            
            self._names_dims_array_new = dims_out
        if self._concat_complexDim:
            dims_in.append('complex')
            dims_out[dims_out.index(self._name_dim_concat_complexDim)] = f'({self._name_dim_concat_complexDim} complex)'

        print(f"Preparing new names for the concatenated array dimensions. From {self._names_dims_array} to {self._names_dims_array_new}.") if self._verbose > 1 else None

        ## Assert that if we are concatenating the dictElements dimension, that it is in the list of dimensions
        if self._method_handling_dictElements == 'concatenate':
            assert self._name_dim_concat_dictElements in self._names_dims_array_new, f"Cannot concatenate the dictElements dimension {self._name_dim_concat_dictElements} because it is not in the list of dimensions {self._names_dims_array_new}. Please rename the name_dim_concat_dictElements or change the name_dim_concat_dictElements variable, possibly to a compound name like '(time complex)' or '(xy points)'."

        
        ## Make a function for windowing the data
        def window_data(data, win_idx):
            """
            Window the data. Assume numpy input.
            """
            if self._idx_windows is None:
                return data
            if self._idx_windows[win_idx] is None:
                return data
            else:
                ## Get the indices of the window
                idx_window = (int(self._idx_windows[win_idx][0]), int(self._idx_windows[win_idx][1] + 1))
                ## Window the data
                axis_window = np.where([self._name_dim_array_window in d for d in self._names_dims_array])[0][0]
                data_out = data.take(indices=range(*idx_window), axis=axis_window)
                return data_out
        win = window_data

        ## Rearrange the dict elements
        print(f"Rearranging the dict elements using method '{self._method_handling_dictElements}'.") if self._verbose > 1 else None
        if self._method_handling_dictElements == 'concatenate':
            ### Concatenate the different elements of the dictionary
            ###  along the specified dimension"
            print(f"Concatenating the different elements of the dictionary along the dimension '{self._name_dim_concat_dictElements}', corresponding to array axis: {self._names_dims_array_new.index(self._name_dim_concat_dictElements)}") if self._verbose > 1 else None
            data_out = {'0': np.concatenate(
                [cat(win(data[key],ii)) for ii,key in enumerate(data.keys())],
                axis=self._names_dims_array_new.index(self._name_dim_concat_dictElements),
            )}
            self._names_dims_array_new[self._names_dims_array_new.index(self._name_dim_concat_dictElements)] = '(' + self._name_dim_concat_dictElements + ' ' + self._name_dim_dictElements + ')'
            self._name_dim_dictElements_new = '0'
            print(f"New names for the array dimensions: {self._names_dims_array_new}") if self._verbose > 1 else None
            print(f"New name for the dict dimension: '{self._name_dim_dictElements_new}'") if self._verbose > 1 else None
        elif self._method_handling_dictElements == 'stack':
            ### Stack the different elements of the dictionary
            ###  along the specified dimension
            data_out = {'0': np.stack(
                [cat(win(data[key],ii)) for ii,key in enumerate(data.keys())],
                axis=0,
            )}
            self._names_dims_array_new.insert(0, self._name_dim_dictElements)
            self._name_dim_dictElements_new = '0'
            print(data_out['0'].shape)
        elif self._method_handling_dictElements == 'separate':
            ### Separate the different elements of the dictionary
            ###  into different arrays
            data_out = {key: cat(win(data[key],ii)) for ii,key in enumerate(data.keys())}
            self._name_dim_dictElements_new = self._name_dim_dictElements
        
        ## Set the data
        self.data = data_out
        ## Set the names of the dimensions of the data arrays
        self.names_dims_array_preDecomp = self._names_dims_array_new
        ## Set the name of the dimension of the dictionary elements
        self.name_dim_dictElements_preDecomp = self._name_dim_dictElements_new

    
    def normalize_data(
        self,
        mean_subtract: bool=False,
        std_divide: bool=False,
        dim_name: str='time',
    ):
        """
        Normalize the data.
        self.rearrange_data() must be run before this function.

        Args:
            mean_subtract (bool):
                Whether to subtract the mean from the data.
            std_divide (bool):
                Whether to divide the data by the standard deviation.
            dim_name (str):
                Name of the dimension to normalize over.
                Should be the name after rearranging the data.
        """
        ## Place params into config
        self.config['mean_subtract'] = mean_subtract
        self.config['std_divide'] = std_divide

        ## Skip everything if we don't need to normalize
        if not mean_subtract and not std_divide:
            return None

        ## Assert that the data has been rearranged
        assert self.data is not None, "self.data is None. Please run self.rearrange_data() before running self.normalize_data()."

        ## Get the dimension to normalize over
        dim_idx = self.names_dims_array_preDecomp.index(dim_name)

        ## Normalize the data
        self.data = {key: torch.as_tensor(self.data[key]) for key in self.data.keys()}
        if mean_subtract:
            arrs_mean = {key: torch.mean(self.data[key], dim=dim_idx, keepdims=True) for key in self.data.keys()}
        else:
            arrs_mean = {key: 0 for key in self.data.keys()}

        if std_divide:
            arrs_std = {key: torch.std(self.data[key], dim=dim_idx, keepdims=True) for key in self.data.keys()}
        else:
            arrs_std = {key: 1 for key in self.data.keys()}

        self.data = {key: ((self.data[key] - arrs_mean[key]) / arrs_std[key]).type(torch.float32).cpu().numpy() for key in self.data.keys()}

            
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
                The arrays should be numpy arrays.
            method (str):
                Method for fitting the TCA model.
                See Tensorly's documentation for the params_method
                 argument inputs.
                Good options are from Tensorly's decomposition methods:
                    - 'CP_NN_HALS': non-negative CP decomposition
                        using the efficient HALS algorithm.
                    - 'CP': Standard CP decomposition.
                    - 'Randomized_CP': Randomized CP decomposition.
                        Allows for large input tensors.
            params_method (dict):
                Parameters for the method. See Tensorly's documentation
                 for the params_method argument inputs for the method.
            backend (str):
                Backend for the tensorly package. Use 'pytorch' for
                 most applications.
            DEVICE (str):
                Device for the tensorly package. Set to 'cuda' if you
                 have a GPU and want to use it. Otherwise, set to 'cpu'.
            verbose (bool or int):
                Verbosity level. Set to 0 for no output. Set to 1 for
                 warnings. Set to 2 for info.

        Set Attributes:
            self._model (tensorly.decomposition.candecomp_parafac.CP):
                Fitted TCA model.
            self.factors (list of np.ndarray):
                List of the factors of the TCA model.
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


        ## Run the TCA model
        tl.set_backend(self._backend)
        print(f"Running the TCA model with method '{self.method}'.") if self._verbose > 1 else None
        self._model = tl.decomposition.__dict__[method](**self.params_method)

        self._cleanup()  ## Clean up any previous runs
        cp_all = {key: self._model.fit_transform(torch.as_tensor(d, device=self._DEVICE)) for key,d in self.data.items()}
        self.factors = {key_factor: {key: cp.factors[ii].cpu().numpy() for ii, key in enumerate(self.names_dims_array_preDecomp)} for key_factor,cp in cp_all.items()}

        ## Clean up
        self._cleanup()

        ## Place config into run_config
        self.config['method'] = method
        self.config['backend'] = backend
        self.config['params_method'] = params_method
        self.config['device'] = DEVICE


    def rearrange_factors(
        self,
        factors: dict=None,
        undo_concat_dictElements: bool=True,
        undo_concat_complexDim: bool=True,
    ):
        """
        Rearrange the factors of the TCA model.
        Undo the concatenation of the dictElements dimension.

        Args:
            factors (dict of np.ndarray):
                Dictionary of factors of the TCA model.
            undo_concat_dictElements (bool):
                Whether to undo the concatenation of the dictElements
                 dimension. Result will be a list of factors along the
                 dictElements concatenated dimension.
            undo_concat_complexDim (bool):
                Whether to undo the concatenation of the complexDim
                 dimension. Result will be complex valued arrays of
                 the same shape.
        """

        ## Set attributes
        self.factors = factors if factors is not None else self.factors

        ## Assert that if undo_concat_complexDim is True, then self._concat_complexDim is True
        assert not (undo_concat_complexDim and not self._concat_complexDim), f"FR ERROR: Cannot undo concatenation of complexDim dimension because it was not concatenated in the first place."
        ## Assert that if undo_concat_dictElements is True, then self._concat_dictElements is True
        assert not (undo_concat_dictElements and (self._method_handling_dictElements=='separate')), f"FR ERROR: Cannot undo concatenation of dictElements dimension because it was not concatenated in the first place."

        self.factors_rearranged = copy.deepcopy(self.factors)
        self.names_dims_array_postDecomp = copy.deepcopy(self.names_dims_array_preDecomp)

        if undo_concat_dictElements:
            assert self._method_handling_dictElements == 'concatenate', f"FR ERROR: Cannot undo concatenation of dictElements dimension because it was not concatenated in the first place."
            ## Undo the concatenation of the dictElements dimension
            idx_dictElementConcat = np.where([self._name_dim_dictElements in n for n in self.names_dims_array_postDecomp])[0][0]  ## Get the index of the dimension that was concatenated in the pre-decomposition arrays
            name_dim_dictElementConcat = self.names_dims_array_postDecomp[idx_dictElementConcat]  ## Get the name of the dimension that was concatenated in the pre-decomposition arrays
            idx_dictElementPreConcat = np.where([n in self._name_dim_concat_dictElements for n in self._names_dims_array])[0][0]  ## Get the index of the dimension that was concatenated in the original arrays
            ## Get the relative size of all the dictElements vs the size of the concatenated dictElements
            size_factor = self.factors_rearranged['0'][name_dim_dictElementConcat].shape[0] / sum([d[idx_dictElementPreConcat] for d in self.shapes_dictElements])
            lens_dictElements = [int(d[idx_dictElementPreConcat] * size_factor) for d in self.shapes_dictElements]  ## Get the lengths of the dictElements dimensions along the concatenated dimension in the original arrays
            lens_cumsum_dictElements = np.cumsum([0] + lens_dictElements)  ## Get the cumsum of the above
            print(f"Rearranging the dictElements dimension called: '{name_dim_dictElementConcat}' of shape {self.factors_rearranged['0'][name_dim_dictElementConcat].shape} into a list of chunks of lengths {lens_dictElements}.") if self._verbose > 1 else None 

            ## Fix names_dims_array
            name_dim_sansDictElement = self.names_dims_array_postDecomp[idx_dictElementConcat].replace(self._name_dim_dictElements, '')[1:-2]
            self.names_dims_array_postDecomp[idx_dictElementConcat] = name_dim_sansDictElement  ## Remove dictElement name from names_dims_array
            self.name_dim_dictElements_postDecomp = self._name_dim_dictElements  ## Replace name_dim_dictElements with the original name
            print(f"New names_dims_array in self.factors_rearranged: {self.names_dims_array_postDecomp}, new name_dim_dictElements: '{self.name_dim_dictElements_postDecomp}'") if self._verbose > 1 else None

            ### Prepare the new factor
            factor_postDecomp = {name_dim_sansDictElement + '_' + name: np.take(self.factors_rearranged['0'][name_dim_dictElementConcat], np.arange(lens_cumsum_dictElements[ii], lens_cumsum_dictElements[ii+1]), axis=0) for ii,name in enumerate(self._names_dictElements)}  ## Split the concatenated factor into a list of chunks

            ## Make a new dict of factors with the new factor
            self.factors_rearranged['0'][self.name_dim_dictElements_postDecomp] = factor_postDecomp  ## Replace the concatenated factor with the new factor
            del self.factors_rearranged['0'][name_dim_dictElementConcat]  ## Delete the concatenated factor
        else:
            self.name_dim_dictElements_postDecomp = self._name_dim_dictElements

        if undo_concat_complexDim:
            ## Undo the concatenation of the complexDim dimension
            def undo_concat_complexDim(factor: np.ndarray):
                """
                Undo the concatenation of the complexDim dimension.
                """
                len_factor = factor.shape[0]
                return factor[0:len_factor:2] + 1j*factor[1:len_factor:2]

            print(f"Rearranging the complexDim") if self._verbose > 1 else None
            for ii, (path, val) in enumerate(helpers.find_subDict_key(s='.*complex.*', d=self.factors_rearranged)):
                new_key = re.sub(r' complex\x29', '', path[-1])[1:]  ## Remove the 'complex)' from the key. \x29 is the ascii code for ')'. The [1:] is to remove the leading '(' in the name.
                helpers.deep_update_dict(
                    dictionary=self.factors_rearranged, 
                    key=path, 
                    new_val=undo_concat_complexDim(val),
                    new_key=new_key,
                    in_place=True,
                )

        ## Place factors into run_data
        self.run_data['factors'] = self.factors
        self.run_data['names_dims_array_preDecomp'] = self.names_dims_array_preDecomp
        self.run_data['name_dim_dictElements_preDecomp'] = self.name_dim_dictElements_preDecomp
        self.run_data['factors_rearranged'] = self.factors_rearranged
        self.run_data['names_dims_array'] = self.names_dims_array_postDecomp
        self.run_data['name_dim_dictElements'] = self.name_dim_dictElements_postDecomp
        ## Place info into run_info
        self.run_info['names_dims_array'] = self.names_dims_array_postDecomp
        self.run_info['name_dim_dictElements'] = self.name_dim_dictElements_postDecomp
        self.run_info['num_dictElements'] = self.num_dictElements

    def plot_factors(
        self, 
        factors: dict=None,
        figure_saver: util.Figure_Saver=None,
        show_figures: bool=True,
    ):
        """
        Plot the factors.

        Args:
            factors (dict): 
                The factors to plot. If None, then uses self.factors_rearranged
                 if not None, and self.factors as last resort.
            figure_saver (util.Figure_Saver):
                If None, then figures are not saved.
                Should be an instance of util.Figure_Saver.
            show_figures (bool):
                If True, then figures are shown.
        """
        import matplotlib.pyplot as plt
        ## Set attributes
        if factors is None:
            if self.factors_rearranged is None:
                if self.factors is None:
                    raise Exception("FR ERROR: No factors to plot.")
                factors = self.factors
                name_factors = 'factors'
            else:
                factors = self.factors_rearranged
                name_factors = 'factors_rearranged'
        else:
            name_factors = 'factors'

        ## Toggle interactive mode
        ### Check if interactive mode is on
        existing_interactive_mode = plt.isinteractive()
        ### Set interactive mode to on if show_figures is True
        plt.ion() if show_figures else plt.ioff()

        ## Plot the factors
        ftp = factors_to_plot = [d for d in helpers.find_subDict_key(s='.*', d=factors) if isinstance(d[1], dict)==False]
        for ii, (path, val) in enumerate(ftp):
            ## Get the name of the factor
            name_factor = path[-1]
            title_figure = f"{name_factors}_[{name_factor}]"
            ## Plot the factor
            fig, ax = plt.subplots()
            ax.plot(val)
            ax.set_title(title_figure)
            ax.set_xlabel(f'{name_factor}_bin')
            ax.legend(np.arange(val.shape[1]))

            ## Save the figure
            if figure_saver is not None:
                figure_saver.save(fig, title_figure)

        ## Toggle interactive mode back to original state
        plt.ion() if existing_interactive_mode else plt.ioff()
            

    def _cleanup(self):
        """
        Clear the CUDA cache and garbage collect.
        """
        if hasattr(self, '_DEVICE'):
            if 'cuda' in self._DEVICE.type:
                for ii in range(5):
                    torch.cuda.empty_cache()
                    time.sleep(0.1)
                    gc.collect()
                    time.sleep(0.1)
        else:
            [gc.collect() for ii in range(5)]


    # def _check_inputs(self, data):
    #     """
    #     Check the inputs for type and value.
    #     data is passed in because it is large and not set as
    #      an attribute until after this function is called.

    #     Args:
    #         data (dict of np.ndarray):
    #             Dictionary of data arrays.
    #             Each element of the dictionary should be a numpy
    #              array. Number of dimensions should be the same
    #              as the number of names in self.names_dims_array.
    #     """
    #     ## Assertions
    #     ### Assert that for each argument, the type matches the expected type
    #     ### Assert that the data is a dictionary, and that all the values are numpy arrays, and that all arrays have the same number of dimensions
    #     assert isinstance(data, dict), 'data must be a dictionary'
    #     assert all([isinstance(v, np.ndarray) for v in data.values()]), 'all values in data must be numpy arrays'
    #     assert len(set([v.ndim for v in data.values()])) == 1, 'all arrays in data must have the same number of dimensions'
    #     ### Assert that the names of the dimensions are unique strings, and that the number of names matches the number of dimensions in the data
    #     assert all([isinstance(n, str) for n in self.names_dimensions]), 'names of dimensions must be strings'
    #     assert len(self.names_dimensions) == len(set(self.names_dimensions)), 'names of dimensions must be unique'
    #     assert len(self.names_dimensions) == len(data[list(data.keys())[0]].shape), 'number of names of dimensions must match the number of dimensions in the data'
    #     ### Assert that the dimensions to concatenate are tuples of strings, and that the strings are in the list of names of dimensions
    #     assert all([isinstance(d, tuple) for d in self.dims_to_concatenate]), 'dimensions to concatenate must be tuples'
    #     assert all([all([isinstance(n, str) for n in d]) for d in self.dims_to_concatenate]), 'dimensions to concatenate must be tuples of strings'
    #     assert all([all([n in self.names_dimensions for n in d]) for d in self.dims_to_concatenate]), 'dimensions to concatenate must be tuples of strings that are in the list of names of dimensions'
    #     ### Assert that the method is a string
    #     assert isinstance(self.method, str), 'method must be a string'
    #     ### Assert that the parameters for the method are a dictionary
    #     assert isinstance(self.params_method, dict), 'parameters for method must be a dictionary'


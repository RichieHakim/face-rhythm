"""Tensor Component Analysis (TCA) wrapper around :mod:`tensorly`.

Provides the ``TCA`` class for decomposing multi-way arrays (time x points x
frequency x ...) produced upstream by the spectral pipeline. Handles dict-of-
arrays ingestion, axis concatenation, complex-to-real unfolding, normalization,
and the actual decomposition via tensorly's CP / NN-HALS / Randomized CP
solvers on either numpy or pytorch backends.
"""

from typing import Union
import time
from functools import partial
import gc
import copy
import re

import numpy as np
from tqdm.auto import tqdm
import torch
import tensorly as tl
import einops

from . import util
from . import helpers

## Define TCA class as a subclass of util.FR_Module
class TCA(util.FR_Module):
    """
    Performs Tensor Component Analysis (TCA) on multi-way arrays produced by
    the spectral pipeline using ``tensorly`` solvers. RH 2022

    Args:
        verbose (Union[bool, int]):
            Verbosity level. One of \n
            * ``0``: No messages.
            * ``1``: Warnings only.
            * ``2``: Info messages. \n
            (Default is ``1``)

    Attributes:
        config (dict):
            Configuration dictionary populated by ``__init__`` and updated by
            subsequent method calls.
        run_info (dict):
            Run-time information populated after fitting and rearranging.
        run_data (dict):
            Run-time data (factors and dimension names) populated after
            fitting and rearranging.
    """
    def __init__(
        self,
        verbose: Union[bool, int]=1,
    ):
        """
        Initializes the ``TCA`` object and prepares empty config/run state.
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
        Rearranges the input data dictionary into a single tensor (or set of
        tensors) suitable for TCA. Supports concatenating array dimensions,
        unfolding the complex dimension, combining or stacking dictionary
        elements, and windowing each array along a chosen dimension.

        Args:
            data (dict):
                Dictionary mapping element name to a ``numpy.ndarray`` of
                consistent rank. Arrays may be complex valued.
            names_dims_array (list):
                Names of the dimensions of the data arrays, in axis order.
                (Default is ``['xy', 'points', 'frequency', 'time']``)
            names_dims_concat_array (list):
                List of 2-element lists ``[dim_a, dim_b]`` describing pairs
                of array dimensions to concatenate. ``dim_a`` is folded into
                ``dim_b``, producing a single dimension named
                ``'(dim_a dim_b)'`` with length
                ``len(dim_a) * len(dim_b)``. Pairs are applied in the order
                given. (Default is ``[['xy', 'points']]``)
            concat_complexDim (bool):
                If ``True``, real and imaginary parts are stacked and folded
                into ``name_dim_concat_complexDim``. Requires complex valued
                input. (Default is ``True``)
            name_dim_concat_complexDim (str):
                Name of the array dimension into which the complex dimension
                is folded. Typically ``'time'``. (Default is ``'time'``)
            name_dim_dictElements (str):
                Semantic name for the dictionary elements (e.g. ``'trials'``
                or ``'videos'``). (Default is ``'trials'``)
            method_handling_dictElements (str):
                How to combine dictionary elements. One of \n
                * ``'concatenate'``: Concatenate elements along
                  ``name_dim_concat_dictElements``; output is a single
                  array of the same rank as inputs.
                * ``'stack'``: Stack elements along a new leading axis;
                  output is a single array with one extra dimension.
                * ``'separate'``: Keep each element as its own tensor;
                  decompositions run independently. \n
                (Default is ``'concatenate'``)
            name_dim_concat_dictElements (str):
                Array dimension along which to concatenate dictionary
                elements. Only used when ``method_handling_dictElements`` is
                ``'concatenate'``. (Default is ``'time'``)
            idx_windows (list):
                Per-element ``(start, end)`` index pairs (inclusive) defining
                a window along ``name_dim_array_window``. If ``None``, the
                full array is used. (Default is ``None``)
            name_dim_array_window (str):
                Array dimension along which ``idx_windows`` is applied. Only
                used when ``idx_windows`` is not ``None``.
                (Default is ``'time'``)
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
            Folds the array dimensions listed in
            ``self._names_dims_concat_array`` (and optionally the complex
            dimension) using ``einops.rearrange``.

            Args:
                data (np.ndarray):
                    Input array with axes ordered according to
                    ``self._names_dims_array``.

            Returns:
                (np.ndarray):
                    data_out (np.ndarray):
                        Rearranged array with concatenated dimensions.
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
            Selects a window from a single dictionary entry along the
            dimension named ``self._name_dim_array_window``.

            Args:
                data (np.ndarray):
                    Input array for one dictionary element.
                win_idx (int):
                    Index into ``self._idx_windows`` identifying which
                    window to apply.

            Returns:
                (np.ndarray):
                    data_out (np.ndarray):
                        Windowed array, or the original array if no window
                        is specified for this entry.
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
        Normalizes ``self.data`` along a named dimension by optional mean
        subtraction and/or standard-deviation scaling. Requires that
        ``self.rearrange_data`` has already populated ``self.data``.

        Args:
            mean_subtract (bool):
                If ``True``, subtracts the mean along ``dim_name``.
                (Default is ``False``)
            std_divide (bool):
                If ``True``, divides by the standard deviation along
                ``dim_name``. (Default is ``False``)
            dim_name (str):
                Name of the array dimension to normalize over. Must match a
                name in ``self.names_dims_array_preDecomp`` (i.e. the names
                that exist after ``rearrange_data``). (Default is ``'time'``)
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
        Fits a TCA model to the rearranged data using a ``tensorly``
        decomposition. Populates ``self.factors``, ``self.factors_raw``, and
        ``self.factor_weights``.

        Args:
            data (dict):
                Dictionary of ``numpy.ndarray`` data arrays of identical
                shape. If ``None``, ``self.data`` (set by
                ``rearrange_data``) is used. (Default is ``None``)
            method (str):
                ``tensorly`` decomposition class to instantiate. One of \n
                * ``'CP_NN_HALS'``: Non-negative CP decomposition via the
                  HALS algorithm.
                * ``'CP'``: Standard CP decomposition.
                * ``'RandomizedCP'``: Randomized CP decomposition for large
                  tensors.
                * ``'ConstrainedCP'``: Constrained CP decomposition. \n
                (Default is ``'CP_NN_HALS'``)
            params_method (dict):
                Keyword arguments forwarded to the ``tensorly`` decomposition
                class. See ``tensorly`` documentation for valid keys.
                (Default is the ``CP_NN_HALS`` parameter set defined in the
                signature)
            backend (str):
                ``tensorly`` backend. One of \n
                * ``'pytorch'``: Recommended for most use cases.
                * ``'numpy'``: NumPy backend. \n
                (Default is ``'pytorch'``)
            DEVICE (str):
                Torch device string (e.g. ``'cpu'`` or ``'cuda'``) used when
                ``backend`` is ``'pytorch'``. (Default is ``'cpu'``)
            verbose (Union[bool, int]):
                Verbosity level. ``0`` is silent, ``1`` warnings, ``2`` info.
                (Default is ``1``)
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
        cp_all = {key: self._model.fit_transform(torch.as_tensor(d, device=self._DEVICE)) for key, d in self.data.items()}
        self.factors_raw = cp_all
        self.factors = {key_factor: {key: cp.factors[ii].cpu().numpy() for ii, key in enumerate(self.names_dims_array_preDecomp)} for key_factor,cp in cp_all.items()}
        self.factor_weights = {key: cp.weights.cpu().numpy() for key, cp in cp_all.items()}

        ## Clean up
        self._cleanup()

        ## Place config into run_config
        self.config['method'] = method
        self.config['backend'] = backend
        self.config['params_method'] = params_method
        self.config['device'] = DEVICE


    def order_factors_by_EVR(self, data: dict=None, factors: dict=None, weights: dict=None, overwrite_factors: bool=True):
        """
        Reorders TCA factors by descending explained variance ratio (EVR) on
        each data tensor.

        Args:
            data (dict):
                Dictionary of ``numpy.ndarray`` data tensors. If ``None``,
                ``self.data`` is used. (Default is ``None``)
            factors (dict):
                Dictionary of factor sets keyed to match ``data``. If
                ``None``, ``self.factors`` is used. (Default is ``None``)
            weights (dict):
                Dictionary of CP weights keyed to match ``data``. If
                ``None``, ``self.factor_weights`` is used.
                (Default is ``None``)
            overwrite_factors (bool):
                If ``True``, writes the reordered results back to
                ``self.factors``, ``self.factor_weights``, and
                ``self.evrs_ordered``. (Default is ``True``)

        Returns:
            (tuple): tuple containing:
                orders (dict):
                    Per-key sort indices used to reorder factors.
                factors_ordered (dict):
                    Per-key dictionaries of factors reordered by EVR.
                weights_ordered (dict):
                    Per-key CP weight vectors reordered by EVR.
                evrs_ordered (dict):
                    Per-key explained variance ratios in descending order.
        """
        factors = factors if factors is not None else self.factors
        weights = weights if weights is not None else self.factor_weights
        data = data if data is not None else self.data

        assert [key_d == key_f for key_d, key_f in zip(data.keys(), factors.keys())], "Data keys and factors keys must match."

        outs = {key_d: helpers.order_cp_factors_by_EVR(
            tensor_dense=torch.as_tensor(d),
            cp_factors=[torch.as_tensor(f) for f in f.values()],
            cp_weights=torch.as_tensor(weights[key_d]),
            orthogonalizable_EVR=True,
        ) for (key_d, d), (key_f, f) in zip(data.items(), factors.items())}
        orders = {key: out[0] for key, out in outs.items()}
        evrs = {key: out[1] for key, out in outs.items()}

        factors_ordered = {key: {key_factor: factors[key][key_factor][:, orders[key]] for key_factor in factors[key].keys()} for key in factors.keys()}
        weights_ordered = {key: weights[key][orders[key]] for key in weights.keys()}
        evrs_ordered = {key: evrs[key] for key in evrs.keys()}
        
        if overwrite_factors:
            self.factors = factors_ordered
            self.factor_weights = weights_ordered
            self.evrs_ordered = evrs_ordered

        return orders, factors_ordered, weights_ordered, evrs_ordered


    def rearrange_factors(
        self,
        factors: dict=None,
        undo_concat_dictElements: bool=True,
        undo_concat_complexDim: bool=True,
    ):
        """
        Reverses the dimension folding applied in ``rearrange_data`` so the
        fitted factors can be interpreted in the original data space.
        Populates ``self.factors_rearranged``,
        ``self.names_dims_array_postDecomp``, and
        ``self.name_dim_dictElements_postDecomp``.

        Args:
            factors (dict):
                Dictionary of factors to rearrange. If ``None``,
                ``self.factors`` is used. (Default is ``None``)
            undo_concat_dictElements (bool):
                If ``True``, splits the concatenated dictionary-elements
                dimension back into per-element factors. Requires that
                ``method_handling_dictElements`` was ``'concatenate'``.
                (Default is ``True``)
            undo_concat_complexDim (bool):
                If ``True``, recombines real and imaginary halves of the
                folded complex dimension into complex-valued arrays.
                Requires that ``concat_complexDim`` was ``True``.
                (Default is ``True``)
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
                Recombines an interleaved real/imaginary factor back into a
                complex-valued factor by treating even rows as the real part
                and odd rows as the imaginary part.

                Args:
                    factor (np.ndarray):
                        Interleaved real/imaginary factor along axis 0.

                Returns:
                    (np.ndarray):
                        factor_complex (np.ndarray):
                            Complex-valued factor with half the original
                            length along axis 0.
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
        Plots each leaf factor as a normalized line plot, one figure per
        factor. Optionally writes figures to disk via a
        ``util.Figure_Saver``.

        Args:
            factors (dict):
                Nested dictionary of factors to plot. If ``None``,
                ``self.factors_rearranged`` is used if available, otherwise
                ``self.factors``. (Default is ``None``)
            figure_saver (util.Figure_Saver):
                Saver used to persist figures. If ``None``, figures are not
                saved. (Default is ``None``)
            show_figures (bool):
                If ``True``, enables interactive mode so figures are shown.
                (Default is ``True``)
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
            ax.plot(val / val.max(axis=0))
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
        Clears the CUDA cache (when a CUDA device is configured) and runs
        Python garbage collection to release lingering tensors between fits.
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


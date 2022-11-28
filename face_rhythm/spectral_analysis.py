from typing import Union
from pathlib import Path

import numpy as np
import torch
import hdfdict
from tqdm import tqdm

from .util import FR_Module
from . import helpers

class VQT_Analyzer(FR_Module):
    """
    A class for generating normalized spectrograms for point
     displacement traces. The spectrograms are generated using
     the Variable Q-Transform (VQT) algorithm.
    RH 2022
    """
    def __init__(
        self,
        params_VQT: dict={
            'Fs_sample': 90,
            'Q_lowF': 3,
            'Q_highF': 20,
            'F_min': 1,
            'F_max': 40,
            'n_freq_bins': 55,
            'win_size': 501,
            'plot_pref': False,
            'downsample_factor': 4,
            'DEVICE_compute': 'cpu',
            'DEVICE_return': 'cpu',
            'return_complex': False,
            'filters': None, 
            'plot_pref': False,
            'progressBar': True,
        },
        normalization_factor: float=0.99,
        spectrogram_exponent: float=1.0,
        verbose: int=1,
    ):
        super().__init__()

        ## Set attributes
        self._normalization_factor = float(normalization_factor)
        self._spectrogram_exponent = float(spectrogram_exponent)
        self._verbose = int(verbose)
        self.spectrograms = None

        ## Initalize VQT filters
        self.VQT = helpers.VQT(**params_VQT)

        ## For FR_Module compatibility
        self.config = {
            'params_VQT': params_VQT,
            'normalization_factor': normalization_factor,
            'spectrogram_exponent': spectrogram_exponent,
            'verbose': verbose,
        }
        self.run_info = {
            'VQT_args': self.VQT.args,
        }
        self.run_data = {
            'VQT': {key: self.VQT.__dict__[key] for key in ['filters', 'wins']},
            'frequencies': self.VQT.freqs,
        }
        ## Append the self.run_info data to self.run_data
        # self.run_data.update(self.run_info)
        # self.run_data['config'] = self.config

    def cleanup(self):
        """
        Delete the large data objects that are no longer needed.
        """
        import gc
        print(f"FR: Deleting all attributes")
        while len(self.__dict__.keys()) > 0:
            key = list(self.__dict__.keys())[0]
            del self.__dict__[key]
        gc.collect()
        gc.collect()


    def transform(self, points_tracked: dict, point_positions: np.ndarray):
        """
        Generate spectrograms from the tracked points.
        Reshape the inputs, run the VQT, and normalize the spectrograms.

        Args:
            points_tracked (dict):
                A dictionary of 3D numpy arrays of shape(n_frames, n_points, 2)
                 containing the tracked points.
            point_positions (np.ndarray):
                A 2D numpy array of shape(n_points, 2) containing the positions
                 of the tracked points.
            name_pointsTracked (str):
                A string to use to index the points_tracked dict.
            idx_point (list):
                Index of point to transform and return or plot.
            plot (bool):
                Whether to plot the spectrograms.

        Returns:
            spectrograms (dict):
                A list of spectrograms for each element in idx_points.
        """
        ## Check inputs
        self._check_inputs(points_tracked, point_positions)

        ## Prepare traces
        print(f"Preparing traces. Reshaping and subtracting offsets...") if self._verbose > 1 else None
        self.point_positions = self._prepare_pointPositions(point_positions)
        
        ## Compute spectrograms
        self.spectrograms ={key: self._normalize_spectrogram(self.VQT(self._prepare_displacements(points, self.point_positions))).cpu().numpy() for key, points in tqdm(points_tracked.items(), disable=not self._verbose > 1, desc="Computing spectrograms", leave=False)}

        ## Update self.run_data
        self.run_data.update({
            'spectrograms': self.spectrograms,
            'point_positions': self.point_positions,
        })
        
    def _check_inputs(self, points_tracked: dict, point_positions: np.ndarray):
        """
        Check the inputs to the transform function.

        Args:
            points_tracked (dict):
                A dictionary of 3D numpy arrays of shape(n_frames, n_points, 2)
                 containing the tracked points.
            point_positions (np.ndarray):
                A 2D numpy array of shape(n_points, 2) containing the positions
                 of the tracked points.
        """
        ## Assertions
        ### Assert that the points_tracked dict is valid
        assert isinstance(points_tracked, (dict, hdfdict.hdfdict.LazyHdfDict)), f"points_tracked must be a dict or hdfdict.hdfdict.LazyHdfDict, not {type(points_tracked)}. See docstring for details."
        ### Assert that points_tracked contains 3D numpy arrays of shape(n_frames, n_points, 2)
        for key, value in points_tracked.items():
            assert isinstance(value, np.ndarray), f"points_tracked must contain numpy arrays, not {type(value)}. See docstring for details."
            assert value.ndim == 3, f"points_tracked must contain 3D numpy arrays, not {value.ndim}D. Shape should be (n_frames, n_points, 2). See docstring for details."
            assert value.shape[2] == 2, f"points_tracked must contain 3D numpy arrays of shape(n_frames, n_points, 2), not {value.shape}. See docstring for details."
        ### Assert that point_positions is a 2D numpy array of shape(n_points, 2)
        assert isinstance(point_positions, np.ndarray), f"point_positions must be a numpy array, not {type(point_positions)}. See docstring for details."
        assert point_positions.ndim == 2, f"point_positions must be a 2D numpy array, not {point_positions.ndim}D. Shape should be (n_points, 2). See docstring for details."
        assert point_positions.shape[1] == 2, f"point_positions must be a 2D numpy array of shape(n_points, 2), not {point_positions.shape}. See docstring for details."

    ## Prepare normalization function
    def _normalize_spectrogram(self, spectrogram):
        """
        Normalize spectrogram by dividing by the total power
            across all frequencies at each time point.

        Args:
            spectrogram (torch.Tensor):
                A spectrogram of shape: (n_points, n_freq_bins, n_frames)                    
        """
        s_exp = spectrogram ** self._spectrogram_exponent
        s_mean = torch.mean(torch.sum(s_exp , dim=1) , dim=0)  ## Mean of the summed power across all frequencies and points. Shape (n_frames,)
        return  s_exp / ((self._normalization_factor * s_mean[None,None,:]) + (1-self._normalization_factor))  ## Normalize the spectrogram by the mean power across all frequencies and points. Shape (n_points, n_freq_bins, n_frames)

    def _prepare_displacements(self, traces, point_positions):
        """
        A function to reshape and subtract offsets from traces.
        Reshape from (n_frames, n_points, 2) to (n_points * 2, n_frames)
        """
        return torch.as_tensor(traces.reshape((traces.shape[0], -1), order='F').T, dtype=torch.float32) - point_positions[:,None]

    def _prepare_pointPositions(self, point_positions):
        """
        A function to reshape point_positions from (n_points, 2) to (n_points * 2)
        """
        return torch.as_tensor(point_positions.reshape((-1,), order='F').T, dtype=torch.float32)


    def test_transform(self, points_tracked: dict, point_positions: np.ndarray, idx_point: list=[0,], name_points: str='0', plot: bool=True):
        """
        Do a test transform to check that the spectrograms are being
         computed correctly and look good.
        Use the plot to check it out visually.

        Args:
            points_tracked (dict):
                A dictionary of 3D numpy arrays of shape(n_frames, n_points, 2)
                 containing the tracked points.
            point_positions (np.ndarray):
                A 2D numpy array of shape(n_points, 2) containing the positions
                 of the tracked points.
            name_pointsTracked (str):
                A string to use to index the points_tracked dict.
            idx_point (list):
                Index of point to transform and return or plot.
            plot (bool):
                Whether to plot the spectrograms.

        Returns:
            spectrograms (dict):
                A list of spectrograms for each element in idx_points.
        """
        ## Check inputs
        self._check_inputs(points_tracked, point_positions)

        ## Prepare traces
        point_positions = self._prepare_pointPositions(point_positions[idx_point,:])

        ## Compute spectrogram
        spectrogram = self._normalize_spectrogram(self.VQT(self._prepare_displacements(points_tracked[name_points][:,idx_point,:], point_positions))).cpu().numpy()

        ## Plot
        if plot:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 1, figsize=(10, 5))
            ## set the y axis to the frequency bins
            axs[0].imshow(spectrogram[0,:,:], aspect='auto', origin='lower', cmap='hot', extent=[0, spectrogram.shape[2], self.VQT.freqs[0], self.VQT.freqs[-1]])
            axs[1].imshow(spectrogram[1,:,:], aspect='auto', origin='lower', cmap='hot', extent=[0, spectrogram.shape[2], self.VQT.freqs[0], self.VQT.freqs[-1]])
            axs[0].set_title(f'Spectrogram of x and y displacements of point {idx_point}')
            axs[1].set_xlabel('Time')
            axs[0].set_ylabel('Frequency')
            plt.show()

        return spectrogram

    
    def __repr__(self): return f"{self.__class__.__name__}(VQT={self.VQT.args}, spectrogram_exponent={self._spectrogram_exponent}, normalization_factor={self._normalization_factor})"
    def __getitem__(self, index): return self.spectrograms(index)
    def __len__(self): return len(self.spectrograms)
    def __iter__(self): return iter(self.spectrograms.items())    
    def __call__(self, points_tracked: dict, point_positions: np.ndarray, name_points: str='0'): 
        """
        Compute spectrograms for each point in points_tracked.
        Wrapper for self.transform()
        See self.transform() for details.
        """
        return self.transform(points_tracked, point_positions, name_points)
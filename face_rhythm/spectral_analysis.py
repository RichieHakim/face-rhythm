"""Spectral analysis of point-tracked motion via Variable-Q Transform (VQT).

``VQT_Analyzer`` converts point trajectories (from :mod:`face_rhythm.point_tracking`)
into per-point spectrograms with a Variable-Q transform, applies 1/f and
per-timepoint normalization, and writes the resulting complex or magnitude
tensors out for downstream :mod:`face_rhythm.decomposition`.
"""

from typing import Union
from pathlib import Path
import math

import numpy as np
import torch
from tqdm.auto import tqdm

from .util import FR_Module
from . import helpers

class VQT_Analyzer(FR_Module):
    """
    Computes normalized Variable-Q Transform (VQT) spectrograms for point
    displacement traces. RH 2022

    Args:
        params_VQT (dict):
            Keyword arguments forwarded to ``vqt.VQT`` (the Variable Q-Transform
            implementation in :mod:`vqt`). Notable keys include ``Fs_sample``
            (sampling rate in Hz), ``Q_lowF`` and ``Q_highF`` (Q-factors at the
            low and high frequency bounds), ``F_min`` and ``F_max`` (frequency
            range in Hz), ``n_freq_bins``, ``window_type``, ``downsample_factor``,
            and ``take_abs``. (Default is the dict shown in the signature)
        batch_size (int):
            Number of points processed per VQT batch. (Default is ``10``)
        device (str):
            Torch device on which the VQT model is run (e.g. ``'cpu'`` or
            ``'cuda'``). (Default is ``'cpu'``)
        normalization_factor (float):
            Strength of the per-timepoint power normalization, in the range
            ``[0, 1]``. ``0`` disables normalization; ``1`` forces every time
            point to have equal total power. (Default is ``0.99``)
        spectrogram_exponent (float):
            Exponent applied to the spectrogram magnitudes prior to
            normalization. (Default is ``1.0``)
        one_over_f_exponent (float):
            Exponent for the 1/f correction; the spectrogram is multiplied by
            ``freqs ** one_over_f_exponent``. ``0`` disables the correction.
            (Default is ``1.0``)
        verbose (int):
            Verbosity level. ``0`` is silent, higher values print and show
            progress bars. (Default is ``1``)

    Attributes:
        spectrograms (dict):
            Dict mapping each input key to its normalized spectrogram array.
            Populated by :meth:`transform_all`.
        x_axis (dict):
            Dict mapping each input key to the time axis (in samples) of its
            spectrogram. Populated by :meth:`transform_all`.
        freqs (dict):
            Dict mapping each input key to the frequency bin centers (in Hz).
            Populated by :meth:`transform_all`.
        point_positions (torch.Tensor):
            Reshaped reference positions used to subtract offsets from traces.
        vqt_model (vqt.VQT):
            The underlying VQT filter-bank model.
        config (dict):
            Constructor arguments echoed for ``FR_Module`` serialization.
        run_data (dict):
            Output payload (filters, frequencies, spectrograms, axes) used by
            ``FR_Module`` for export.
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
            'window_type': 'hann',
            'symmetry': 'center',
            'taper_asymmetric': True,
            'downsample_factor': 8,
            'padding': 'valid',
            'fft_conv': True,
            'fast_length': True,
            'take_abs': True,
            'filters': None,
            'plot_pref': False,
        },
        batch_size: int=10,
        device='cpu',
        normalization_factor: float=0.99,
        spectrogram_exponent: float=1.0,
        one_over_f_exponent: float=1.0,
        verbose: int=1,
    ):
        """Initializes the analyzer, builds the VQT filter bank, and stores
        config metadata for ``FR_Module`` serialization."""
        super().__init__()

        ## Set attributes
        self._params_VQT = params_VQT
        self._batch_size = int(batch_size)
        self._device = device
        self._normalization_factor = float(normalization_factor)
        self._spectrogram_exponent = float(spectrogram_exponent)
        self._one_over_f_exponent = float(one_over_f_exponent)
        self._verbose = int(verbose)

        self.spectrograms = None
        self.x_axis = None
        self.freqs = None

        self._demo_spectrogram = None

        ## Initalize VQT filters
        import vqt
        self.vqt_model = vqt.VQT(**params_VQT)
        self.vqt_model.cpu()

        ## For FR_Module compatibility
        self.config = {
            'params_VQT': params_VQT,
            'batch_size': batch_size,
            'device': device,
            'normalization_factor': normalization_factor,
            'spectrogram_exponent': spectrogram_exponent,
            'one_over_f_exponent': one_over_f_exponent,
            'verbose': verbose,
        }
        self.run_info = {
        }
        self.run_data = {
            'VQT': {key: getattr(self.vqt_model, key).cpu().detach().numpy() for key in ['filters', 'wins']},
            'frequencies': self.vqt_model.freqs,
        }
        ## Append the self.run_info data to self.run_data
        # self.run_data.update(self.run_info)
        # self.run_data['config'] = self.config

    def cleanup(self):
        """
        Deletes every attribute on the instance and triggers garbage collection
        to free large tensors held by the analyzer.
        """
        import gc
        print(f"FR: Deleting all attributes")
        while len(self.__dict__.keys()) > 0:
            key = list(self.__dict__.keys())[0]
            del self.__dict__[key]
        gc.collect()
        gc.collect()


    def transform(self, points_tracked: np.ndarray, point_positions: np.ndarray):
        """
        Transforms a single batch of tracked points into a normalized VQT
        spectrogram.

        Args:
            points_tracked (np.ndarray):
                Tracked point coordinates. shape: *(n_frames, n_points, 2)*.
            point_positions (np.ndarray):
                Reference positions of the tracked points used to compute
                displacements. shape: *(n_points, 2)*.

        Returns:
            (tuple): tuple containing:
                spectrograms (np.ndarray):
                    Normalized spectrograms for the x and y displacement
                    components. shape: *(2, n_points, n_freq_bins, n_frames_ds)*,
                    where ``n_frames_ds`` is the downsampled frame count.
                x_axis (np.ndarray):
                    Time axis of the spectrogram, in samples at the original
                    ``Fs_sample`` rate. shape: *(n_frames_ds,)*.
                freqs (np.ndarray):
                    Frequency bin centers in Hz. shape: *(n_freq_bins,)*.
        """
        ## Prepare traces
        point_positions = self._prepare_pointPositions(point_positions)
        points_tracked = self._prepare_displacements(points_tracked, point_positions)
        ## Compute spectrograms
        freqs = self.vqt_model.freqs
        xAxis = self.vqt_model.get_xAxis(points_tracked.shape[-1])
        ### send vqt_model to device
        self.vqt_model.to(self._device)
        spec = torch.cat([
            self.vqt_model(p.to(self._device)).cpu()
            for p in tqdm(helpers.make_batches(points_tracked, batch_size=self._batch_size), disable=not self._verbose > 1, desc='Computing spectrograms', leave=True, position=0, total=int(math.ceil(points_tracked.shape[0] / self._batch_size)), mininterval=1.0)
        ], dim=0)
        self.vqt_model.to('cpu')
        ## Reshape and normalize spectrograms
        spec_rs = self._normalize_spectrogram(spec)
        return spec_rs.cpu().numpy(), xAxis.cpu().numpy(), freqs.cpu().numpy()


    def transform_all(self, points_tracked: dict, point_positions: np.ndarray):
        """
        Generates spectrograms for every entry in a dict of tracked-point
        arrays and stores the results on the instance.

        Args:
            points_tracked (dict):
                Mapping from a name to a tracked-points array of shape
                *(n_frames, n_points, 2)*.
            point_positions (np.ndarray):
                Reference positions of the tracked points used to compute
                displacements. shape: *(n_points, 2)*.

        Set Attributes:
            spectrograms (dict):
                Dict mapping each input key to its normalized spectrogram.
            x_axis (dict):
                Dict mapping each input key to the spectrogram time axis.
            freqs (dict):
                Dict mapping each input key to the frequency bin centers in Hz.
            run_data (dict):
                Updated with ``spectrograms``, ``x_axis``, and
                ``point_positions`` for ``FR_Module`` export.
        """
        ## Check inputs
        self._check_inputs(points_tracked, point_positions)

        ## Prepare traces
        print(f"Preparing traces. Reshaping and subtracting offsets...") if self._verbose > 1 else None
        self.point_positions = self._prepare_pointPositions(point_positions)
        ## Compute spectrograms
        print(f"Computing spectrograms...") if self._verbose > 1 else None
        self.spectrograms, self.x_axis, self.freqs = {}, {}, {}
        for key, points in tqdm(points_tracked.items(), disable=not self._verbose > 1, desc='Computing spectrograms', leave=True, position=1):
            self.spectrograms[key], self.x_axis[key], self.freqs[key] = self.transform(points, point_positions)

        ## Update self.run_data
        self.run_data.update({
            'spectrograms': self.spectrograms,
            'x_axis': self.x_axis,
            'point_positions': self.point_positions,
        })

    def _check_inputs(self, points_tracked: dict, point_positions: np.ndarray):
        """
        Validates the structure, dtype, and shape of the inputs to
        :meth:`transform_all` and :meth:`demo_transform`.

        Args:
            points_tracked (dict):
                Mapping from a name to a tracked-points array of shape
                *(n_frames, n_points, 2)*.
            point_positions (np.ndarray):
                Reference positions of the tracked points. shape:
                *(n_points, 2)*.
        """
        ## Assertions
        ### Assert that the points_tracked dict is valid
        assert isinstance(points_tracked, (dict,)), f"points_tracked must be a dict, not {type(points_tracked)}. See docstring for details."
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
        Normalizes a spectrogram by dividing each time point by the mean total
        power across frequencies and points, then applies a 1/f correction.

        Args:
            spectrogram (torch.Tensor):
                Raw VQT output. shape: *(n_points * 2, n_freq_bins, n_frames)*.
                May be real (when ``take_abs=True``) or complex.

        Returns:
            (torch.Tensor):
                spectrogram (torch.Tensor):
                    Normalized spectrogram split into x/y components. shape:
                    *(2, n_points, n_freq_bins, n_frames)*. Same dtype family as
                    the input (real or complex).
        """
        ## Check inputs
        s = spectrogram

        if torch.is_complex(s) == False:
            s_exp = s ** self._spectrogram_exponent
            s_mean = torch.mean(torch.sum(s_exp , dim=1) , dim=0)  ## Mean of the summed power across all frequencies and points. Shape (n_frames,)
            s_norm =  s_exp / ((self._normalization_factor * s_mean[None,None,:]) + (1-self._normalization_factor))  ## Normalize the spectrogram by the mean power across all frequencies and points. Shape (n_points, n_freq_bins, n_frames)

        elif torch.is_complex(s) == True:
            s_mag = torch.abs(s)
            s_phase = torch.angle(s)
            s_exp = s_mag ** self._spectrogram_exponent
            s_mean = torch.mean(torch.sum(s_exp , dim=1) , dim=0)  ## Mean of the summed power across all frequencies and points. Shape (n_frames,)
            s_mag_norm = s_exp / ((self._normalization_factor * s_mean[None,None,:]) + (1-self._normalization_factor))  ## Normalize the spectrogram by the mean power across all frequencies and points. Shape (n_points, n_freq_bins, n_frames)
            s_norm = torch.polar(s_mag_norm, s_phase)

        ## Do 1/f correction
        s_norm = s_norm * torch.as_tensor(self.vqt_model.freqs[None,:,None] ** self._one_over_f_exponent, dtype=torch.float32) if self._one_over_f_exponent != 0 else s_norm
        s_norm_rs = s_norm.reshape(2, int(s_norm.shape[0]/2), s_norm.shape[1], s_norm.shape[2])
        return s_norm_rs

    def _prepare_displacements(self, traces, point_positions):
        """
        Reshapes tracked-point traces from *(n_frames, n_points, 2)* to
        *(n_points * 2, n_frames)* (Fortran order) and subtracts the reference
        positions to yield displacement traces.

        Args:
            traces (np.ndarray):
                Tracked point coordinates. shape: *(n_frames, n_points, 2)*.
            point_positions (torch.Tensor):
                Flattened reference positions. shape: *(n_points * 2,)*.

        Returns:
            (torch.Tensor):
                displacements (torch.Tensor):
                    Per-component displacement traces. shape:
                    *(n_points * 2, n_frames)*, dtype: *float32*.
        """
        out = torch.as_tensor(traces.reshape((traces.shape[0], -1), order='F').T, dtype=torch.float32) - point_positions[:, None]
        return out

    def _prepare_pointPositions(self, point_positions):
        """
        Flattens reference point positions from *(n_points, 2)* to
        *(n_points * 2,)* using Fortran order.

        Args:
            point_positions (np.ndarray):
                Reference positions. shape: *(n_points, 2)*.

        Returns:
            (torch.Tensor):
                point_positions (torch.Tensor):
                    Flattened reference positions. shape: *(n_points * 2,)*,
                    dtype: *float32*.
        """
        return torch.as_tensor(point_positions.reshape((-1,), order='F').T, dtype=torch.float32)


    def demo_transform(
        self,
        points_tracked: dict,
        point_positions: np.ndarray,
        idx_point: list=[0,],
        name_points: str='0',
        plot: bool=True,
    ):
        """
        Runs a single-point demo transform for visual sanity checking and
        prints the projected memory footprint of the full spectrogram set.

        Args:
            points_tracked (dict):
                Mapping from a name to a tracked-points array of shape
                *(n_frames, n_points, 2)*.
            point_positions (np.ndarray):
                Reference positions of the tracked points. shape:
                *(n_points, 2)*.
            idx_point (list):
                Indices of points within the selected entry to transform and
                plot. (Default is ``[0,]``)
            name_points (str):
                Key into ``points_tracked`` selecting which array to use.
                (Default is ``'0'``)
            plot (bool):
                If ``True``, displays a matplotlib figure with the x and y
                spectrograms. (Default is ``True``)

        Returns:
            (tuple): tuple containing:
                spec (np.ndarray):
                    Demo spectrogram. shape:
                    *(2, n_freq_bins, n_frames_ds)*.
                x_axis (np.ndarray):
                    Time axis of the spectrogram, in samples at the original
                    ``Fs_sample`` rate. shape: *(n_frames_ds,)*.
                freqs (np.ndarray):
                    Frequency bin centers in Hz. shape: *(n_freq_bins,)*.
        """
        ## Check inputs
        self._check_inputs(points_tracked, point_positions)

        ## Transform
        spec, x_axis, freqs = self.transform(points_tracked[name_points][:,idx_point,:][:,None,:], point_positions[idx_point,:][None,...])
        spec = spec.squeeze(1)
        # spec, x_axis, freqs = out['spec'][:,0,:,:], out['x_axis'], out['freqs']
        print(f"Demo spectrogram shape: {spec.shape}")

        ## Plot
        if plot:
            import matplotlib.pyplot as plt
            x_0 = x_axis[0] / self.vqt_model.Fs_sample
            x_N = x_axis[-1] / self.vqt_model.Fs_sample
            fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
            axs[0].imshow(np.abs(spec[0,:,:]), aspect='auto', origin='lower', cmap='hot')
            axs[1].imshow(np.abs(spec[1,:,:]), aspect='auto', origin='lower', cmap='hot')
            axs[0].set_title(f'Spectrogram of x and y displacements of point {idx_point}')
            axs[1].set_xlabel('Time (s)')
            axs[0].set_ylabel('Frequency')
            ## set yticks values
            yticks = np.linspace(0, len(freqs), num=8, endpoint=False, dtype=int)
            yticklabels = np.round(freqs[yticks], 2)
            axs[0].set_yticks(yticks)
            axs[0].set_yticklabels(yticklabels)
            axs[1].set_yticks(yticks)
            axs[1].set_yticklabels(yticklabels)

            plt.show()


        ## Compute size of all output spectrograms
        ### Get the output shape for each spectrogram
        def _get_avgpool1d_downsample_length(length, kernel_size, stride, padding):
            """Returns the output length of a 1D average pool with the given
            kernel, stride, and padding."""
            return int((length + 2 * padding - kernel_size) / stride + 1)
        shapes_pt = [p.shape for p in points_tracked.values()]
        shapes_spec = [(s[2], len(self.vqt_model.freqs), s[1], _get_avgpool1d_downsample_length(s[0], kernel_size=int(self.vqt_model.downsample_factor), stride=self.vqt_model.downsample_factor, padding=0)) for s in shapes_pt]
        sizes_spec = [np.prod(s) * spec.itemsize / 1e9 for s in shapes_spec]
        print(f'Total size of all spectrograms: {np.sum(sizes_spec):.8f} GB') if self._verbose > 1 else None
        print(f'Individual spectrogram sizes (in GB): {sizes_spec}') if self._verbose > 1 else None

        return spec, x_axis, freqs


    def __repr__(self):
        """Returns a concise string representation of the analyzer."""
        return f"{self.__class__.__name__}( normalization_factor={self._normalization_factor}, spectrogram_exponent={self._spectrogram_exponent}, VQT={self.vqt_model}, verbose={self._verbose} )"
    def __getitem__(self, index):
        """Indexes into ``self.spectrograms`` by key."""
        return self.spectrograms(index)
    def __len__(self):
        """Returns the number of stored spectrogram entries."""
        return len(self.spectrograms)
    def __iter__(self):
        """Iterates over ``(key, spectrogram)`` pairs in ``self.spectrograms``."""
        return iter(self.spectrograms.items())
    def __call__(self, points_tracked: dict, point_positions: np.ndarray, name_points: str='0'):
        """
        Computes spectrograms for one entry of ``points_tracked``. Thin wrapper
        around :meth:`transform`; see that method for details.

        Args:
            points_tracked (dict):
                Mapping from a name to a tracked-points array of shape
                *(n_frames, n_points, 2)*.
            point_positions (np.ndarray):
                Reference positions of the tracked points. shape:
                *(n_points, 2)*.
            name_points (str):
                Key into ``points_tracked`` selecting which array to transform.
                (Default is ``'0'``)
        """
        return self.transform(points_tracked[name_points], point_positions)

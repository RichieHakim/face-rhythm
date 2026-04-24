"""Multi-session (cross-session) image alignment.

Ported from ROICaT (https://github.com/RichieHakim/ROICaT,
``roicat/tracking/alignment.py`` and ``roicat/helpers.py``).
Both projects (C) Rich Hakim — released under the face-rhythm LICENSE
alongside the rest of the package. The ROICaT source is GPL-3.0-only;
because Rich is the sole author of both packages, there is no license
conflict; this module re-licenses the ported portions under face-rhythm's
terms for face-rhythm users.

This module provides :class:`Aligner`, which registers a list of FOV images
to a template using one of several geometric-registration backends. The
public API and call-shape intentionally match ROICaT's
``tracking.alignment.Aligner`` so notebooks and scripts that used the
ROICaT entry-point can swap ``roicat.tracking.alignment`` for
``face_rhythm.alignment_multisession`` unchanged.

Backends ported:
    - ``'RoMa'`` (optional — requires ``pip install face-rhythm[multisession]``)
    - ``'ECC_cv2'`` (OpenCV-only, always available)
    - ``'PhaseCorrelation'`` (torch-FFT only, always available)
    - ``'NullRegistration'`` (identity, always available)

Backends deliberately NOT ported (pull heavy deps that face-rhythm users
don't need): LoFTR, DISK_LightGlue, DeepFlow, OpticalFlowFarneback, SIFT,
ORB. If you need them, install and use ROICaT directly.

Low-level helpers (warp_matrix_to_remappingIdx, remap_images,
compose_transform_matrices, cv2RemappingIdx_to_pytorchFlowField,
find_geometric_transformation, make_batches, hash_file) are imported from
:mod:`face_rhythm.helpers`, which already carries their ROICaT-ported
equivalents. Only the helpers unique to alignment (ImageAlignmentChecker,
phase_correlation, 2-D Butterworth bandpass filter construction, Dijkstra
path reconstruction) are re-implemented here.
"""

from typing import Any, Dict, List, Optional, Tuple, Union  ## typing

import functools
import warnings
from pathlib import Path  ## built-ins

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.sparse
import scipy.sparse.csgraph
import skimage  ## core face-rhythm dep (scikit_image). Used for euclidean RANSAC.
import skimage.measure
import skimage.transform
import torch
from tqdm.auto import tqdm  ## third-party

from face_rhythm import helpers  ## local  (warp_matrix_to_remappingIdx, remap_images, compose_transform_matrices, find_geometric_transformation, make_batches, hash_file)


## ---------------------------------------------------------------------------
## Minimal base-class stub (replaces roicat.util.ROICaT_Module for the port).
## Only the two features the Aligner actually touches are implemented:
##   * ``self.params`` dict for storing per-call kwargs
##   * ``self._locals_to_params(locals_dict, keys)`` helper
## ---------------------------------------------------------------------------

class _AlignerModuleStub:
    """Tiny base class providing the param-tracking hooks the ported Aligner
    expects. Replaces ``roicat.util.ROICaT_Module`` without pulling in the
    serialization machinery or ``system_info()`` side-effects.
    """

    def __init__(self):
        self.params: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _locals_to_params(
        locals_dict: Dict[str, Any],
        keys: List[str],
    ) -> Dict[str, Any]:
        """Extract a subset of keys from a ``locals()`` dict.

        Args:
            locals_dict (Dict[str, Any]): The dict returned by ``locals()``.
            keys (List[str]): Keys to extract.

        Returns:
            (Dict[str, Any]): Sub-dictionary containing only the requested keys.
        """
        out = {}
        for key in keys:
            if key in locals_dict:
                out[key] = locals_dict[key]
            else:
                warnings.warn(f"key={key} not found in locals_dict. Skipping.")
        return out


## ---------------------------------------------------------------------------
## Helpers unique to this port (not already in face_rhythm.helpers).
## ---------------------------------------------------------------------------

def make_distance_grid(
    shape: Tuple[int, int] = (512, 512),
    p: int = 2,
    idx_center: Optional[Tuple[int, int]] = None,
    use_fftshift_center: bool = False,
) -> np.ndarray:
    """Create an (H, W) array of Minkowski-p distances to a reference index.

    Ported from ROICaT helpers.make_distance_grid.

    Args:
        shape (Tuple[int, int]):
            Grid shape (H, W).
        p (int):
            Minkowski order (``1``: Manhattan, ``2``: Euclidean, ``inf``:
            Chebyshev). Values above 2 approximate the max-norm.
        idx_center (Optional[Tuple[int, int]]):
            Center index for the distances. If ``None``, uses the geometric
            middle of the array (between two pixels on even shapes).
        use_fftshift_center (bool):
            If ``True``, uses the index where ``np.fft.fftshift(np.fft.fftfreq(N))``
            is zero as the center (correct reference for fftshifted 2-D FFTs).

    Returns:
        (np.ndarray):
            grid_dist (np.ndarray): Array of shape ``shape`` with Minkowski-p
            distances to the center.
    """
    if use_fftshift_center:
        freqs_h = np.fft.fftshift(np.fft.fftfreq(shape[0]))
        freqs_w = np.fft.fftshift(np.fft.fftfreq(shape[1]))
        idx_center = (int(np.argmin(np.abs(freqs_h))), int(np.argmin(np.abs(freqs_w))))

    shape_arr = np.array(shape)
    if idx_center is not None:
        axes = [np.linspace(-idx_center[i], shape_arr[i] - idx_center[i] - 1, shape_arr[i]) for i in range(len(shape_arr))]
    else:
        axes = [np.arange(-(d - 1) / 2, (d - 1) / 2 + 0.5) for d in shape_arr]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=0)
    grid_dist = np.linalg.norm(grid, ord=p, axis=0)
    return grid_dist


def design_butter_bandpass(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth bandpass filter (with low/highpass edge cases).

    Ported from ROICaT helpers.design_butter_bandpass.

    Args:
        lowcut (float): Low-cutoff frequency. If ``<= 0``, a lowpass is used.
        highcut (float): High-cutoff frequency. If ``>= fs/2``, a highpass is used.
        fs (float): Sample rate.
        order (int): Butterworth filter order.

    Returns:
        (Tuple[np.ndarray, np.ndarray]):
            b, a — Numerator and denominator polynomials of the IIR filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low <= 0:
        b, a = scipy.signal.butter(N=order, Wn=high, btype='low')
    elif high >= 1:
        b, a = scipy.signal.butter(N=order, Wn=low, btype='high')
    else:
        b, a = scipy.signal.butter(N=order, Wn=[low, high], btype='band')
    return b, a


def make_2D_frequency_filter(
    hw: Tuple[int, int],
    low: float = 5.0,
    high: float = 6.0,
    order: int = 3,
    distance_p: int = 100,
) -> np.ndarray:
    """Build a 2-D fftshifted bandpass mask for phase-correlation scoring.

    Ported from ROICaT helpers.make_2D_frequency_filter. The filter is the
    1-D Butterworth magnitude response (from :func:`design_butter_bandpass`)
    evaluated on a Minkowski-``distance_p`` distance grid
    (:func:`make_distance_grid`).

    Args:
        hw (Tuple[int, int]): Output height and width.
        low (float): Low cutoff in pixel units.
        high (float): High cutoff in pixel units.
        order (int): Butterworth order.
        distance_p (int): Minkowski norm for the distance grid (``100`` ~ Chebyshev).

    Returns:
        (np.ndarray): 2-D filter of shape ``hw`` with values in ``[0, 1]``.
    """
    ## Distance grid starting from the fftshifted center.
    grid = make_distance_grid(shape=hw, p=distance_p, use_fftshift_center=True)

    ## Oversample the 1-D kernel so the interpolation is smooth.
    n_x = max(hw) * 10
    fs = max(hw) * 1
    low = max(0, low)
    high = min((max(hw) / 2) - 1, high)
    b, a = design_butter_bandpass(lowcut=low, highcut=high, fs=fs, order=order)
    w, h = scipy.signal.freqz(b, a, worN=n_x)
    x_kernel = (fs * 0.5 / np.pi) * w
    kernel = np.abs(h)

    filt = np.interp(x=grid, xp=x_kernel, fp=kernel)
    return filt


def phase_correlation(
    im_template: Union[np.ndarray, torch.Tensor],
    im_moving: Union[np.ndarray, torch.Tensor],
    mask_fft: Optional[Union[np.ndarray, torch.Tensor]] = None,
    return_filtered_images: bool = False,
    eps: float = 1e-8,
) -> Union[np.ndarray, torch.Tensor, Tuple]:
    """Phase-correlation of two images along the last two axes.

    Ported from ROICaT helpers.phase_correlation.

    Args:
        im_template (Union[np.ndarray, torch.Tensor]):
            Template image(s). Shape ``(..., H, W)``; leading dims broadcast.
        im_moving (Union[np.ndarray, torch.Tensor]):
            Moving image(s). Shape ``(..., H, W)``; broadcasts against template.
        mask_fft (Optional[Union[np.ndarray, torch.Tensor]]):
            Optional 2-D bandpass mask. Assumed to already be fftshifted (this
            function un-shifts it so it lines up with the raw fft output).
        return_filtered_images (bool):
            If ``True``, additionally returns the mask-filtered template and
            moving images in the image domain.
        eps (float):
            Floor to avoid division by zero in the phase-correlation
            normalization.

    Returns:
        (Union[np.ndarray, torch.Tensor, Tuple]):
            cc (np.ndarray | torch.Tensor): Phase-correlation response. Shape
                matches the broadcast of inputs. Returned as numpy if the
                template was numpy, else torch.
            (optionally) filtered template and moving images in the image
            domain — only returned when ``return_filtered_images=True``.
    """
    fft2, fftshift, ifft2 = torch.fft.fft2, torch.fft.fftshift, torch.fft.ifft2
    axes = (-2, -1)

    return_numpy = isinstance(im_template, np.ndarray)
    im_template = torch.as_tensor(im_template)
    im_moving = torch.as_tensor(im_moving)

    fft_template = fft2(im_template, dim=axes)
    fft_moving = fft2(im_moving, dim=axes)

    if mask_fft is not None:
        mask_fft = torch.as_tensor(mask_fft)
        ## un-fftshift so the mask aligns with raw-FFT coordinates
        mask_fft = fftshift(mask_fft, dim=axes)
        mask = mask_fft[tuple([None] * (im_template.ndim - 2) + [slice(None)] * 2)]
        fft_template = fft_template * mask
        fft_moving = fft_moving * mask

    ## Cross-power spectrum, normalized magnitude.
    R = fft_template * torch.conj(fft_moving)
    R = R / (torch.abs(R) + eps)
    cc = fftshift(ifft2(R, dim=axes), dim=axes).real

    if not return_filtered_images:
        return cc.cpu().numpy() if return_numpy else cc
    if return_numpy:
        return (
            cc.cpu().numpy(),
            torch.abs(ifft2(fft_template, dim=axes)).cpu().numpy(),
            torch.abs(ifft2(fft_moving, dim=axes)).cpu().numpy(),
        )
    return cc, torch.abs(ifft2(fft_template, dim=axes)), torch.abs(ifft2(fft_moving, dim=axes))


def get_path_between_nodes(
    idx_start: int,
    idx_end: int,
    predecessors: np.ndarray,
    max_length: int = 9999,
) -> List[int]:
    """Reconstruct a shortest path from a predecessor matrix.

    Ported from ROICaT helpers.get_path_between_nodes. The predecessor matrix
    is the one returned by :func:`scipy.sparse.csgraph.shortest_path` (so
    ``predecessors[idx_end, idx_current]`` gives the previous node on the
    shortest path from ``idx_current`` to ``idx_end``).

    Args:
        idx_start (int): First node.
        idx_end (int): Destination node.
        predecessors (np.ndarray): Square predecessor matrix.
        max_length (int): Safety cap to avoid infinite loops.

    Returns:
        (List[int]): Node indices along the shortest path,
        ``[idx_start, ..., idx_end]``.

    Raises:
        AssertionError: Input validation (shapes, integer types, no-path placeholder).
        ValueError: Path length exceeds ``max_length``.
    """
    assert idx_start < predecessors.shape[0], "idx_start is out of range"
    assert idx_end < predecessors.shape[0], "idx_end is out of range"
    assert predecessors.ndim == 2, "predecessors matrix must be 2D"
    assert predecessors.shape[0] == predecessors.shape[1], "predecessors matrix must be square"
    assert isinstance(idx_start, int), "idx_start must be an integer"
    assert isinstance(idx_end, int), "idx_end must be an integer"
    assert isinstance(max_length, int), "max_length must be an integer"
    assert predecessors[idx_end, idx_start] != -9999, (
        f"Possibly no path exists. predecessors[{idx_end}, {idx_start}] == -9999."
    )

    path = [int(idx_start)]
    idx_current = int(idx_start)
    while idx_current != idx_end:
        if len(path) > max_length:
            raise ValueError("Path length exceeds max_length")
        idx_current = int(predecessors[idx_end, idx_current])
        path.append(idx_current)
    return path


## ---------------------------------------------------------------------------
## ImageAlignmentChecker — always instantiated inside fit_geometric, so must
## be ported verbatim. Uses phase_correlation + make_2D_frequency_filter.
## ---------------------------------------------------------------------------

class ImageAlignmentChecker:
    """Score whether a set of images is spatially aligned via phase correlation.

    Ported from ROICaT helpers.ImageAlignmentChecker.

    The class constructs two band-selectable 2-D filters in the phase-correlation
    domain — an "in" filter over the center (within ``radius_in``) and an
    "out" filter away from the center — and compares statistics of the
    phase-correlation peak under each filter to get an alignment z-score.

    Args:
        hw (Tuple[int, int]): Image height and width (all inputs must match).
        radius_in (Union[float, Tuple[float, float]]): Either the upper bound
            of the "in" bandpass (lower bound is 0) or an explicit
            ``(low, high)`` tuple.
        radius_out (Union[float, Tuple[float, float]]): Either the lower bound
            of the "out" bandpass (upper bound is ``min(H, W) / 2``) or an
            explicit ``(low, high)`` tuple.
        order (int): Butterworth order for both filters. Values > 5 may make
            the filters collapse numerically.
        device (str): Torch device string (``'cpu'``, ``'cuda:0'``, ...).
    """

    def __init__(
        self,
        hw: Tuple[int, int],
        radius_in: Union[float, Tuple[float, float]],
        radius_out: Union[float, Tuple[float, float]],
        order: int = 5,
        device: str = 'cpu',
    ):
        self.hw = tuple(hw)
        self.order = int(order)
        self.device = str(device)

        if isinstance(radius_in, (int, float, complex)):
            radius_in = (float(0.0), float(radius_in))
        elif isinstance(radius_in, (tuple, list, np.ndarray, torch.Tensor)):
            radius_in = tuple(float(r) for r in radius_in)
        else:
            raise ValueError(f"radius_in must be a float or tuple of floats. Found type: {type(radius_in)}")

        if isinstance(radius_out, (int, float, complex)):
            radius_out = (float(radius_out), float(min(self.hw)) / 2)
        elif isinstance(radius_out, (tuple, list, np.ndarray, torch.Tensor)):
            radius_out = tuple(float(r) for r in radius_out)
        else:
            raise ValueError(f"radius_out must be a float or tuple of floats. Found type: {type(radius_out)}")

        ## Precompute both 2-D filters on `device`.
        self.filt_in, self.filt_out = (
            torch.as_tensor(
                make_2D_frequency_filter(hw=self.hw, low=bp[0], high=bp[1], order=order),
                dtype=torch.float32,
                device=device,
            )
            for bp in [radius_in, radius_out]
        )

    def score_alignment(
        self,
        images: Union[np.ndarray, torch.Tensor, List, Tuple],
        images_ref: Optional[Union[np.ndarray, torch.Tensor, List, Tuple]] = None,
    ) -> Dict[str, Any]:
        """Compute per-pair alignment statistics.

        Args:
            images (Union[np.ndarray, torch.Tensor, list, tuple]): Stack of
                images, shape ``(N, H, W)`` (or ``(H, W)`` for a single).
            images_ref (Optional[...]): Reference images. If ``None``, ``images``
                is compared against itself (N×N scoring).

        Returns:
            (Dict[str, Any]): Per-pair statistics — ``'mean_in'``, ``'mean_out'``,
            ``'ptile95_out'``, ``'max_in'``, ``'std_in'``, ``'std_out'``,
            ``'max_diff'``, ``'z_in'`` (the primary score),
            ``'r_in'``, and the phase-correlation ``'pc'`` array.
        """
        def _fix_images(ims):
            assert isinstance(ims, (np.ndarray, torch.Tensor, list, tuple)), (
                f"images must be np.ndarray, torch.Tensor, or list/tuple. Got {type(ims)}"
            )
            if isinstance(ims, (list, tuple)):
                assert all(isinstance(im, (np.ndarray, torch.Tensor)) for im in ims), (
                    f"all items must be np.ndarray or torch.Tensor. Got {set(type(im) for im in ims)}"
                )
                assert all(im.ndim == 2 for im in ims), (
                    f"all items must be 2D. Got {set(im.shape for im in ims)}"
                )
                if isinstance(ims[0], np.ndarray):
                    ims = np.stack([np.array(im) for im in ims], axis=0)
                else:
                    ims = torch.stack([torch.as_tensor(im) for im in ims], dim=0)
            else:
                if ims.ndim == 2:
                    ims = ims[None, :, :]
                assert ims.ndim == 3, f"images must be 3D. Got shape {ims.shape}"
                assert ims.shape[1:] == self.hw, (
                    f"images must have shape (N, {self.hw[0]}, {self.hw[1]}). Got {ims.shape}"
                )
            return torch.as_tensor(ims, dtype=torch.float32, device=self.device)

        images = _fix_images(images)
        images_ref = _fix_images(images_ref) if images_ref is not None else images

        ## All-to-all phase correlation: shape (n_ref, n_images, H, W).
        pc = phase_correlation(images_ref[None, :, :, :], images[:, None, :, :])

        filt_in = self.filt_in[None, None, :, :]
        filt_out = self.filt_out[None, None, :, :]

        mean_out = (pc * filt_out).sum(dim=(-2, -1)) / filt_out.sum(dim=(-2, -1))
        mean_in = (pc * filt_in).sum(dim=(-2, -1)) / filt_in.sum(dim=(-2, -1))
        ## 95th-percentile of the pc image, weighted by filt_out (only count pixels inside the bandpass).
        ptile95_out = torch.quantile(
            (pc * filt_out).reshape(pc.shape[0], pc.shape[1], -1)[:, :, filt_out.reshape(-1) > 1e-3],
            0.95,
            dim=-1,
        )
        max_in = (pc * filt_in).amax(dim=(-2, -1))
        std_out = torch.sqrt(torch.mean((pc - mean_out[:, :, None, None])**2 * filt_out, dim=(-2, -1)))
        std_in = torch.sqrt(torch.mean((pc - mean_in[:, :, None, None])**2 * filt_in, dim=(-2, -1)))

        max_diff = max_in - ptile95_out
        z_in = max_diff / std_out
        r_in = max_diff / ptile95_out

        outs = {
            'pc': pc.cpu().numpy(),
            'mean_out': mean_out,
            'mean_in': mean_in,
            'ptile95_out': ptile95_out,
            'max_in': max_in,
            'std_out': std_out,
            'std_in': std_in,
            'max_diff': max_diff,
            'z_in': z_in,
            'r_in': r_in,
        }
        return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in outs.items()}

    def __call__(self, images):
        return self.score_alignment(images)


## ---------------------------------------------------------------------------
## Registration backends.
## ---------------------------------------------------------------------------

class ImageRegistrationMethod:
    """Base class for image-to-image registration backends.

    Subclasses implement either :meth:`_forward_rigid` (to emit keypoint pairs
    for the RANSAC pipeline in :meth:`fit_rigid`) or override
    :meth:`fit_rigid` directly.
    """

    def __init__(self, device: str = 'cpu', verbose: Union[bool, int] = False):
        self.device = device
        self.verbose = verbose

    @staticmethod
    def _compute_rigid_transform(
        src_pts: np.ndarray,
        dst_pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Orthogonal-Procrustes pure-rotation+translation estimate.

        Args:
            src_pts (np.ndarray): ``(N, 2)`` source points.
            dst_pts (np.ndarray): ``(N, 2)`` destination points.

        Returns:
            (Tuple[np.ndarray, np.ndarray]): ``R`` (2x2 rotation) and ``t`` (2,).
        """
        src_center = src_pts.mean(axis=0)
        dst_center = dst_pts.mean(axis=0)
        src_shifted = src_pts - src_center
        dst_shifted = dst_pts - dst_center

        U, _, Vt = np.linalg.svd(src_shifted.T @ dst_shifted)
        R = Vt.T @ U.T
        ## Reflection check.
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = dst_center - R @ src_center
        return R, t

    def fit_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        inl_thresh: float = 2.0,
        max_iter: int = 10,
        confidence: float = 0.99,
        constraint: str = 'homography',
        **kwargs,
    ) -> np.ndarray:
        """Estimate a constrained warp (3x3) between two images via RANSAC.

        Subclasses that emit keypoint pairs use this default implementation.
        The estimator branches on ``constraint``:
            - ``'rigid'``: Procrustes (rotation + translation).
            - ``'euclidean'``: :func:`skimage.measure.ransac` with :class:`skimage.transform.EuclideanTransform`.
            - ``'similarity'``: :func:`cv2.estimateAffinePartial2D`.
            - ``'affine'``: :func:`cv2.estimateAffine2D`.
            - ``'homography'``: :func:`cv2.findHomography` with MAGSAC.

        Args:
            im_template (Union[np.ndarray, torch.Tensor]): Template image.
            im_moving (Union[np.ndarray, torch.Tensor]): Moving image.
            inl_thresh (float): RANSAC inlier threshold.
            max_iter (int): RANSAC max iterations.
            confidence (float): RANSAC confidence.
            constraint (str): Warp family.

        Returns:
            (np.ndarray): 3x3 warp matrix (affine rows padded with ``[0, 0, 1]``
            where appropriate).

        Raises:
            RuntimeError: A fitting branch fails (e.g. RANSAC returns None).
            ValueError: Unknown ``constraint``.
        """
        ## 1. Detect & match keypoints (subclass).
        kptsA, kptsB = self._forward_rigid(im_template, im_moving, **kwargs)

        ## 2. Convert to numpy.
        src_pts = kptsA.cpu().numpy().astype(np.float32)
        dst_pts = kptsB.cpu().numpy().astype(np.float32)

        if len(kptsA) < 3:
            warnings.warn(f"number of points is less than needed. len(kptsA)={len(kptsA)}")
            return np.eye(3, dtype=np.float32)

        ## 3. Dispatch on constraint.
        if constraint == 'rigid':
            R, t = self._compute_rigid_transform(src_pts, dst_pts)
            warp_matrix = np.eye(3, dtype=np.float32)
            warp_matrix[:2, :2] = R
            warp_matrix[:2, 2] = t
            return warp_matrix

        if constraint == 'euclidean':
            model_robust, inliers = skimage.measure.ransac(
                (src_pts, dst_pts),
                skimage.transform.EuclideanTransform,
                min_samples=2,
                residual_threshold=inl_thresh,
                max_trials=max_iter,
                stop_probability=confidence,
            )
            if model_robust is None:
                raise RuntimeError("Euclidean (rigid) fit failed")
            return model_robust.params.astype(np.float32)

        if constraint == 'similarity':
            M_sim, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=inl_thresh,
                maxIters=max_iter,
                confidence=confidence,
            )
            if M_sim is None:
                raise RuntimeError("Similarity fit failed")
            return np.vstack([M_sim, [0.0, 0.0, 1.0]]).astype(np.float32)

        if constraint == 'affine':
            M_affine, inliers = cv2.estimateAffine2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=inl_thresh,
                maxIters=max_iter,
                confidence=confidence,
            )
            if M_affine is None:
                raise RuntimeError("Affine fit failed")
            return np.vstack([M_affine, [0.0, 0.0, 1.0]]).astype(np.float32)

        if constraint == 'homography':
            if len(kptsA) < 4:
                warnings.warn(f"number of points too few for homography. len(kptsA)={len(kptsA)}")
                return np.eye(3, dtype=np.float32)
            warp_matrix, inliers = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=inl_thresh,
                maxIters=max_iter,
                confidence=confidence,
            )
            if warp_matrix is None:
                raise RuntimeError("Homography fit failed")
            return warp_matrix.astype(np.float32)

        raise ValueError(f"Unknown constraint: {constraint}")

    def _forward_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        raise NotImplementedError(f"_forward_rigid not implemented for {self.__class__.__name__}")


class RoMa(ImageRegistrationMethod):
    """Feature-matching registration backend using the RoMa model.

    Requires the optional dependency ``romatch-roicat`` (install via
    ``pip install face-rhythm[multisession]``). The package imports as
    ``romatch`` regardless of which PyPI distribution you pulled it from.

    On first use the constructor downloads ~1.5 GB of weights via
    :func:`torch.hub.load_state_dict_from_url` into
    ``Path(torch.hub.get_dir()) / "checkpoints"``. Set the ``TORCH_HOME``
    environment variable before import to redirect the cache.

    Args:
        model_type (str): ``'outdoor'`` or ``'indoor'``.
        n_points (int): Number of matched points to sample per pair.
        batch_size (int): Sub-batch size for the matching sampler.
        device (str): Torch device string.
        weight_urls (Dict): Primary download URLs + MD5 hashes.
        fallback_weight_urls (Dict): OSF mirror URLs + matching hashes.
        verbose (bool): Verbosity.
    """

    ## Primary URL + MD5 hash. Hashes are load-bearing for the download check.
    DEFAULT_WEIGHT_URLS = {
        "romatch": {
            "outdoor": {
                "url": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
                "hash": "9a451dfb65745e777bf916db6ea84933",
                "filename": "roma_outdoor.pth",
            },
            "indoor": {
                "url": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
                "hash": "349a17aaa21883bb164b1a5884febb21",
                "filename": "roma_indoor.pth",
            },
        },
        "dinov2": {
            "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
            "hash": "19a02c10947ed50096ce382b46b15662",
            "filename": "dinov2_vitl14_pretrain.pth",
        },
    }
    DEFAULT_FALLBACK_WEIGHT_URLS = {
        "romatch": {
            "outdoor": {
                "url": "https://osf.io/cmzpa/download",
                "hash": "9a451dfb65745e777bf916db6ea84933",
                "filename": "roma_outdoor.pth",
            },
            "indoor": {
                "url": "https://osf.io/uzx64/download",
                "hash": "349a17aaa21883bb164b1a5884febb21",
                "filename": "roma_indoor.pth",
            },
        },
        "dinov2": {
            "url": "https://osf.io/tmj5c/download",
            "hash": "19a02c10947ed50096ce382b46b15662",
            "filename": "dinov2_vitl14_pretrain.pth",
        },
    }

    def __init__(
        self,
        model_type: str = 'outdoor',
        n_points: int = 10000,
        batch_size: int = 1000,
        device: str = 'cpu',
        weight_urls: Optional[Dict] = None,
        fallback_weight_urls: Optional[Dict] = None,
        verbose: bool = False,
    ):
        ## Lazy import: surface a clean error at call time, not at module import.
        try:
            import PIL  ## noqa: F401  (imported for class-level side-effect checks)
            from romatch import roma_outdoor, roma_indoor
        except ImportError as e:
            raise ImportError(
                "RoMa backend requires the optional 'multisession' extra. "
                "Install with: pip install face-rhythm[multisession]"
            ) from e

        super().__init__(device=device, verbose=verbose)

        self.roma_model_type = model_type
        self.n_points = n_points
        self.batch_size = batch_size
        self.weight_urls = weight_urls if weight_urls is not None else self.DEFAULT_WEIGHT_URLS
        self.fallback_weight_urls = fallback_weight_urls if fallback_weight_urls is not None else self.DEFAULT_FALLBACK_WEIGHT_URLS

        def download_and_check_weights(url, filename, hash):
            dir_save = str(Path(torch.hub.get_dir()) / "checkpoints")
            weights = torch.hub.load_state_dict_from_url(url, map_location=device, file_name=filename)
            path_weights = str(Path(dir_save) / filename)
            actual_hash = helpers.hash_file(path=path_weights, type_hash='MD5')
            if hash != actual_hash:
                raise ValueError(
                    f"RoMa weights hash mismatch. Expected: {hash}. Found: {actual_hash}. "
                    f"Path: {path_weights}, URL: {url}"
                )
            return weights

        def safe_download_and_check_weights(primary, fallback):
            try:
                return download_and_check_weights(url=primary["url"], filename=primary["filename"], hash=primary["hash"])
            except Exception as e:
                warnings.warn(f"Download/hash check failed: {e}. Falling back to mirror.")
                return download_and_check_weights(url=fallback["url"], filename=fallback["filename"], hash=fallback["hash"])

        weights = safe_download_and_check_weights(
            primary=self.weight_urls["romatch"][model_type],
            fallback=self.fallback_weight_urls["romatch"][model_type],
        )
        weights_dinov2 = safe_download_and_check_weights(
            primary=self.weight_urls["dinov2"],
            fallback=self.fallback_weight_urls["dinov2"],
        )

        ## use_custom_corr=False skips the optional CUDA `local_corr` extension,
        ## which the PyPI `romatch-roicat` wheel does not ship compiled.
        if model_type == 'outdoor':
            self.model = roma_outdoor(device=device, weights=weights, dinov2_weights=weights_dinov2, use_custom_corr=False)
        elif model_type == 'indoor':
            self.model = roma_indoor(device=device, weights=weights, dinov2_weights=weights_dinov2, use_custom_corr=False)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'outdoor' or 'indoor'.")

    def _match(
        self,
        im1: Union[np.ndarray, torch.Tensor],
        im2: Union[np.ndarray, torch.Tensor],
        device: Optional[str] = None,
    ):
        ff, certainty = self.model.match(
            self._prepare_image(im1),
            self._prepare_image(im2),
            device=self.device if device is None else device,
        )
        return ff, certainty

    def _forward_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ):
        h, w = im_moving.shape[0], im_moving.shape[1]

        ff, certainty = self._match(im_template, im_moving, device=self.device)

        def get_points(ff, certainty, num):
            matches, certainty = self.model.sample(ff, certainty, num=num)
            kptsA, kptsB = self.model.to_pixel_coordinates(matches, h, w, h, w)
            return kptsA, kptsB, certainty

        ## Batch the sampler into chunks of ``batch_size`` points.
        batch_ns = [int(batch.sum()) for batch in helpers.make_batches(
            np.ones(self.n_points), batch_size=self.batch_size, min_batch_size=10,
        )]
        outs = [get_points(ff, certainty, num=n) for n in batch_ns]

        kptsA, kptsB, _ = [torch.cat([out[ii] for out in outs], dim=0) for ii in range(3)]
        return kptsA, kptsB

    def _prepare_image(self, image: Union[np.ndarray, torch.Tensor]):
        """Convert a float image in [0, 1] to a PIL.Image RGB for RoMa."""
        import PIL.Image
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        return PIL.Image.fromarray(image * 255).convert("RGB")


class ECC_cv2(ImageRegistrationMethod):
    """OpenCV Enhanced Correlation Coefficient (ECC) registration.

    Wraps :func:`face_rhythm.helpers.find_geometric_transformation`
    (which itself wraps :func:`cv2.findTransformECC`). On failure, retries
    with a larger Gaussian filter size.

    Args:
        mode_transform (str): One of ``'translation'``, ``'euclidean'``,
            ``'affine'``, ``'homography'``.
        n_iter (int): Max ECC iterations.
        termination_eps (float): Convergence tolerance.
        gaussFiltSize (Union[float, int]): Gaussian-filter kernel size used
            as a smoothing pre-pass before the ECC iteration.
        auto_fix_gaussFilt_step (Optional[int]): If set, on ECC failure the
            kernel size is incremented by this value and ECC is retried
            recursively. ``None`` disables the retry.
        device (str): Ignored (ECC runs on CPU).
        verbose (Union[bool, int]): Verbosity.
    """

    def __init__(
        self,
        mode_transform: str = 'euclidean',
        n_iter: int = 200,
        termination_eps: float = 1e-09,
        gaussFiltSize: Union[float, int] = 1,
        auto_fix_gaussFilt_step: Optional[int] = 10,
        device: str = 'cpu',
        verbose: Union[bool, int] = False,
    ):
        super().__init__(device=device, verbose=verbose)

        valid_modes = {'translation', 'euclidean', 'affine', 'homography'}
        assert mode_transform in valid_modes, f"mode_transform must be one of {valid_modes}"
        assert isinstance(gaussFiltSize, (float, int)), "gaussFiltSize must be a number"

        self.mode_transform = mode_transform
        self.n_iter = n_iter
        self.termination_eps = termination_eps
        self.gaussFiltSize = int(np.round(gaussFiltSize))
        self.auto_fix_gaussFilt_step = auto_fix_gaussFilt_step

    def fit_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> np.ndarray:
        """Estimate a 3x3 warp matrix via ECC with recursive gauss-filt fallback.

        Args:
            im_template (Union[np.ndarray, torch.Tensor]): Template image.
            im_moving (Union[np.ndarray, torch.Tensor]): Moving image.

        Returns:
            (np.ndarray): 3x3 warp matrix (with ``[0, 0, 1]`` row for affine).
        """
        def _recursive_closure(im_template, im_moving, gaussFiltSize, depth=0, max_depth=100):
            depth += 1
            try:
                warp_matrix = helpers.find_geometric_transformation(
                    im_template=im_template,
                    im_moving=im_moving,
                    warp_mode=self.mode_transform,
                    n_iter=self.n_iter,
                    termination_eps=self.termination_eps,
                    gaussFiltSize=gaussFiltSize,
                )
            except Exception as e:
                if self.auto_fix_gaussFilt_step is not None:
                    if self.verbose:
                        print(f"ECC error: {e}. Retrying with larger gaussFiltSize.")
                    if depth > max_depth:
                        print(f"Reached maximum depth of {max_depth}. Returning identity warp.")
                        return np.eye(3)[:2, :] if self.mode_transform != 'homography' else np.eye(3)
                    return _recursive_closure(
                        im_template=im_template,
                        im_moving=im_moving,
                        gaussFiltSize=gaussFiltSize + self.auto_fix_gaussFilt_step,
                        depth=depth,
                        max_depth=max_depth,
                    )
                print(f"ECC failed: {e}. Defaulting to identity warp.")
                return np.eye(3)[:2, :] if self.mode_transform != 'homography' else np.eye(3)
            return warp_matrix

        warp_matrix = _recursive_closure(
            im_template=im_template,
            im_moving=im_moving,
            gaussFiltSize=self.gaussFiltSize,
        )
        ## Pad 2x3 affine → 3x3 homogeneous.
        if warp_matrix.shape[0] == 2:
            warp_matrix = np.concatenate([warp_matrix, np.array([[0, 0, 1]])], axis=0)
        return warp_matrix


class PhaseCorrelationRegistration(ImageRegistrationMethod):
    """Translation-only registration via :func:`phase_correlation` peak detection.

    Supports an optional bandpass (lowcut, highcut) on the phase-correlation
    mask for robustness against low/high-frequency noise.

    Args:
        device (str): Torch device for the FFT.
        bandpass_freqs (Optional[List[float]]): ``[low, high]`` cutoffs.
            ``None`` skips the bandpass.
        order (int): Butterworth order for the bandpass.
        verbose (bool): Verbosity.
    """

    def __init__(
        self,
        device: str = 'cpu',
        bandpass_freqs: Optional[List[float]] = None,
        order: int = 5,
        verbose: bool = False,
    ):
        super().__init__(device=device, verbose=verbose)
        self.bandpass_freqs = bandpass_freqs
        self.order = order

    def fit_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> np.ndarray:
        filt = None
        if self.bandpass_freqs is not None:
            filt = make_2D_frequency_filter(
                hw=im_template.shape[-2:],
                low=self.bandpass_freqs[0],
                high=self.bandpass_freqs[1],
                order=self.order,
            )

        cc = phase_correlation(
            im_template=im_template,
            im_moving=im_moving,
            mask_fft=filt,
            return_filtered_images=False,
            eps=1e-8,
        )

        ## Locate correlation peak, convert to (shift_x, shift_y).
        shift_y_raw, shift_x_raw = np.unravel_index(cc.argmax(), cc.shape)
        shifts = (
            float(np.ceil(cc.shape[-1] / 2) - shift_x_raw),
            float(np.floor(cc.shape[-2] / 2) - shift_y_raw),
        )

        warp_matrix = np.eye(3, dtype=np.float32)
        warp_matrix[:2, 2] = shifts
        return warp_matrix


class NullRegistration(ImageRegistrationMethod):
    """Identity registration — returns an identity warp for every pair.

    Useful for (a) debugging the fit_geometric pipeline, (b) evaluating
    pre-registered images, and (c) a zero-cost ``method`` baseline.
    """

    def __init__(self, device: Optional[str] = None, verbose: bool = False):
        super().__init__(device=device if device is not None else 'cpu', verbose=verbose)

    def fit_rigid(
        self,
        im_template: Union[np.ndarray, torch.Tensor],
        im_moving: Union[np.ndarray, torch.Tensor],
        **kwargs,
    ) -> np.ndarray:
        ## Identity affine (2x3); fit_geometric pads to 3x3.
        return np.eye(3, dtype=np.float32)[:2, :]


## ---------------------------------------------------------------------------
## Aligner — the only class users actually touch.
## ---------------------------------------------------------------------------

## Backend string → class LUT. Populated after class definitions.
_METHODS_LUT: Dict[str, type] = {
    'RoMa': RoMa,
    'ECC_cv2': ECC_cv2,
    'PhaseCorrelation': PhaseCorrelationRegistration,
    'NullRegistration': NullRegistration,
}

## Kwarg schema for each backend. Mirrors ROICaT's default so a user can
## pass the full dict (as the notebook does) and only the selected backend's
## kwargs are read.
_DEFAULT_KWARGS_METHOD: Dict[str, Dict[str, Any]] = {
    'RoMa': {
        'model_type': 'outdoor',
        'n_points': 10000,
        'batch_size': 1000,
    },
    'ECC_cv2': {
        'mode_transform': 'euclidean',
        'n_iter': 200,
        'termination_eps': 1e-09,
        'gaussFiltSize': 31,
        'auto_fix_gaussFilt_step': 10,
    },
    'PhaseCorrelation': {
        'bandpass_freqs': [1, 30],
        'order': 5,
    },
    'NullRegistration': {},
}


class Aligner(_AlignerModuleStub):
    """Register a list of FOV images to a template via a chosen backend.

    Public API mirrors ROICaT's ``tracking.alignment.Aligner`` so existing
    notebooks can swap the import path without further changes.

    Workflow:
        1. ``aligner = Aligner(...)``.
        2. ``aligner.fit_geometric(template=..., ims_moving=[...],
           method='RoMa' | 'ECC_cv2' | 'PhaseCorrelation' | 'NullRegistration',
           ...)``.
        3. Use ``aligner.remappingIdx_geo`` (list of ``(H, W, 2)`` float32
           arrays) to warp points / images, or ``aligner.transform_images(
           ims_moving, remappingIdx=aligner.remappingIdx_geo)``.
        4. Inspect alignment quality via ``aligner.plot_alignment_results_geometric()``.

    Args:
        use_match_search (bool): If any image has score <= ``z_threshold``
            against the template, run the Dijkstra match-search step to find
            a pairwise path through other images.
        all_to_all (bool): Always run the all-to-all match search, even when
            direct registrations all pass ``z_threshold``. Much slower
            (``O(N^2)``).
        radius_in (float): ``ImageAlignmentChecker`` inner radius (pixels * ``um_per_pixel``).
        radius_out (float): ``ImageAlignmentChecker`` outer radius.
        order (int): Butterworth order for the in/out filters.
        z_threshold (float): z-score cutoff below which a pair is considered
            mis-aligned. Default is ``4.0``; the multi-session notebook sets
            ``50`` to always trigger the match-search.
        um_per_pixel (float): Pixel scale (must match across all images).
        device (str): Torch device string for the backends (e.g. ``'cuda:0'``).
            Note that :class:`ECC_cv2` and :class:`PhaseCorrelationRegistration`
            ignore this and run on CPU; RoMa on CPU is prohibitively slow.
        verbose (Union[bool, int]): Verbosity (``True``, ``False``, or
            integer levels).
    """

    def __init__(
        self,
        use_match_search: bool = True,
        all_to_all: bool = False,
        radius_in: float = 4,
        radius_out: float = 20,
        order: int = 5,
        z_threshold: float = 4.0,
        um_per_pixel: float = 1.0,
        device: str = 'cpu',
        verbose: Union[bool, int] = True,
    ):
        super().__init__()

        self.params['__init__'] = self._locals_to_params(
            locals_dict=locals(),
            keys=[
                'use_match_search', 'all_to_all', 'radius_in', 'radius_out',
                'order', 'z_threshold', 'um_per_pixel', 'device', 'verbose',
            ],
        )

        self.use_match_search = use_match_search
        self.all_to_all = all_to_all
        self.radius_in = radius_in
        self.radius_out = radius_out
        self.order = order
        self.z_threshold = z_threshold
        self.device = device

        assert isinstance(um_per_pixel, (int, float, np.number)), (
            "um_per_pixel must be a single scalar. If FOV images have different "
            "pixel scales, preprocess to a uniform pixel size before alignment."
        )
        self.um_per_pixel = float(um_per_pixel)
        self._verbose = verbose

        self.remappingIdx_geo: Optional[List[np.ndarray]] = None
        self.warp_matrices = None
        self._HW: Optional[Tuple[int, int]] = None

    def _crop_image(
        self,
        image: np.ndarray,
        borders: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop ``(top, bottom, left, right)`` borders from ``image``."""
        return image[borders[0]:image.shape[0] - borders[1], borders[2]:image.shape[1] - borders[3]]

    def _compose_warps(
        self,
        warp_0: np.ndarray,
        warps_to_add: List[np.ndarray],
        warpMat_or_remapIdx: str = 'warpMat',
    ) -> np.ndarray:
        """Compose a list of warps into a single warp.

        Only the ``'warpMat'`` branch (matrix composition via
        :func:`face_rhythm.helpers.compose_transform_matrices`) is implemented
        — that's the only branch ``fit_geometric`` ever calls. The
        ``'remapIdx'`` branch exists in ROICaT for nonrigid path composition
        and raises :class:`NotImplementedError` here.
        """
        if warpMat_or_remapIdx == 'warpMat':
            fn_compose = helpers.compose_transform_matrices
        elif warpMat_or_remapIdx == 'remapIdx':
            raise NotImplementedError(
                "'remapIdx' composition is only needed for nonrigid flow composition, "
                "which this port does not include. Use 'warpMat'."
            )
        else:
            raise ValueError("warpMat_or_remapIdx must be 'warpMat' or 'remapIdx'")

        if len(warps_to_add) == 0:
            return warp_0
        warp_out = warp_0.copy()
        for warp_to_add in warps_to_add:
            warp_out = fn_compose(warp_out, warp_to_add)
        return warp_out

    def _fix_input_images(
        self,
        ims_moving: List[np.ndarray],
        template: Union[int, float, np.ndarray],
        template_method: str,
    ) -> Tuple[List[np.ndarray], Union[int, np.ndarray]]:
        """Coerce all inputs to ``float32`` and resolve ``template`` to int/ndarray."""
        if any(im.dtype != np.float32 for im in ims_moving):
            print(f"WARNING: ims_moving are not all dtype np.float32, found "
                  f"{np.unique([im.dtype for im in ims_moving])}, converting...")
        ims_moving = [im.astype(np.float32) for im in ims_moving]

        if template_method == 'image':
            if isinstance(template, int):
                assert 0 <= template < len(ims_moving), (
                    f"template must be 0 <= idx < {len(ims_moving)}, not {template}"
                )
                template = ims_moving[template]
            elif isinstance(template, float):
                assert 0.0 <= template <= 1.0, f"fractional template must be in [0, 1], not {template}"
                idx = int(len(ims_moving) * template)
                print(f"Converting fractional index {template} -> {idx}")
                template = ims_moving[idx]
            elif isinstance(template, np.ndarray):
                assert template.ndim == 2, f"template must be 2D, got ndim={template.ndim}"
            else:
                raise ValueError(f"template must be np.ndarray, int, or float, got {type(template)}")
            if template.dtype != np.float32:
                print(f"WARNING: template dtype {template.dtype} != float32, converting...")
                template = template.astype(np.float32)
        elif template_method == 'sequential':
            assert isinstance(template, (int, float)), f"template must be int/float, got {type(template)}"
            if isinstance(template, float):
                assert 0.0 <= template <= 1.0, f"fractional template must be in [0, 1], not {template}"
                idx = int(len(ims_moving) * template)
                print(f"Converting fractional index {template} -> {idx}")
                template = idx
            assert 0 <= template < len(ims_moving), f"template must be 0 <= idx < {len(ims_moving)}, not {template}"
        return ims_moving, template

    def fit_geometric(
        self,
        template: Union[int, float, np.ndarray],
        ims_moving: List[np.ndarray],
        template_method: str = 'sequential',
        mask_borders: Tuple[int, int, int, int] = (0, 0, 0, 0),
        method: str = 'RoMa',
        kwargs_method: Optional[Dict[str, Dict[str, Any]]] = None,
        constraint: str = 'affine',
        kwargs_RANSAC: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
    ) -> List[np.ndarray]:
        """Fit geometric (rigid-ish) warps from ``ims_moving`` to ``template``.

        Calls the backend identified by ``method`` once per pair, composes
        warps across sequential templates (if any), then scores alignment via
        :class:`ImageAlignmentChecker`. If any pair fails the ``z_threshold``
        gate (and ``use_match_search=True``), runs a Dijkstra search through
        all intermediate images to reconstruct better paths.

        Args:
            template (Union[int, float, np.ndarray]): Template image or index
                (fractional indices in ``[0, 1]`` are mapped to ``int(N * f)``).
            ims_moving (List[np.ndarray]): Same-shape images to register.
            template_method (str): ``'image'`` (template is a concrete image
                or pinned index) or ``'sequential'`` (each image is registered
                to its neighbor along a chain to the template index).
            mask_borders (Tuple[int, int, int, int]): Pre-crop borders
                ``(top, bottom, left, right)``.
            method (str): One of the backend keys in ``_METHODS_LUT``.
            kwargs_method (Optional[Dict]): Per-backend kwargs. The dict is
                keyed by backend name so the same dict can be passed for any
                ``method`` choice. If ``None``, uses :data:`_DEFAULT_KWARGS_METHOD`.
            constraint (str): Warp family passed through to
                :meth:`ImageRegistrationMethod.fit_rigid`.
            kwargs_RANSAC (Optional[Dict]): RANSAC kwargs for ``fit_rigid``.
            verbose (Optional[bool]): Overrides ``self._verbose`` if not ``None``.

        Returns:
            (List[np.ndarray]): ``self.remappingIdx_geo`` — one
            ``(H, W, 2) float32`` array per input image.
        """
        if kwargs_method is None:
            kwargs_method = _DEFAULT_KWARGS_METHOD
        if kwargs_RANSAC is None:
            kwargs_RANSAC = {'inl_thresh': 2.0, 'max_iter': 10, 'confidence': 0.99}

        self.params['fit_geometric'] = self._locals_to_params(
            locals_dict=locals(),
            keys=['template', 'template_method', 'mask_borders', 'method',
                  'kwargs_method', 'constraint', 'kwargs_RANSAC', 'verbose'],
        )

        verbose = verbose if verbose is not None else self._verbose

        assert method in _METHODS_LUT, f"method must be one of {list(_METHODS_LUT)}"
        self.model = _METHODS_LUT[method](
            device=self.device,
            verbose=verbose,
            **kwargs_method.get(method, {}),
        )

        assert len(ims_moving) > 0, "ims_moving must be a non-empty list of images."
        shape = ims_moving[0].shape
        for im in ims_moving:
            assert im.shape == shape, "All images in ims_moving must have the same shape."
        valid_template_methods = {'sequential', 'image'}
        assert template_method in valid_template_methods, (
            f"template_method must be one of {valid_template_methods}"
        )

        ims_moving, template = self._fix_input_images(
            ims_moving=ims_moving, template=template, template_method=template_method,
        )
        H, W = ims_moving[0].shape
        self._HW = (H, W) if self._HW is None else self._HW

        def _register(ims_moving, template, template_method):
            """Fit + compose warps across ``ims_moving`` wrt ``template``."""
            ims_moving = [self._crop_image(im, mask_borders) for im in ims_moving]
            template = self._crop_image(template, mask_borders) if isinstance(template, np.ndarray) else template

            warp_matrices_raw = []
            for ii, im_moving in tqdm(
                enumerate(ims_moving),
                desc='Finding geometric registration warps',
                total=len(ims_moving),
                disable=not self._verbose,
            ):
                if template_method == 'sequential':
                    if ii < template:
                        im_template = ims_moving[ii + 1]
                    elif ii == template:
                        im_template = ims_moving[ii]
                    else:
                        im_template = ims_moving[ii - 1]
                else:
                    im_template = template

                warp_matrix = self.model.fit_rigid(
                    im_template=im_template,
                    im_moving=im_moving,
                    constraint=constraint,
                    **kwargs_RANSAC,
                )
                warp_matrices_raw.append(warp_matrix)

            ## Compose sequential warps.
            warp_matrices = []
            if template_method == 'sequential':
                for ii in np.arange(0, template):
                    warp_composed = self._compose_warps(
                        warp_0=warp_matrices_raw[ii],
                        warps_to_add=warp_matrices_raw[ii + 1:template + 1],
                        warpMat_or_remapIdx='warpMat',
                    )
                    warp_matrices.append(warp_composed)
                warp_matrices.append(warp_matrices_raw[template])
                for ii in np.arange(template + 1, len(ims_moving)):
                    warp_composed = self._compose_warps(
                        warp_0=warp_matrices_raw[ii],
                        warps_to_add=warp_matrices_raw[template:ii][::-1],
                        warpMat_or_remapIdx='warpMat',
                    )
                    warp_matrices.append(warp_composed)
            else:
                warp_matrices = warp_matrices_raw

            ## Pad 2x3 affine to 3x3 homogeneous uniformly.
            def _extend(w):
                if w.shape == (2, 3):
                    return np.vstack([w, [0, 0, 1]])
                if w.shape == (3, 3):
                    return w
                raise ValueError(f"Unexpected warp_matrix shape: {w.shape}")
            warp_matrices = [_extend(w) for w in warp_matrices]
            return np.stack(warp_matrices, axis=0)  ## shape (N, 3, 3)

        def _register_safe(ims_moving, template, template_method):
            """Fall back to CPU if a backend raises a device-specific NotImplementedError."""
            try:
                return _register(ims_moving=ims_moving, template=template, template_method=template_method)
            except NotImplementedError as e:
                warnings.warn(f"Error during keypoint matching: {e}")
                if "is not currently implemented for the" in str(e):
                    print("Attempting fallback to CPU.")
                    self.model = _METHODS_LUT[method](
                        device='cpu',
                        verbose=verbose,
                        **kwargs_method.get(method, {}),
                    )
                    return _register(ims_moving=ims_moving, template=template, template_method=template_method)
                raise

        ## Initial pass.
        warp_matrices_all_to_template = _register_safe(
            ims_moving=ims_moving, template=template, template_method=template_method,
        )

        ## Alignment scoring.
        im_template_global = ims_moving[template] if template_method == 'sequential' else template
        remappingIdx_geo_all_to_template = [
            helpers.warp_matrix_to_remappingIdx(warp_matrix=wm, x=W, y=H) for wm in warp_matrices_all_to_template
        ]
        images_warped_all_to_template = self.transform_images(
            ims_moving=ims_moving, remappingIdx=remappingIdx_geo_all_to_template,
        )

        ## IAC always lives on CPU: the filters are small and GPU memory can be scarce.
        iac_geo = ImageAlignmentChecker(
            hw=tuple(self._HW),
            radius_in=self.radius_in * self.um_per_pixel,
            radius_out=self.radius_out * self.um_per_pixel,
            order=self.order,
            device='cpu',
        )
        score_template_to_all = iac_geo.score_alignment(
            images=images_warped_all_to_template,
            images_ref=im_template_global,
        )['z_in'][:, 0]
        alignment_template_to_all = score_template_to_all > self.z_threshold
        idx_not_aligned = np.where(alignment_template_to_all == False)[0]
        if verbose:
            print(f"Alignment z_in scores: {[float(f'{s:.1f}') for s in score_template_to_all]}. "
                  f"z_threshold: {self.z_threshold}.")

        alignment_all_to_all = None
        score_all_to_all = None
        if (len(idx_not_aligned) > 0) or self.all_to_all:
            idx_toSearch = idx_not_aligned if not self.all_to_all else np.arange(len(ims_moving))
            if len(idx_not_aligned) > 0:
                print(f"Warning: Alignment failed for images idx: {idx_toSearch}.")
            if self.all_to_all:
                print("Performing all-to-all matching using the match_search algorithm.")

            if self.use_match_search or self.all_to_all:
                print("Attempting to find best matches using match search algorithm...")

                def _update_warps(idx, warp_matrices_all_to_all, alignment_all_to_all, score_all_to_all):
                    """Register ``idx`` images against all others; reconstruct paths via Dijkstra."""
                    for idx_current in idx:
                        im_current = ims_moving[idx_current]
                        warp_matrices_all_to_all[idx_current] = _register_safe(
                            ims_moving=ims_moving, template=im_current, template_method='image',
                        )
                        remappingIdx_geo_all_to_current = [
                            helpers.warp_matrix_to_remappingIdx(warp_matrix=wm, x=W, y=H)
                            for wm in warp_matrices_all_to_all[idx_current]
                        ]
                        images_warped_all_to_current = self.transform_images(
                            ims_moving=ims_moving, remappingIdx=remappingIdx_geo_all_to_current,
                        )
                        score_all_to_all[idx_current] = iac_geo.score_alignment(
                            images=images_warped_all_to_current, images_ref=im_current,
                        )['z_in'][:, 0]
                        alignment_all_to_all[idx_current] = score_all_to_all[idx_current] > self.z_threshold

                    ## Build connection graph (template = row/col 0).
                    alignment_matrix_full = np.concatenate([
                        np.concatenate([np.array(0)[None,], alignment_template_to_all])[None, :],
                        np.concatenate([alignment_template_to_all[:, None],
                                        np.nan_to_num(alignment_all_to_all, nan=0.0)], axis=1),
                    ], axis=0)
                    cost_full = np.concatenate([
                        np.concatenate([np.array(0)[None,], score_template_to_all])[None, :],
                        np.concatenate([score_template_to_all[:, None],
                                        np.nan_to_num(score_all_to_all, nan=0.0)], axis=1),
                    ], axis=0)
                    cost_full[cost_full == 0] = np.inf
                    cost_full = (1 / cost_full) * alignment_matrix_full.astype(np.float32)
                    cost_full[np.arange(len(cost_full)), np.arange(len(cost_full))] = 0.0

                    distances, predecessors = scipy.sparse.csgraph.shortest_path(
                        csgraph=scipy.sparse.csr_matrix(cost_full.astype(np.float32)),
                        method='D',
                        directed=True,
                        return_predecessors=True,
                        unweighted=False,
                    )

                    ## Reconstruct warps via path composition.
                    warp_matrices_new = []
                    for idx_im in range(len(ims_moving)):
                        if not np.isinf(distances[0, idx_im + 1]):
                            path = get_path_between_nodes(
                                idx_start=idx_im + 1,
                                idx_end=0,
                                predecessors=predecessors,
                            )
                            warps_to_add = [
                                warp_matrices_all_to_all[idx_to - 1, idx_from - 1]
                                for idx_from, idx_to in zip(path[:-2], path[1:-1])
                            ]
                            warps_to_add += [warp_matrices_all_to_template[path[-2] - 1]]
                            warp_matrix_current = self._compose_warps(
                                warp_0=np.eye(3, 3, dtype=np.float32),
                                warps_to_add=warps_to_add,
                                warpMat_or_remapIdx='warpMat',
                            )
                            warp_matrices_new.append(warp_matrix_current)
                        else:
                            warp_matrices_new.append(np.eye(3, 3, dtype=np.float32))

                    remappingIdx_new = [
                        helpers.warp_matrix_to_remappingIdx(warp_matrix=wm, x=W, y=H) for wm in warp_matrices_new
                    ]
                    images_warped_new = self.transform_images(
                        ims_moving=ims_moving, remappingIdx=remappingIdx_new,
                    )
                    score_template_new = iac_geo.score_alignment(
                        images=images_warped_new, images_ref=im_template_global,
                    )['z_in'][:, 0]
                    alignment_template_new = score_template_new > self.z_threshold
                    idx_no_path = np.where(np.logical_not(alignment_template_new))[0]

                    if len(idx_no_path) > 0:
                        print(f"Warning: Could not find a path to alignment for images idx: {idx_no_path}")
                    return warp_matrices_new, warp_matrices_all_to_all, alignment_all_to_all, idx_no_path

                warp_matrices_all_to_all = np.tile(
                    np.eye(3, 3)[None, None, :, :], reps=(len(ims_moving), len(ims_moving), 1, 1),
                )
                alignment_all_to_all = np.ones((len(ims_moving), len(ims_moving)), dtype=np.float32) * np.nan
                score_all_to_all = np.ones((len(ims_moving), len(ims_moving)), dtype=np.float32) * np.nan
                print(f"Finding alignment between images idx: {idx_toSearch} and all other images...")

                warp_matrices_new, warp_matrices_all_to_all, alignment_all_to_all, idx_no_path = _update_warps(
                    idx=idx_toSearch,
                    warp_matrices_all_to_all=warp_matrices_all_to_all,
                    alignment_all_to_all=alignment_all_to_all,
                    score_all_to_all=score_all_to_all,
                )
                warp_matrices_all_to_template = warp_matrices_new
                if len(idx_no_path) == 0:
                    if self._verbose:
                        print("All images aligned successfully after one round of path finding.")
                else:
                    idx_remaining = sorted(list(set(range(len(ims_moving))) - set(list(idx_toSearch))))
                    warnings.warn(
                        f"Could not find a path for image idx: {idx_no_path}. "
                        f"Now doing a dense search for alignment between all images..."
                    )
                    if self._verbose:
                        print(f"Finding alignment between remaining images and all other images: {idx_remaining}...")
                    warp_matrices_new, warp_matrices_all_to_all, alignment_all_to_all, idx_no_path = _update_warps(
                        idx=idx_remaining,
                        warp_matrices_all_to_all=warp_matrices_all_to_all,
                        alignment_all_to_all=alignment_all_to_all,
                        score_all_to_all=score_all_to_all,
                    )
                    if len(idx_no_path) == 0:
                        if self._verbose:
                            print("All images aligned successfully after dense search.")
                        warp_matrices_all_to_template = warp_matrices_new
                    else:
                        warnings.warn(
                            f"Could not find a path for image idx: {idx_no_path}. Some images may not be aligned."
                        )
            else:
                warnings.warn(
                    f"Alignment failed for images idx: {idx_not_aligned}. "
                    f"Use 'use_match_search=True' to attempt match-search."
                )
        else:
            if self._verbose:
                print("All images aligned successfully!")
            alignment_all_to_all = None

        ## Final outputs.
        self.remappingIdx_geo = [
            helpers.warp_matrix_to_remappingIdx(warp_matrix=wm, x=W, y=H) for wm in warp_matrices_all_to_template
        ]
        self.ims_registered_geo = self.transform_images(
            ims_moving=ims_moving, remappingIdx=self.remappingIdx_geo,
        )
        score_all_to_all_final = iac_geo.score_alignment(images=self.ims_registered_geo)['z_in']
        alignment_all_to_all_final = score_all_to_all_final > self.z_threshold
        score_template_to_all_final = iac_geo.score_alignment(
            images=self.ims_registered_geo, images_ref=im_template_global,
        )['z_in'][:, 0]
        alignment_template_to_all_final = score_template_to_all_final > self.z_threshold

        self.results_geometric = {
            'warp_matrices': warp_matrices_all_to_template,
            'image_alignment_checker': iac_geo,
            'direct': {
                'alignment_template_to_all': alignment_template_to_all,
                'score_template_to_all': score_template_to_all,
                'alignment_all_to_all': alignment_all_to_all,
                'score_all_to_all': score_all_to_all,
            },
            'final': {
                'alignment_all_to_all': alignment_all_to_all_final,
                'score_all_to_all': score_all_to_all_final,
                'alignment_template_to_all': alignment_template_to_all_final,
                'score_template_to_all': score_template_to_all_final,
            },
        }
        return self.remappingIdx_geo

    def transform_images(
        self,
        ims_moving: Union[List[np.ndarray], np.ndarray],
        remappingIdx: Union[List[np.ndarray], np.ndarray],
    ) -> Union[List[np.ndarray], np.ndarray]:
        """Apply per-image remapping indices via :func:`face_rhythm.helpers.remap_images`.

        Args:
            ims_moving (Union[List[np.ndarray], np.ndarray]): Images to warp
                (list) or a single ndarray (returned as a bare array).
            remappingIdx (Union[List[np.ndarray], np.ndarray]): Matching
                remap arrays, shape ``(H, W, 2)`` each. ``cv2`` backend is used
                with a per-image ``border_value = im_moving.mean()`` so the
                cropped border matches the image statistics.

        Returns:
            (Union[List[np.ndarray], np.ndarray]): Registered images.
        """
        squeeze_output = False
        if not isinstance(ims_moving, (list, tuple)):
            if isinstance(ims_moving, np.ndarray):
                ims_moving = [ims_moving]
                squeeze_output = True
            else:
                raise ValueError("ims_moving must be a list or np.ndarray")
        if not isinstance(remappingIdx, (list, tuple)):
            if isinstance(remappingIdx, np.ndarray):
                remappingIdx = [remappingIdx]
            else:
                raise ValueError("remappingIdx must be a list or np.ndarray")

        assert len(ims_moving) == len(remappingIdx), (
            "Number of images must match number of remapping indices."
        )

        ims_registered = []
        for im_moving, remapIdx in zip(ims_moving, remappingIdx):
            remapper = functools.partial(
                helpers.remap_images,
                remappingIdx=remapIdx,
                backend='cv2',  ## match ROICaT's transform_images: cv2 is required for per-channel border_value
                interpolation_method='linear',
                border_mode='constant',
                border_value=float(im_moving.mean()),
            )
            ## Handle grayscale (H, W) vs color (H, W, C) uniformly.
            if im_moving.ndim == 3:
                im_registered = np.stack(
                    [remapper(im_moving[:, :, ii]) for ii in range(im_moving.shape[2])], axis=-1,
                )
            else:
                im_registered = remapper(im_moving)
            ims_registered.append(im_registered)
        return ims_registered[0] if squeeze_output else ims_registered

    def plot_alignment_results_geometric(
        self,
        plot_direct: bool = True,
    ) -> Tuple[plt.Figure, Optional[plt.Figure]]:
        """Render two 2-panel heatmaps (score + alignment) per stage.

        Args:
            plot_direct (bool): If ``True`` and a direct all-to-all matrix was
                produced (i.e. match-search ran), also render the "direct"
                stage. Otherwise only the "final" stage is drawn.

        Returns:
            (Tuple[plt.Figure, Optional[plt.Figure]]):
                fig_final — figure for the post-registration results.
                fig_direct — figure for the direct (pre-match-search) results,
                or ``None`` if the match-search didn't run.
        """
        assert hasattr(self, 'results_geometric'), (
            "Missing results_geometric attribute. Run fit_geometric first."
        )
        assert hasattr(self, 'ims_registered_geo'), (
            "Missing ims_registered_geo attribute. Run fit_geometric first."
        )

        results_geometric = self.results_geometric
        fig_final = self._plot_results(results_geometric['final'], 'final')
        if (results_geometric['direct']['alignment_all_to_all'] is not None
                and results_geometric['direct']['score_all_to_all'] is not None
                and plot_direct):
            fig_direct = self._plot_results(results_geometric['direct'], 'direct')
        else:
            fig_direct = None
        return fig_final, fig_direct

    def _plot_results(self, results: Dict[str, Any], name: str) -> plt.Figure:
        """Two-panel score/alignment heatmap. Diagonal is zeroed out for readability."""
        inv_eye = 1 - np.eye(results['alignment_all_to_all'].shape[0])
        cmap = 'viridis'
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs = axs.flatten()
        plt.colorbar(axs[0].imshow(results['score_all_to_all'] * inv_eye, cmap=cmap))
        axs[0].set_title(f'score_all_to_all ({name})')
        axs[1].imshow(results['alignment_all_to_all'] * inv_eye, cmap=cmap)
        axs[1].set_title(f'alignment_all_to_all ({name})')
        plt.tight_layout()
        return fig

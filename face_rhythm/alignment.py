"""Image alignment pipeline and video frame ingestion.

``Image_preparation_pipeline`` builds a clean reference image for registration
from a sequence of frames by downsampling, masking with a VQT spectrogram to
keep only low-spectral-variance (non-behavior) frames, and applying CLAHE.
Also provides SFTP / local video frame extractors used to seed alignment.
"""

from typing import Dict, Optional

import os
import base64
import io
from PIL import Image

import numpy as np
import ffmpeg
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm

from face_rhythm import rois

class Image_preparation_pipeline():
    """
    Builds a clean reference image for registration by downsampling frames,
    selecting frames with low spectral variance via a VQT spectrogram, and
    applying CLAHE contrast enhancement. RH 2023

    Args:
        ds_factor (int):
            Spatial downsampling factor applied before spectral analysis.
            (Default is ``20``)
        ptile_specVar_keep (float):
            Percentile cutoff for the per-frame mean spectral magnitude;
            frames at or below this percentile are kept as **low-variance**
            (non-behavior) frames. (Default is ``10``)
        ptile_intensity_keep (float):
            Percentile cutoff used when normalizing pixel intensities (kept
            for backwards compatibility; current implementation no longer
            applies this clip). (Default is ``90``)
        params_vqt (Dict):
            Keyword arguments forwarded to ``vqt.VQT`` for the spectral
            analysis. Recognized keys include ``Fs_sample`` (sample rate),
            ``Q_lowF``, ``Q_highF`` (quality factors at the low and high
            frequency bounds), ``F_min``, ``F_max`` (frequency range),
            ``n_freq_bins`` (number of frequency bins), ``window_type``,
            ``downsample_factor``, ``fft_conv`` (use FFT-based convolution),
            and ``plot_pref``. (Default is the dictionary shown in the
            signature)
        clip_limit (float):
            ``clipLimit`` argument forwarded to OpenCV CLAHE.
            (Default is ``2.0``)
        grid_size (int):
            Tile grid size forwarded to OpenCV CLAHE. (Default is ``20``)
        verbose (bool):
            If ``True``, prints progress messages and shows intermediate
            plots. (Default is ``True``)
    """
    def __init__(
        self,
        ds_factor: int = 20,
        
        ptile_specVar_keep: float = 10,
        ptile_intensity_keep: float = 90,
        params_vqt: Dict = {
            "Fs_sample": 250,
            "Q_lowF": 3.5,
            "Q_highF": 20,
            "F_min": 0.5,
            "F_max": 60,
            "n_freq_bins": 50,
            "window_type": 'hann',
            "downsample_factor": 10,
            "fft_conv": True,
            "plot_pref": False,
        },
        
        clip_limit: float = 2.0,
        grid_size: int = 20,
        
        verbose: bool = True,
    ) -> None:
        """Initializes the pipeline and stores all preprocessing parameters."""
        self.ds_factor = ds_factor
        self.ptile_specVar_keep = ptile_specVar_keep
        self.ptile_intensity_keep = ptile_intensity_keep
        self.params_vqt = params_vqt
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.verbose = verbose

    def downsample(self, images: np.ndarray, ds_factor: Optional[int]=None) -> np.ndarray:
        """
        Spatially downsamples a stack of images by ``ds_factor`` using bilinear
        interpolation, collapsing any color channel by mean.

        Args:
            images (np.ndarray):
                Input image stack. shape: *(n_frames, H, W)* or
                *(n_frames, H, W, C)*.
            ds_factor (Optional[int]):
                Integer downsampling factor. If ``None``, ``self.ds_factor``
                is used. (Default is ``None``)

        Returns:
            (np.ndarray):
                images_ds (np.ndarray):
                    Downsampled image stack.
                    shape: *(n_frames, H // ds_factor, W // ds_factor)*,
                    dtype: *float32*.
        """
        import torch
        images_tensor = torch.as_tensor(images, dtype=torch.float32)
        images_tensor = images_tensor.mean(-1) if images_tensor.ndim == 4 else images_tensor
        ds_factor = self.ds_factor if ds_factor is None else ds_factor
        images_ds = torch.nn.functional.interpolate(images_tensor.permute(0, 1, 2)[None, ...], size=(images_tensor.shape[1]//ds_factor, images_tensor.shape[2]//ds_factor), mode='bilinear', align_corners=False).float().numpy()[0]
        
        if self.verbose:
            print(f"Images shape: {images.shape}")
            print(f"Images downsampled shape: {images_ds.shape}")
            plt.figure()
            plt.imshow(images_ds[0])
        return images_ds
    
    def find_low_spectral_variance_idx(
        self,
        images_ds: np.ndarray,
        images: np.ndarray,
        ptile_specVar_keep: Optional[float] = 10,
        ptile_intensity_keep: Optional[float] = 90,
        params_vqt: Optional[Dict] = {
            "Fs_sample": 250,
            "Q_lowF": 3.5,
            "Q_highF": 20,
            "F_min": 0.5,
            "F_max": 60,
            "n_freq_bins": 50,
            "window_type": 'hann',
            "downsample_factor": 10,
            "fft_conv": True,
            "plot_pref": False,
        },
    ):
        """
        Selects frames whose mean VQT spectral magnitude lies in the lowest
        ``ptile_specVar_keep`` percentile and returns a normalized mean image
        over those frames for use as a registration reference.

        Args:
            images_ds (np.ndarray):
                Downsampled image stack used to compute spectrograms.
                shape: *(n_frames, H_ds, W_ds)*.
            images (np.ndarray):
                Full-resolution image stack used to build the reference image.
                shape: *(n_frames, H, W)* or *(n_frames, H, W, C)*.
            ptile_specVar_keep (Optional[float]):
                Percentile cutoff for the mean spectral magnitude; frames at
                or below this percentile are kept. If ``None``,
                ``self.ptile_specVar_keep`` is used. (Default is ``10``)
            ptile_intensity_keep (Optional[float]):
                Percentile cutoff for intensity normalization (currently
                unused in the active code path; retained for backwards
                compatibility). If ``None``, ``self.ptile_intensity_keep`` is
                used. (Default is ``90``)
            params_vqt (Optional[Dict]):
                Keyword arguments forwarded to ``vqt.VQT``. If ``None``,
                ``self.params_vqt`` is used. (Default is the dictionary shown
                in the signature)

        Returns:
            (np.ndarray):
                im (np.ndarray):
                    Square-rooted, max-normalized mean of the kept full-
                    resolution frames. shape: *(H, W)*, dtype: *float32* (or
                    matching the dtype of ``images.mean(0)``).
        """
        import vqt
        import torch
        
        ptile_specVar_keep = self.ptile_specVar_keep if ptile_specVar_keep is None else ptile_specVar_keep
        ptile_intensity_keep = self.ptile_intensity_keep if ptile_intensity_keep is None else ptile_intensity_keep
        params_vqt = self.params_vqt if params_vqt is None else params_vqt
        
        my_vqt = vqt.VQT(**params_vqt)

        spec = my_vqt.forward(torch.as_tensor(images_ds, dtype=torch.float32).reshape(images_ds.shape[0], -1).T)
        spec = spec.permute(1, 2, 0).reshape(spec.shape[1], spec.shape[2], images_ds.shape[1], images_ds.shape[2])
        xAxis = my_vqt.get_xAxis(n_samples=images_ds.shape[0])
        
        if self.verbose:
            ## plot a random spectrogram
            plt.figure()
            plt.imshow(spec[:, :, np.random.randint(0, spec.shape[-2]), np.random.randint(0, spec.shape[-1])], aspect='auto')
        
        vals = spec.numpy().mean((0, 2, 3))
        idx = np.int64(xAxis[np.where(vals < np.percentile(vals, ptile_specVar_keep))[0]])

        images = images.mean(-1) if images.ndim == 4 else images
        # v = images[idx].astype(np.float32).var(0)
        # v = np.clip(v, 1e-1, np.percentile(v, ptile_intensity_keep))
        im = images[idx].mean(0)
        # im /= v
        # im = np.nan_to_num(im, 0)
        # im = np.clip(im, 0, np.percentile(im, 90))
        im = im ** 0.5
        im /= im.max()
        
        if self.verbose:
            plt.figure()
            plt.imshow(im)
            
        return im
    
    def apply_clahe(
        self,
        image: np.ndarray,
        clip_limit: Optional[float] = 2.0,
        grid_size: Optional[int] = 20,
    ) -> np.ndarray:
        """
        Applies CLAHE contrast enhancement to a single image via
        ``rois.Image_Aligner.augment_images``.

        Args:
            image (np.ndarray):
                Input image. shape: *(H, W)*.
            clip_limit (Optional[float]):
                CLAHE ``clipLimit`` argument. If ``None``, ``self.clip_limit``
                is used. (Default is ``2.0``)
            grid_size (Optional[int]):
                CLAHE tile grid size. If ``None``, ``self.grid_size`` is used.
                (Default is ``20``)

        Returns:
            (np.ndarray):
                im_aug (np.ndarray):
                    CLAHE-enhanced image. shape: *(H, W)*.
        """
        import functools
        
        clip_limit = self.clip_limit if clip_limit is None else clip_limit
        grid_size = self.grid_size if grid_size is None else grid_size

        aligner = rois.Image_Aligner(verbose=self.verbose)

        augmenter = functools.partial(
            aligner.augment_images,
            use_CLAHE=True,
            CLAHE_grid_size=grid_size,
            CLAHE_clipLimit=clip_limit,
            CLAHE_normalize=True,
        )

        im_aug = augmenter(ims=[image])[0]

        if self.verbose:
            plt.figure()
            plt.imshow(im_aug)
            
        return im_aug
    
    def apply_pipeline(
        self,
        images: np.ndarray,
    ):
        """
        Runs the full reference-image pipeline: downsample, select
        low-spectral-variance frames, then apply CLAHE.

        Args:
            images (np.ndarray):
                Input image stack. shape: *(n_frames, H, W)* or
                *(n_frames, H, W, C)*.

        Returns:
            (np.ndarray):
                im_aug (np.ndarray):
                    CLAHE-enhanced reference image. shape: *(H, W)*.
        """
        print(f"downsampling...") if self.verbose > 0 else None
        images_ds = self.downsample(images)
        print(f"computing spectrograms...") if self.verbose > 0 else None
        im = self.find_low_spectral_variance_idx(images_ds, images)
        print(f"applying CLAHE...") if self.verbose > 0 else None
        im_aug = self.apply_clahe(im)
        
        return im_aug


class SFTPVideoFrameExtractor:
    """
    Extracts frames from remote video files over SFTP and returns them as a
    NumPy array. The password is held base64-encoded with a random salt and
    only decoded transiently when constructing the SFTP URL; frame extraction
    is streamed through ffmpeg so the full file is never downloaded. RH 2023

    Args:
        host (str):
            Hostname or IP address of the remote server.
        username (str):
            Username for authenticating to the remote server.
        password (str):
            Password for authenticating to the remote server. Stored
            internally in base64-encoded form with a random salt.
        port (int):
            TCP port for the SFTP connection. (Default is ``22``)
        verbose (bool):
            If ``True``, prints progress messages during probing and frame
            retrieval. (Default is ``True``)
    """
    
    def __init__(self, host: str, username: str, password: str, port: int = 22, verbose: bool = True) -> None:
        """Stores the connection parameters and base64-encodes the password with a random salt."""
        # Encode the password in base64 for secure internal storage.
        self.host = host
        self.username = username
        self.port = port
        # encode password
        self._salt = os.urandom(32)
        self._encoded_password = base64.b64encode(self._salt + password.encode())
        self.verbose = verbose
    
    def _decode_password(self) -> str:
        """
        Decodes the stored password by stripping the random salt prefix and
        base64-decoding the remainder.

        Returns:
            (str):
                password (str):
                    Plaintext password.
        """
        return base64.b64decode(self._encoded_password)[32:].decode()
    
    def extract_frames(self, remote_video_path: str, time_start: float, duration: int, fps: Optional[float]=None) -> np.ndarray:
        """
        Streams a window of frames from a remote video over SFTP using ffmpeg
        and returns them as a stacked NumPy array. The number of frames
        returned is ``int(fps * duration)``.

        Args:
            remote_video_path (str):
                Path to the video file on the remote server, e.g.
                ``"/path/to/video.mp4"``.
            time_start (float):
                Start time, in seconds, of the extraction window.
            duration (int):
                Length of the extraction window, in seconds. The total number
                of frames returned is ``fps * duration``.
            fps (Optional[float]):
                Frame rate of the video. If ``None``, it is probed from the
                video metadata via ``ffmpeg.probe``. (Default is ``None``)

        Returns:
            (np.ndarray):
                frames (np.ndarray):
                    Decoded frames stacked along axis 0.
                    shape: *(n_frames, H, W, 3)*, dtype: *uint8*.

        Raises:
            RuntimeError:
                If ffmpeg fails while extracting frames from the SFTP stream.
        """
        # Decode password for constructing the SFTP URL.
        password = self._decode_password()
        
        # Construct the SFTP URL. Include the port only if it is not the default 22.
        if self.port == 22:
            sftp_url = f"sftp://{self.username}:{password}@{self.host}{remote_video_path}"
        else:
            sftp_url = f"sftp://{self.username}:{password}@{self.host}:{self.port}{remote_video_path}"
                
        try:
            # Use ffmpeg with a select filter to output multiple frames as PNG images.
            # Replace the ffmpeg output command with this version that explicitly forces lossless PNG output:
            ## Get fps
            if fps is None:
                print(f"probing video url") if self.verbose else None
                probe = ffmpeg.probe(sftp_url)
                print(f"getting fps of video")
                fps = eval(probe['streams'][0]['avg_frame_rate'])
                print(f"found fps: {fps}")
            num_frames = int(fps * duration)
            print(f"getting frames from remote") if self.verbose else None
            out, err = (
                ffmpeg
                .input(sftp_url, ss=time_start)
                .output(
                    "pipe:",
                    format="image2pipe",
                    vcodec="png",
                    vframes=num_frames,
                    pix_fmt="rgb24",  # Force full color lossless output.
                    vsync=0
                ).run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Error extracting frames: {e.stderr.decode()}") from e

        # Helper function to split concatenated PNG images from the pipe.
        def split_pngs(data: bytes) -> list:
            """
            Splits a buffer of concatenated PNG byte streams into a list of
            individual PNG byte strings using the PNG file signature.

            Args:
                data (bytes):
                    Concatenated PNG byte stream as produced by ffmpeg's
                    ``image2pipe`` muxer.

            Returns:
                (list):
                    images (list):
                        List of ``bytes`` objects, one per PNG image.
            """
            signature = b'\x89PNG\r\n\x1a\n'
            images = []
            start = data.find(signature)
            while start != -1:
                next_start = data.find(signature, start + 1)
                if next_start == -1:
                    images.append(data[start:])
                    break
                else:
                    images.append(data[start:next_start])
                    start = next_start
            return images

        print(f"splitting pngs") if self.verbose else None
        png_data_list = split_pngs(out)
        frames = []
        print(f"loading into an array") if self.verbose else None
        for png_data in png_data_list:
            img = Image.open(io.BytesIO(png_data))
            frames.append(np.array(img))
        return np.array(frames)
    
    
def get_frames(path, time_start, time_end, verbose=False):
    """
    Reads a contiguous range of frames from a local video using OpenCV by
    seeking with ``cv2.CAP_PROP_POS_FRAMES``. Stops early if the requested
    range extends past EOF rather than raising.

    Args:
        path (str):
            Path to the local video file.
        time_start (float):
            Start time, in seconds, of the read window.
        time_end (float):
            End time, in seconds, of the read window. The number of frames
            requested is ``int((time_end - time_start) * fps)``.
        verbose (bool):
            If ``True``, displays a tqdm progress bar over the seek loop.
            (Default is ``False``)

    Returns:
        (np.ndarray):
            ims (np.ndarray):
                Decoded frames stacked along axis 0. shape:
                *(n_frames, H, W, 3)*, dtype matches the dtype returned by
                ``cv2.VideoCapture.read`` (typically *uint8*).

    Raises:
        ValueError:
            If no frames could be read in the requested interval.
    """
    vc = cv2.VideoCapture(path)
    ## Get Fs
    fps = vc.get(cv2.CAP_PROP_FPS)
    print(f"Found fps: {fps}")
    ## Get sample_start and sample_end
    sample_start = time_start * fps
    sample_end = time_end * fps
    print(f"sample_start: {sample_start}, sample_end: {sample_end}")
    ## Get number of frames
    num_frames = int(sample_end - sample_start)
    print(f"num_frames: {num_frames}")
    ## Get frames. If the request extends past EOF we stop early rather
    ## than crashing — dtype is taken from the first successful read.
    frames = []
    for i in tqdm(range(num_frames), disable=not verbose):
        vc.set(cv2.CAP_PROP_POS_FRAMES, int(sample_start + i))
        ret, frame = vc.read()
        if not ret:
            break
        frames.append(frame)
    if not frames:
        raise ValueError(
            f"No frames could be read from {path} in the interval "
            f"[{time_start}, {time_end}] s."
        )
    ims = np.array(frames, dtype=frames[0].dtype)
    return ims
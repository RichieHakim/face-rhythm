from typing import Dict, Optional

import os
import base64
import io
from PIL import Image

import numpy as np
import ffmpeg
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from face_rhythm import rois

class Image_preparation_pipeline():
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
        self.ds_factor = ds_factor
        self.ptile_specVar_keep = ptile_specVar_keep
        self.ptile_intensity_keep = ptile_intensity_keep
        self.params_vqt = params_vqt
        self.clip_limit = clip_limit
        self.grid_size = grid_size
        self.verbose = verbose

    def downsample(self, images: np.ndarray, ds_factor: Optional[int]=None) -> np.ndarray:
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
        print(f"downsampling...") if self.verbose > 0 else None
        images_ds = self.downsample(images)
        print(f"computing spectrograms...") if self.verbose > 0 else None
        im = self.find_low_spectral_variance_idx(images_ds, images)
        print(f"applying CLAHE...") if self.verbose > 0 else None
        im_aug = self.apply_clahe(im)
        
        return im_aug


class SFTPVideoFrameExtractor:
    """
    A class to extract the first frame from remote video files via SFTP and return it as a NumPy array.
    
    The password is stored in an encoded form and only decoded when constructing the SFTP URL.
    The extraction leverages ffmpeg to stream just the required data.
    
    RH 2023
    
    Args:
        host (str):
            The hostname or IP address of the remote server.
        username (str):
            The username to authenticate with the remote server.
        password (str):
            The password to authenticate with the remote server.
        port (int, optional):
            The port to use for SFTP. Defaults to 22.
    """
    
    def __init__(self, host: str, username: str, password: str, port: int = 22, verbose: bool = True) -> None:
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
        Decodes the stored password from its encoded form.
        
        Returns:
            str:
                The decoded password.
        """
        return base64.b64decode(self._encoded_password)[32:].decode()
    
    def extract_frames(self, remote_video_path: str, time_start: float, duration: int, fps: Optional[float]=None) -> np.ndarray:
        """
        Extracts the first frame from a video file located on a remote server via SFTP,
        and returns it as a NumPy array.
        
        Args:
            remote_video_path (str):
                The path to the video file on the remote server, e.g. "/path/to/video.mp4".
            time_start (float):
                The time in seconds from which to start extracting frames.
            duration (int):
                The duration in seconds for which to extract frames.
                This is used to determine the number of frames to extract.
                The number of frames is calculated as fps * duration.
                This is used to determine the number of frames to extract.
            fps (float, optional):
                The frames per second of the video. If not provided, it will 
                be determined from the video metadata.
                
        Returns:
            np.ndarray:
                The first frame of the video as a NumPy array.
                
        Raises:
            RuntimeError: If ffmpeg fails to extract the frame.
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
            print(f"probing video url") if self.verbose else None
            probe = ffmpeg.probe(sftp_url)
            if fps is None:
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
    ## Get frames
    frames = []
    for i in tqdm(range(num_frames), disable=not verbose):
        ## Set the frame position
        vc.set(cv2.CAP_PROP_POS_FRAMES, int(sample_start + i))
        ret, frame = vc.read()
        if not ret:
            break
        frames.append(frame)
    ims = np.array(frames, dtype=frame.dtype)
    return ims
# Code adapted from  Dolensek et al 2020
# https://zenodo.org/record/3618395#.YJr0BWZKjOQ

import numpy as np

import pims
from skimage.feature import hog
from joblib import Parallel, delayed

from tqdm import tqdm


def imagesToHogsCellCrop(vid_chunk, pixelsPerCell):
    vid_chunk = np.moveaxis(vid_chunk, -1, 0)

    kwargs = dict(orientations=8, pixels_per_cell=(pixelsPerCell, pixelsPerCell), cells_per_block=(1, 1),
                  visualize=True, transform_sqrt=False)

    r = Parallel(n_jobs=32, verbose=5)(delayed(hog)(vid_chunk[i, ...], **kwargs) for i in range(len(vid_chunk)))

    hog_arrs, hog_ims = zip(*r)

    return hog_arrs, hog_ims


def hog_chunked(video, mask, crop_limits, nframes, sbin, Lyb, Lxb, cell_size):
    nt0 = min(1000, nframes)  # chunk size
    nsegs = int(np.ceil(nframes / nt0))

    # time ranges
    tf = np.floor(np.linspace(0, nframes, nsegs + 1)).astype(int)

    hog_arrays = []
    hog_images = []  # np.zeros((len(video), video.frame_shape[0], video.frame_shape[1]), np.float32)

    for n in tqdm(range(nsegs), desc="Computing HoG"):
        t = tf[n]
        im = prep_chunk(video, mask, crop_limits, t, nt0, sbin, Lyb, Lxb)

        hog_arrs, hog_ims = imagesToHogsCellCrop(im, cell_size)
        hog_arrays.append(hog_arrs)
        # hog_images[t:t+nt0,...] = hog_ims
        hog_images.append(hog_ims)

    hog_arrays = np.stack(hog_arrays)
    hog_arrays = hog_arrays.reshape(hog_arrays.shape[0] * hog_arrays.shape[1], hog_arrays.shape[2])

    return hog_arrays, hog_images


def hog_workflow(video_filepath, cell_size, mask_path='', display_plots=False):
    video = pims.Video(video_filepath)
    Ly = video.frame_shape[0]
    Lx = video.frame_shape[1]
    nframes = len(video)
    sbin = 0

    if mask_path:
        mask = np.load(mask_path)
    else:
        mask = select_mask(video_filepath)

    crop_limits = get_crop_limits(mask)

    Lyb, Lxb = get_binned_limits(sbin, crop_limits)

    hog_arrays, hog_images = hog_chunked(video, mask, crop_limits, nframes, sbin, Lyb, Lxb, cell_size)

    return hog_arrays, hog_images
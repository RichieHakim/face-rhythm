# Code adapted from  Dolensek et al 2020
# https://zenodo.org/record/3618395#.YJr0BWZKjOQ

import numpy as np
from skimage.feature import hog

from joblib import Parallel, delayed
from tqdm.auto import tqdm

from face_rhythm.util_old import helpers, batch
from .facemap import ChunkedAnalyzer


def imagesToHogsCellCrop(vid_chunk, pixelsPerCell):
    vid_chunk = np.moveaxis(vid_chunk, -1, 0)

    kwargs = dict(orientations=8, pixels_per_cell=(pixelsPerCell, pixelsPerCell), cells_per_block=(1, 1),
                  visualize=True, transform_sqrt=False)

    r = Parallel(n_jobs=32, verbose=5)(delayed(hog)(vid_chunk[i, ...], **kwargs) for i in range(len(vid_chunk)))

    hog_arrs, hog_ims = zip(*r)

    return hog_arrs, hog_ims


def hog_chunked(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb, cell_size):

    chunk_size = min(1000, nframes)
    num_chunks = int(np.ceil(nframes / chunk_size))

    chunk_starts = np.floor(np.linspace(0, nframes, num_chunks)).astype(int)

    hog_arrays = []
    hog_images = []  # np.zeros((len(video), video.frame_shape[0], video.frame_shape[1]), np.float32)

    for chunk_ind, chunk_start in enumerate(tqdm(chunk_starts, "Computing HoG")):
        video_slice = batch.chunked_video_slicer(video_paths, vid_lens, chunk_start, chunk_size)
        im = batch.prep_chunk(video_slice, mask, crop_limits, sbin, Lyb, Lxb)

        hog_arrs, hog_ims = imagesToHogsCellCrop(im, cell_size)
        hog_arrays.append(hog_arrs)
        hog_images.append(hog_ims)

    hog_arrays = np.stack(hog_arrays)
    hog_arrays = hog_arrays.reshape(hog_arrays.shape[0] * hog_arrays.shape[1], hog_arrays.shape[2])
    hog_images_flat = [im for sublist in hog_images for im in sublist]
    hog_images = np.stack(hog_images_flat)

    return hog_arrays, hog_images



def hog_workflow(config_filepath):
    config = helpers.load_config(config_filepath)

    sbin = config['Comps']['sbin']
    cell_size = config['Comps']['cell_size']

    for session in config['General']['sessions']:
        video_paths = session['videos']
        vid_lens = session['vid_lens_true']
        nframes = session['numFrames_total']
        mask = helpers.load_nwb_ts(session['nwb'],'Original Points','mask_frame_displacement')
        crop_limits = batch.get_crop_limits(mask)
        Lyb, Lxb = batch.get_binned_limits(sbin, crop_limits)

        hog_arrays, hog_images = hog_chunked(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb, cell_size)

        helpers.create_nwb_group(session['nwb'], 'HoG')
        helpers.create_nwb_ts(session['nwb'], 'HoG', 'arrays', hog_arrays, config['Video']['Fs'])
        helpers.create_nwb_ts(session['nwb'], 'HoG', 'images', hog_images, config['Video']['Fs'])


class HoG(ChunkedAnalyzer):
    def __init__(self, *args, cell_size):
        super().__init__(*args)
        self.cell_size = cell_size
        self.hog_arrays = []
        self.hog_images = []

    def images_to_hogs(self, im):
        vid_chunk = np.moveaxis(im, -1, 0)

        kwargs = dict(orientations=8, pixels_per_cell=(self.cell_size, self.cell_size), cells_per_block=(1, 1),
                      visualize=True, transform_sqrt=False)

        r = Parallel(n_jobs=32, verbose=5)(delayed(hog)(vid_chunk[i, ...], **kwargs) for i in range(len(vid_chunk)))

        hog_arrs, hog_ims = zip(*r)

        return hog_arrs, hog_ims

    def hog_chunked(self):
        chunk_size = min(1000, self.nframes)
        num_chunks = int(np.ceil(self.nframes / chunk_size))

        chunk_starts = np.floor(np.linspace(0, self.nframes, num_chunks)).astype(int)

        hog_arrays = []
        hog_images = []  # np.zeros((len(video), video.frame_shape[0], video.frame_shape[1]), np.float32)

        for chunk_ind, chunk_start in enumerate(tqdm(chunk_starts, "Computing HoG")):
            video_slice = self.chunked_video_slicer(chunk_start, chunk_size)
            im = self.prep_chunk(video_slice)

            hog_arrs, hog_ims = self.images_to_hogs(im)
            hog_arrays.append(hog_arrs)
            hog_images.append(hog_ims)

        hog_arrays = np.stack(hog_arrays)
        hog_arrays = hog_arrays.reshape(hog_arrays.shape[0] * hog_arrays.shape[1], hog_arrays.shape[2])
        hog_images_flat = [im for sublist in hog_images for im in sublist]
        hog_images = np.stack(hog_images_flat)

        self.hog_arrays = hog_arrays
        self.hog_images = hog_images

    def transform(self):
        self.hog_chunked()
        return self.hog_arrays, self.hog_images


def hog_wrapper(config_filepath):
    config = helpers.load_config(config_filepath)

    sbin = config['Comps']['sbin']
    cell_size = config['Comps']['cell_size']

    for session in config['General']['sessions']:
        video_paths = session['videos']
        vid_lens = session['vid_lens_true']
        nframes = session['numFrames_total']
        mask = helpers.load_nwb_ts(session['nwb'],'Original Points','mask_frame_displacement')
        crop_limits = batch.get_crop_limits(mask)
        Lyb, Lxb = batch.get_binned_limits(sbin, crop_limits)

        hog = HoG(video_paths, vid_lens, mask, crop_limits, sbin, Lyb, Lxb, cell_size)
        hog_arrays, hog_images = hog.transform()

        helpers.create_nwb_group(session['nwb'], 'HoG')
        helpers.create_nwb_ts(session['nwb'], 'HoG', 'arrays', hog_arrays, config['Video']['Fs'])
        helpers.create_nwb_ts(session['nwb'], 'HoG', 'images', hog_images, config['Video']['Fs'])
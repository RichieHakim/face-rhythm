# Code Adapted from FaceMap
# https://github.com/MouseLand/facemap

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import IncrementalPCA

from tqdm.auto import tqdm

from face_rhythm.util import helpers, batch
import pdb

def svdecon(X, k=100):
    np.random.seed(0)   # Fix seed to get same output for eigsh
    v0 = np.random.uniform(-1,1,size=min(X.shape))
    NN, NT = X.shape
    if NN>NT:
        COV = (X.T @ X)/NT
    else:
        COV = (X @ X.T)/NN
    if k==0:
        k = np.minimum(COV.shape) - 1
    Sv, U = eigsh(COV, k = k, v0=v0)
    U, Sv = np.real(U), np.abs(Sv)
    U, Sv = U[:, ::-1], Sv[::-1]**.5
    if NN>NT:
        V = U
        U = X @ V
        U = U/(U**2).sum(axis=0)**.5
    else:
        V = (U.T @ X).T
        V = V/(V**2).sum(axis=0)**.5
    return U, Sv, V



def mean_chunked(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb):
    # get subsampled mean across frames
    # grab up to 2000 frames to average over for mean
    nframes_subsample = min(2000, nframes)

    # load in chunks of up to 200 frames (for speed)
    chunk_size = min(200, nframes_subsample)
    num_chunks = int(np.floor(nframes_subsample / chunk_size))
    chunk_starts = np.floor(np.linspace(0, nframes - chunk_size, num_chunks)).astype(int)

    avgframe = np.zeros((Lyb, Lxb), np.float32)
    avgmotion = np.zeros((Lyb, Lxb), np.float32)

    for chunk_start in tqdm(chunk_starts, "Computing Mean"):
        video_slice = batch.chunked_video_slicer(video_paths, vid_lens, chunk_start, chunk_size)
        im = batch.prep_chunk(video_slice, mask, crop_limits, sbin, Lyb, Lxb)

        # add to averages
        avgframe += im.mean(axis=-1)
        immotion = np.abs(np.diff(im, axis=-1))
        avgmotion += immotion.mean(axis=-1)

    avgframe /= float(num_chunks)
    avgmotion /= float(num_chunks)

    return avgframe, avgmotion


def display_averages(avgframe, avgmotion):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(avgframe)
    plt.title('average frame')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(avgmotion, vmin=-10, vmax=10)
    plt.title('average motion')
    plt.axis('off')
    plt.show()


def svd_chunked(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb, avgmotion, ncomps):
    # compute incremental SVD across frames
    # load chunks of 1000 and take 250 PCs from each
    # then concatenate and take SVD of compilation of 250 PC chunks
    # number of components kept from SVD is ncomps

    chunk_size = min(1000, nframes)
    nframes_subsample = min(25000,nframes)
    num_chunks = int(np.floor(nframes_subsample / chunk_size))
    pcs_per_chunk = min(Lyb*Lxb,250)

    # what times to sample
    chunk_starts = np.floor(np.linspace(0, nframes - chunk_size, num_chunks)).astype(int)

    # giant U that we will fill up with smaller SVDs
    U = np.zeros((Lyb * Lxb, num_chunks * pcs_per_chunk), np.float32)

    for chunk_ind, chunk_start in enumerate(tqdm(chunk_starts, "Computing SVD")):
        video_slice = batch.chunked_video_slicer(video_paths, vid_lens, chunk_start, chunk_size)
        im = batch.prep_chunk(video_slice, mask, crop_limits, sbin, Lyb, Lxb)

        im = np.abs(np.diff(im, axis=-1))
        im = np.reshape(im, (Lyb * Lxb, -1))

        # subtract off average motion
        im -= avgmotion.flatten()[:, np.newaxis]

        # take SVD
        usv = svdecon(im, k=pcs_per_chunk)

        U[:, chunk_ind * pcs_per_chunk:(chunk_ind + 1) * pcs_per_chunk] = usv[0]

    # take SVD of concatenated spatial PCs
    USV = svdecon(U, k=ncomps)
    U = USV[0]

    return USV, U


def ipca_fit(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb, ncomps):
    # compute incremental SVD across frames
    # load chunks of 1000 and take 250 PCs from each
    # then concatenate and take SVD of compilation of 250 PC chunks
    # number of components kept from SVD is ncomps

    chunk_size = min(1000, nframes)
    nframes_subsample = min(25000,nframes)
    num_chunks = int(np.floor(nframes_subsample / chunk_size))
    pcs_per_chunk = min(Lyb*Lxb,250)

    # what times to sample
    chunk_starts = np.floor(np.linspace(0, nframes - chunk_size, num_chunks)).astype(int)

    # giant U that we will fill up with smaller SVDs
    ipca = IncrementalPCA(n_components=ncomps)
    U = np.zeros((Lyb * Lxb, num_chunks * pcs_per_chunk), np.float32)

    for chunk_ind, chunk_start in enumerate(tqdm(chunk_starts, "Computing SVD")):
        video_slice = batch.chunked_video_slicer(video_paths, vid_lens, chunk_start, chunk_size)
        im = batch.prep_chunk(video_slice, mask, crop_limits, sbin, Lyb, Lxb)

        im = np.abs(np.diff(im, axis=-1))
        im = np.reshape(im, (Lyb * Lxb, -1))

        # take SVD
        ipca.partial_fit(im.T)

    # take SVD of concatenated spatial PCs
    USV = (ipca.components_.T, ipca.singular_values_, ipca.components_)
    U = USV[0]

    return USV, U, ipca


def display_variance_explained(USV, U):
    plt.plot(np.cumsum(USV[1] ** 2 / (U.shape[0] - 1)))
    plt.show()


def get_variance_explained(USV,U):
    return np.cumsum(USV[1] ** 2 / (U.shape[0] - 1))


def display_eigenvectors(U, Lyb, Lxb, ncomps):
    motMask = np.reshape(U, (Lyb, Lxb, ncomps))
    plt.figure(figsize=(15, 8))
    for i in range(15):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(motMask[:, :, i] / motMask[:, :, i].std(), vmin=-2, vmax=2)
        ax.axis('off')
    plt.show()


def project_pcs_chunked(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb, avgmotion, U):
    # when do these spatial PCs occur in time?
    # project spatial PCs onto movies (in chunks again)

    ncomps = U.shape[1]
    chunk_size = min(1000, nframes)
    num_chunks = int(np.ceil(nframes / chunk_size))

    chunk_starts = np.floor(np.linspace(0, nframes, num_chunks+1)).astype(int)

    # projection of spatial PCs onto movie
    motSVD = np.zeros((nframes, ncomps), np.float32)

    for chunk_ind, chunk_start in enumerate(tqdm(chunk_starts[:-1], "Projecting PCs")):
        chunk_size = min(nframes-chunk_start, chunk_size)
        video_slice = batch.chunked_video_slicer(video_paths, vid_lens, chunk_start, chunk_size)
        im = batch.prep_chunk(video_slice, mask, crop_limits, sbin, Lyb, Lxb)

        im = np.reshape(im, (Lyb * Lxb, -1))

        # we need to keep around the last frame for the next chunk
        imend = im[:, -1]
        if chunk_ind > 0:
            im = np.concatenate((imend[:, np.newaxis], im), axis=-1)
        im = np.abs(np.diff(im, axis=-1))

        # subtract off average motion
        im -= avgmotion.flatten()[:, np.newaxis]

        # project U onto immotion
        vproj = im.T @ U
        if chunk_ind == 0:
            vproj = np.concatenate((vproj[0, :][np.newaxis, :], vproj), axis=0)

        motSVD[chunk_start:(chunk_start+chunk_size), :] = vproj

    return motSVD


def ipca_project(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb, ipca, U):
    # when do these spatial PCs occur in time?
    # project spatial PCs onto movies (in chunks again)

    ncomps = U.shape[1]
    chunk_size = min(1000, nframes)
    num_chunks = int(np.ceil(nframes / chunk_size))

    chunk_starts = np.floor(np.linspace(0, nframes, num_chunks+1)).astype(int)

    # projection of spatial PCs onto movie
    motSVD = np.zeros((nframes, ncomps), np.float32)

    for chunk_ind, chunk_start in enumerate(tqdm(chunk_starts[:-1], "Projecting PCs")):
        chunk_size = min(nframes-chunk_start, chunk_size)
        video_slice = batch.chunked_video_slicer(video_paths, vid_lens, chunk_start, chunk_size)
        im = batch.prep_chunk(video_slice, mask, crop_limits, sbin, Lyb, Lxb)

        im = np.reshape(im, (Lyb * Lxb, -1))

        # we need to keep around the last frame for the next chunk
        imend = im[:, -1]
        if chunk_ind > 0:
            im = np.concatenate((imend[:, np.newaxis], im), axis=-1)
        im = np.abs(np.diff(im, axis=-1))

        # project U onto immotion
        vproj = ipca.transform(im)
        if chunk_ind == 0:
            vproj = np.concatenate((vproj[0, :][np.newaxis, :], vproj), axis=0)

        motSVD[chunk_start:(chunk_start+chunk_size), :] = vproj

    return motSVD


def facemap_workflow(config_filepath):
    config = helpers.load_config(config_filepath)

    sbin = config['Comps']['sbin']
    ncomps = config['Comps']['ncomps']

    for session in config['General']['sessions']:
        video_paths = session['videos']
        vid_lens = session['vid_lens_true']
        nframes = session['numFrames_total']
        mask = helpers.load_nwb_ts(session['nwb'],'Original Points','mask_frame_displacement')
        crop_limits = batch.get_crop_limits(mask)
        Lyb, Lxb = batch.get_binned_limits(sbin, crop_limits)
        print(f'y:{Lyb}, x:{Lxb}, t:{nframes}')

        avgframe, avgmotion = mean_chunked(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb)
        USV, U = svd_chunked(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb, avgmotion, ncomps)
        motSVD = project_pcs_chunked(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb, avgmotion, U)

        variance_explained = get_variance_explained(USV, U)
        display_eigenvectors(U, Lyb, Lxb, ncomps)

        helpers.create_nwb_group(session['nwb'], 'FaceMap')
        helpers.create_nwb_ts(session['nwb'], 'FaceMap', 'eigenvectors', U, config['Video']['Fs'])
        helpers.create_nwb_ts(session['nwb'], 'FaceMap', 'projections', motSVD, config['Video']['Fs'])


def ipca_workflow(config_filepath):
    config = helpers.load_config(config_filepath)

    sbin = config['Comps']['sbin']
    ncomps = config['Comps']['ncomps']

    for session in config['General']['sessions']:
        video_paths = session['videos']
        vid_lens = session['vid_lens_true']
        nframes = session['numFrames_total']
        mask = helpers.load_nwb_ts(session['nwb'], 'Original Points', 'mask_frame_displacement')
        crop_limits = batch.get_crop_limits(mask)
        Lyb, Lxb = batch.get_binned_limits(sbin, crop_limits)
        print(f'y:{Lyb}, x:{Lxb}, t:{nframes}')

        USV, U, ipca = ipca_fit(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb, ncomps)
        motSVD = ipca_project(video_paths, vid_lens, nframes, mask, crop_limits, sbin, Lyb, Lxb,ipca, U)

        variance_explained = get_variance_explained(USV, U)
        display_eigenvectors(U, Lyb, Lxb, ncomps)

        helpers.create_nwb_group(session['nwb'], 'FaceMap')
        helpers.create_nwb_ts(session['nwb'], 'FaceMap', 'eigenvectors', U, config['Video']['Fs'])
        helpers.create_nwb_ts(session['nwb'], 'FaceMap', 'projections', motSVD, config['Video']['Fs'])

# def vis_stuff():
#     if display_plots:
#         display_averages(avgframe, avgmotion)
#         display_variance_explained(USV, U)
#         display_eigenvectors(U, Lyb, Lxb, ncomps)


# def display_trace_video(to_plot, motSVD, jig, start, nframes):
#     motSVD *= np.sign(skew(motSVD, axis=0))[np.newaxis, :]
#
#     cmap = cm.get_cmap('hsv')
#     cmap = cmap(np.linspace(0, .8, 10))
#
#     fig = plt.figure(figsize=(15, 3))
#     plt.subplot(121)
#     for n in range(10):
#         plt.plot(motSVD[:, n] + jig * n, color=cmap[n, :])
#     line = plt.axvline(x=0)
#     plt.subplot(122)
#     im = plt.imshow(to_plot[..., 0])
#
#     def animate(n):
#         line.set_data([n] * 100, np.linspace(0, n * jig, 100))
#         im.set_data(for_plotting[..., n])
#
#     ani = matplotlib.animation.FuncAnimation(fig, animate, frames=nframes, interval=33)
#     ani.save('output.mp4')



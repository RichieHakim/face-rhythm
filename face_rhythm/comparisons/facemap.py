# Code Adapted from FaceMap
# https://github.com/MouseLand/facemap

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigsh

from tqdm import tqdm

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


def get_binned_limits(sbin, crop_limits):
    Ly = [crop_limits[1] - crop_limits[0]]
    Lx = [crop_limits[3] - crop_limits[2]]
    Lyb, Lxb, ir = binned_inds(Ly, Lx, sbin)
    Lyb = Lyb[0]
    Lxb = Lxb[0]
    return Lyb, Lxb


def binned_inds(Ly, Lx, sbin):
    Lyb = np.zeros((len(Ly),), np.int32)
    Lxb = np.zeros((len(Ly),), np.int32)
    ir = []
    ix = 0
    for n in range(len(Ly)):
        Lyb[n] = int(np.floor(Ly[n] / sbin))
        Lxb[n] = int(np.floor(Lx[n] / sbin))
        ir.append(np.arange(ix, ix + Lyb[n] * Lxb[n], 1, int))
        ix += Lyb[n] * Lxb[n]
    return Lyb, Lxb, ir


def spatial_bin(im, sbin, Lyb, Lxb):
    imbin = im.astype(np.float32)
    if sbin > 1:
        imbin = (np.reshape(im[:, :Lyb * sbin, :Lxb * sbin], (-1, Lyb, sbin, Lxb, sbin))).mean(axis=-1).mean(axis=-2)
    return imbin


def prep_chunk(video, mask, crop_limits, t, nt0, sbin, Lyb, Lxb):
    im = np.array(video[t:t + nt0])
    # im is TIME x Ly x Lx x 3 (3 is RGB)
    if im.ndim > 3:
        im = im[:, :, :, 0]

    # mask the image
    im = np.multiply(im, mask)[:, crop_limits[0]:crop_limits[1], crop_limits[2]:crop_limits[3]]

    # spatially bin the image
    im = spatial_bin(im, sbin, Lyb, Lxb)

    # convert im to Ly x Lx x TIME
    im_old = im
    im = np.transpose(im, (1, 2, 0)).astype(np.float32)

    # most movies have integer values
    # convert to float to average
    im = im.astype(np.float32)
    return im


def mean_chunked(video, mask, crop_limits, nframes, sbin, Lyb, Lxb):
    # get subsampled mean across frames
    # grab up to 2000 frames to average over for mean
    nf = min(2000, nframes)

    # load in chunks of up to 200 frames (for speed)
    nt0 = min(200, nframes)
    nsegs = int(np.floor(nf / nt0))

    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0, nsegs)).astype(int)

    avgframe = np.zeros((Lyb, Lxb), np.float32)
    avgmotion = np.zeros((Lyb, Lxb), np.float32)

    ns = 0
    for n in tqdm(range(nsegs), "Computing Mean"):
        t = tf[n]
        im = prep_chunk(video, mask, crop_limits, t, nt0, sbin, Lyb, Lxb)

        # add to averages
        avgframe += im.mean(axis=-1)
        immotion = np.abs(np.diff(im, axis=-1))
        avgmotion += immotion.mean(axis=-1)
        ns += 1

    avgframe /= float(ns)
    avgmotion /= float(ns)

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


def svd_chunked(video, mask, crop_limits, nframes, sbin, Lyb, Lxb, avgmotion, ncomps):
    # compute incremental SVD across frames
    # load chunks of 1000 and take 250 PCs from each
    # then concatenate and take SVD of compilation of 250 PC chunks
    # number of components kept from SVD is ncomps

    nt0 = min(1000, nframes)  # chunk size
    nsegs = int(min(np.floor(25000 / nt0), np.floor(nframes / nt0)))
    nc = 250  # <- how many PCs to keep in each chunk

    # what times to sample
    tf = np.floor(np.linspace(0, nframes - nt0 - 1, nsegs)).astype(int)

    # giant U that we will fill up with smaller SVDs
    U = np.zeros((Lyb * Lxb, nsegs * nc), np.float32)

    for n in tqdm(range(nsegs), desc="Computing SVD"):
        t = tf[n]

        im = prep_chunk(video, mask, crop_limits, t, nt0, sbin, Lyb, Lxb)

        im = np.abs(np.diff(im, axis=-1))
        im = np.reshape(im, (Lyb * Lxb, -1))

        # subtract off average motion
        im -= avgmotion.flatten()[:, np.newaxis]

        # take SVD
        usv = svdecon(im, k=nc)

        U[:, n * nc:(n + 1) * nc] = usv[0]

    # take SVD of concatenated spatial PCs
    USV = svdecon(U, k=ncomps)
    U = USV[0]

    return USV, U


def display_variance_explained(USV, U):
    plt.plot(np.cumsum(USV[1] ** 2 / (U.shape[0] - 1)))
    plt.show()


def display_eigenvectors(U, Lyb, Lxb, ncomps):
    motMask = np.reshape(U, (Lyb, Lxb, ncomps))
    plt.figure(figsize=(15, 8))
    for i in range(15):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(motMask[:, :, i] / motMask[:, :, i].std(), vmin=-2, vmax=2)
        ax.axis('off')
    plt.show()


def project_pcs_chunked(video, mask, crop_limits, nframes, sbin, Lyb, Lxb, avgmotion, U):
    # when do these spatial PCs occur in time?
    # project spatial PCs onto movies (in chunks again)

    ncomps = U.shape[1]
    nt0 = min(1000, nframes)  # chunk size
    nsegs = int(np.ceil(nframes / nt0))

    # time ranges
    tf = np.floor(np.linspace(0, nframes, nsegs + 1)).astype(int)

    # projection of spatial PCs onto movie
    motSVD = np.zeros((nframes, ncomps), np.float32)

    for n in tqdm(range(nsegs), desc="Projecting SVD"):
        t = tf[n]
        im = prep_chunk(video, mask, crop_limits, t, nt0, sbin, Lyb, Lxb)

        im = np.reshape(im, (Lyb * Lxb, -1))

        # we need to keep around the last frame for the next chunk
        if n > 0:
            im = np.concatenate((imend[:, np.newaxis], im), axis=-1)
        imend = im[:, -1]
        im = np.abs(np.diff(im, axis=-1))

        # subtract off average motion
        im -= avgmotion.flatten()[:, np.newaxis]

        # project U onto immotion
        vproj = im.T @ U
        if n == 0:
            vproj = np.concatenate((vproj[0, :][np.newaxis, :], vproj), axis=0)

        motSVD[tf[n]:tf[n + 1], :] = vproj

    return motSVD


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


def batched_pca_workflow(video_filepath, mask_path='', display_plots=False, display_video=False):
    video = pims.Video(video_filepath)
    Ly = video.frame_shape[0]
    Lx = video.frame_shape[1]
    nframes = len(video)
    sbin = 4
    ncomps = 500

    if mask_path:
        mask = np.load(mask_path)
    else:
        mask = select_mask(video_filepath)

    crop_limits = get_crop_limits(mask)

    Lyb, Lxb = get_binned_limits(sbin, crop_limits)

    avgframe, avgmotion = mean_chunked(video, mask, crop_limits, nframes, sbin, Lyb, Lxb)

    USV, U = svd_chunked(video, mask, crop_limits, nframes, sbin, Lyb, Lxb, avgmotion, ncomps)

    motSVD = project_pcs_chunked(video, mask, crop_limits, nframes, sbin, Lyb, Lxb, avgmotion, U)

    if display_plots:
        display_averages(avgframe, avgmotion)
        display_variance_explained(USV, U)
        display_eigenvectors(U, Lyb, Lxb, ncomps)

    # if display_video:
    #     start = 7500
    #     nframes = 1000
    #     to_plot = prep_chunk(video, mask, crop_limits, t, nt0, sbin, Lyb, Lxb)
    #     display_trace_video(to_plot, motSVD, jig, start, nframes)

    return U, motSVD



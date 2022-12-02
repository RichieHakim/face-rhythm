import numpy as np
import imageio

def get_crop_limits(mask):
    mask_indexes = np.ix_(mask.any(1),mask.any(0))
    ymin = mask_indexes[0][0][0]
    ymax = mask_indexes[0][-1][0]
    xmin = mask_indexes[1][0,0]
    xmax = mask_indexes[1][0,-1]
    return [ymin, ymax, xmin,xmax]


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


def prep_chunk(im, mask, crop_limits, sbin, Lyb, Lxb):
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


def chunked_video_slicer(video_paths, vid_lens, chunk_start, chunk_size):
    """
    Consider recursive implementation in the odd case that the second video is
    also too short. This is unlikely, but should be covered for completeness
    Args:
        video_paths
        vid_lens
        chunk_start
        chunk_size

    Returns:
        video_slice (numpy.array)

    """
    vid_locs = np.cumsum(vid_lens)
    vid_start = (vid_locs > chunk_start).nonzero()[0][0]
    chunk_start_relative = (chunk_start - vid_locs[vid_start-1]) if vid_start > 0 else chunk_start
    video1 = imageio.get_reader(video_paths[vid_start], 'ffmpeg')
    frame_shape = video1.get_data(0).shape
    video_slice = np.zeros((chunk_size, *frame_shape))
    if chunk_start > (vid_locs[vid_start] - chunk_size):
        in_video = vid_locs[vid_start] - chunk_start
        video_slice[:in_video] = [video1.get_data(i) for i in range(chunk_start_relative,vid_lens[vid_start])]
        video2 = imageio.get_reader(video_paths[vid_start+1], 'ffmpeg')
        leftover = chunk_size - in_video
        video_slice[in_video:] = [video2.get_data(i) for i in range(0,leftover)]
    else:
        video_slice[:] = [video1.get_data(i) for i in range(chunk_start_relative,chunk_start_relative+chunk_size)]
    return video_slice
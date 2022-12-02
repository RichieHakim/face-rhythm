import scipy.interpolate
import scipy.signal

import copy
import h5py

from ..util_old import helpers

def align_jointInds_to_cameraInds(xAxis_jointInds_absoluteTime, xAxis_cameraInds_absoluteTime, cameraSignal_jointInds, interpolation_kind='cubic', axis=0):
    '''
    Interpolates signal from being on the common temporal axis to the camera time axis
    Args:
        xAxis_jointInds_absoluteTime (np.ndarray):
            1-D array where the length is the number of time points in the common temporal axis,
             and the values of the elements are the absolute times (as floats) or common
             reference time points
        xAxis_cameraInds_absoluteTime (np.ndarray):
            1-D array where the length is the number of time points in the camera's temporal axis,
             and the values of the elements are the absolute times (as floats) or common
             reference time points
         cameraSignal_jointInds (np.ndarray):
             Multidimensional array where the temporal axis is specified by arg 'axis'. This
              is the array that will be interpolated to the new temporal basis
        interpolation kind (string):
            see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
            can be: 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'
        axis (int):
            defines the axis over which the interpolation occurs
    Returns:
        interpolated output
    '''
    xAxis_scaledBetweenBounds = copy.deepcopy(xAxis_cameraInds_absoluteTime)
    xAxis_scaledBetweenBounds[xAxis_scaledBetweenBounds < xAxis_jointInds_absoluteTime[0]] = xAxis_jointInds_absoluteTime[0]
    xAxis_scaledBetweenBounds[xAxis_scaledBetweenBounds > xAxis_jointInds_absoluteTime[-1]] = xAxis_jointInds_absoluteTime[-1]
    function_interp = scipy.interpolate.interp1d(xAxis_jointInds_absoluteTime , cameraSignal_jointInds , kind=interpolation_kind , axis=axis)
    return function_interp(xAxis_scaledBetweenBounds)

def alignment_wrapper(config_filepath):
    config = helpers.load_config(config_filepath)

    neural_paths = config['Neural']['neural_paths']

    for i, session in enumerate(config['General']['sessions']):
        neural_tensor = read(neural_path[i])
        face_tensor = helpers.load_nwb_ts(session['nwb'], 'CQT','Sxx_allPixels_norm')

        neural_tensor_aligned, face_tensor_aligned = align(neural_tensor, face_tensor)

        helpers.create_nwb_group(session['nwb'], 'Neural')
        helpers.create_nwb_ts(session['nwb'], 'Neural', f'neural_tensor', neural_tensor_aligned, 1.0)
        helpers.create_nwb_ts(session['nwb'], 'Neural', f'face_tensor', face_tensor_aligned, 1.0)


def video_prep_wrapper(config_filepath):
    config = helpers.load_config(config_filepath)

    alignment_file_path = config['Neural']['alignment_file_path']
    joint_factor_path = config['Neural']['joint_factor_path']
    alpha_ind = config['Neural']['alpha_ind']

    for session in config['General']['sessions']:
        with h5py.File(alignment_file_path, 'r') as f:
            camTimes_aligned_wsTime = f['camTimes_aligned_wsTime'][()]
            camTimes_wsInd = f['camTimes_wsInd'][()]

        with h5py.File(joint_factor_path, 'r') as f:
            temporal_factors = f['factors_all']['time'][alpha_ind, ...]
            face_factors = f['factors_all']['face'][alpha_ind, ...]

        upsampled_factors = scipy.signal.resample(temporal_factors, camTimes_aligned_wsTime.shape[0], axis=1)
        cam_aligned_factors = align_jointInds_to_cameraInds(camTimes_aligned_wsTime, camTimes_wsInd, upsampled_factors, axis=1)

        helpers.create_nwb_group(session['nwb'], 'Neural')
        helpers.create_nwb_ts(session['nwb'], 'Neural', f'factors_temporal', cam_aligned_factors.T, 1.0)
        helpers.create_nwb_ts(session['nwb'], 'Neural', f'factors_spatial', face_factors.T, 1.0)
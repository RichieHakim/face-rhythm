
from ..util import helpers


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
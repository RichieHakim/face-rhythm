import numpy as np
import sklearn.decomposition
import time

from face_rhythm.util_old import helpers
import gc

def pca_workflow(config_filepath, data_key):
    """
    performs pca on the cleaned optic flow output

    Args:
        config_filepath (Path): path to the config file
        data_key (str): key to the data we wann to dimensionally reduce

    Returns:

    """

    print(f'== Beginning pca ==')
    tic_all = time.time()
    config = helpers.load_config(config_filepath)
    general = config['General']
    video = config['Video']

    for session in general['sessions']:
        tic_session = time.time()
        positions_convDR_meanSub = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', data_key)

        # input_dimRed = np.squeeze(positions_new_sansOutliers[:,1,:])
        tmp_x = np.squeeze(positions_convDR_meanSub[:,0,:])
        tmp_y = np.squeeze(positions_convDR_meanSub[:,1,:])

        input_dimRed_meanSub = np.concatenate((tmp_x - tmp_x.mean(1)[:,None] , tmp_y - tmp_y.mean(1)[:,None]) , axis=0 )

        tic = time.time()
        pca = sklearn.decomposition.PCA()
        factors_temporal = pca.fit_transform(np.float32(input_dimRed_meanSub.T))
        factors_points = pca.components_.T
        helpers.print_time('PCA complete', time.time() - tic)

        helpers.create_nwb_group(session['nwb'], 'PCA')
        helpers.create_nwb_ts(session['nwb'], 'PCA','factors_temporal',factors_temporal, 1.0)
        helpers.create_nwb_ts(session['nwb'], 'PCA', 'factors_points', factors_points, 1.0)
        helpers.create_nwb_ts(session['nwb'], 'PCA', 'input_dimRed_meanSub', input_dimRed_meanSub, video['Fs'])
        helpers.create_nwb_ts(session['nwb'], 'PCA', 'explained_variance', pca.explained_variance_ratio_, 1.0)

        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)

        del factors_temporal, factors_points, input_dimRed_meanSub, positions_convDR_meanSub, tmp_x, tmp_y
        del pca
    
    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End pca ==')

    gc.collect()
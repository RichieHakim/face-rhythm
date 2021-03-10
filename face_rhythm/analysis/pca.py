from matplotlib import pyplot as plt
import numpy as np
import sklearn.decomposition
import time

from face_rhythm.util import helpers


def plot_diagnostics(output_PCA, pca, scores_points):
    """
    displays some pca diagnostics like explained variance

    Args:
        output_PCA (np.ndarray): pca components
        pca (sklearn.PCA): pca object
        scores_points (np.ndarray): projected scores onto points

    Returns:

    """
    # plt.figure()
    # plt.imshow(positions_tracked[:,])
    plt.figure()
    plt.plot(output_PCA[:,:3])
    plt.figure()
    plt.plot(pca.explained_variance_)
    plt.figure()
    plt.plot(output_PCA[:,0] , output_PCA[:,1]  , linewidth=.1)

    plt.figure()
    plt.plot(scores_points[:,:3])


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

        input_dimRed_meanSub = np.concatenate((tmp_x - tmp_x.mean(1)[:,None] , tmp_y - tmp_y.mean(1)[:,None]) , axis=1 )
        # input_dimRed_concat = np.concatenate( (np.squeeze(positions_new_sansOutliers[:,0,:]) , np.squeeze(positions_new_sansOutliers[:,1,:])) , axis=1)

        # input_dimRed_meanSub = input_dimRed_concat - np.matlib.repmat( np.expand_dims(np.mean(input_dimRed_concat , axis=1) , axis=1) , 1 , input_dimRed_concat.shape[1])
        # input_dimRed_meanSub = input_dimRed_concat - input_dimRed_concat.mean(1)[:,None]

        tic = time.time()
        pca = sklearn.decomposition.PCA(n_components=10)
        # pca = sk.decomposition.FastICA(n_components=10)
        pca.fit(np.float32(input_dimRed_meanSub))
        output_PCA = pca.components_.transpose()
        scores_points = np.dot(input_dimRed_meanSub , output_PCA)
        helpers.print_time('PCA complete', time.time() - tic)

        plot_diagnostics(output_PCA, pca, scores_points)

        helpers.create_nwb_group(session['nwb'], 'PCA')
        helpers.create_nwb_ts(session['nwb'], 'PCA','scores_points',scores_points, video['Fs'])
        helpers.save_data(config_filepath, 'input_dimRed_meanSub', input_dimRed_meanSub)

        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)
    
    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End pca ==')
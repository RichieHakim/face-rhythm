from matplotlib import pyplot as plt
import numpy as np
import sklearn.decomposition
import time

from face_rhythm.util import helpers


def plot_diagnostics(output_PCA, pca, scores_points):
    """
    displays some pca diagnostics like explained variance

    Parameters
    ----------
    output_PCA ():
    pca ():
    scores_points ():

    Returns
    -------

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

    Parameters
    ----------
    config_filepath (Path): path to the config file

    Returns
    -------

    """

    print(f'== Beginning pca ==')
    tic_all = time.time()
    config = helpers.load_config(config_filepath)

    #positions_convDR_meanSub = helpers.load_data(config_filepath, data_key)
    positions_convDR_meanSub = helpers.load_nwb_ts(config_filepath, 'Optic Flow', data_key)
    if config['trial_inds']:
        old_shape = positions_convDR_meanSub.shape
        positions_convDR_meanSub = positions_convDR_meanSub.reshape(old_shape[0]*old_shape[1],2,-1)

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

    helpers.create_nwb_group(config_filepath, 'PCA')
    helpers.create_nwb_ts(config_filepath, 'PCA','scores_points',scores_points)
    helpers.save_data(config_filepath, 'input_dimRed_meanSub', input_dimRed_meanSub)
    
    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End pca ==')
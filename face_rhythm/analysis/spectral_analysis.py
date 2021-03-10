import time
import sys

import numpy as np
import scipy.stats
import librosa
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from face_rhythm.util import helpers
from face_rhythm.analysis import tca


def cqt_workflow(config_filepath, data_key):
    """
    computes spectral analysis on the cleaned optic flow output

    Parameters
    ----------
    config_filepath (Path): path to the config file
    data_key (str): data name on which to perform cqt

    Returns
    -------

    """
    
    print(f'== Beginning Spectrogram Computation ==')
    tic_all = time.time()

    ## get parameters
    config = helpers.load_config(config_filepath)
    general = config['General']
    cqt = config['CQT']

    hop_length = cqt['hop_length']
    fmin_rough = cqt['fmin_rough']
    sr = cqt['sr']
    n_bins = cqt['n_bins']
    bins_per_octave = cqt['bins_per_octave']
    fmin = cqt['fmin']
    pixelNum_toUse = cqt['pixelNum_toUse']

    freqs_Sxx = helpers.load_data(config_filepath, 'freqs_Sxx')

    for session in general['sessions']:
        tic_session = time.time()

        positions_convDR_meanSub = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', data_key)
        ## define positions traces to use
        # input_sgram = np.single(np.squeeze(positions_new_sansOutliers))[:,:,:]
        input_sgram = np.single(np.squeeze(positions_convDR_meanSub))[:,:,:]

        ## make a single spectrogram to get some size parameters for preallocation
        Sxx = librosa.cqt(np.squeeze(input_sgram[0,0,:]),
                          sr=sr,
                          hop_length=hop_length,
                          fmin=fmin,
                          n_bins=n_bins,
                          bins_per_octave=bins_per_octave,
                          window='hann')

        # preallocation
        tic = time.time()
        Sxx_allPixels = np.single(np.zeros((input_sgram.shape[0] , Sxx.shape[0] , Sxx.shape[1] , 2)))
        helpers.print_time('Preallocation completed', time.time() - tic_all)


        print(f'starting spectrogram calculation')
        tic = time.time()
        for ii in tqdm(range(input_sgram.shape[0]),total=Sxx_allPixels.shape[0]):

            ## iterated over x and y
            for jj in range(2):
                tmp_input_sgram = np.squeeze(input_sgram[ii,jj,:])


                tmp = librosa.cqt(np.squeeze(input_sgram[ii,jj,:]),
                                  sr=sr,
                                  hop_length=hop_length,
                                  fmin=fmin,
                                  n_bins=n_bins,
                                  bins_per_octave=bins_per_octave,
                                  window='hann')

                ## normalization
                tmp = abs(tmp) * freqs_Sxx[:,None]
        #         tmp = scipy.stats.zscore(tmp , axis=0)
        #         tmp = test - np.min(tmp , axis=0)[None,:]
        #         tmp = scipy.stats.zscore(tmp , axis=1)
        #         tmp = tmp - np.min(tmp , axis=1)[:,None]

                Sxx_allPixels[ii,:,:,jj] = tmp
        # Sxx_allPixels = Sxx_allPixels / np.std(Sxx_allPixels , axis=1)[:,None,:,:]

        print(f'completed spectrogram calculation')
        print('Info about Sxx_allPixels:\n')
        print(f'Shape: {Sxx_allPixels.shape}')
        print(f'Number of elements: {Sxx_allPixels.shape[0]*Sxx_allPixels.shape[1]*Sxx_allPixels.shape[2]*Sxx_allPixels.shape[3]}')
        print(f'Data type: {Sxx_allPixels.dtype}')
        print(f'size of Sxx_allPixels: {round(sys.getsizeof(Sxx_allPixels)/1000000000,3)} GB')
        helpers.print_time('Spectrograms computed', time.time() - tic)

        ### Normalize the spectrograms so that each time point has a similar cumulative spectral amplitude across all dots (basically, sum of power of all frequencies from all dots at a particular time should equal one)
        ## hold onto the normFactor variable because you can use to it to undo the normalization after subsequent steps
        Sxx_allPixels_normFactor = np.mean(np.sum(Sxx_allPixels , axis=1) , axis=0)
        Sxx_allPixels_norm = Sxx_allPixels / Sxx_allPixels_normFactor[None,None,:,:]
        #Sxx_allPixels_norm.shape

        plt.figure()
        plt.imshow(Sxx_allPixels_norm[cqt['pixelNum_toUse'], :, :, 0], aspect='auto', cmap='hot', origin='lower')

        plt.figure()
        plt.plot(Sxx_allPixels_normFactor)

        helpers.create_nwb_group(session['nwb'], 'CQT')
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels', Sxx_allPixels,1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_norm', Sxx_allPixels_norm,1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_normFactor', Sxx_allPixels_normFactor,1.0)

        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End spectrogram computation ==')


def cqt_positions(config_filepath):
    """
    computes spectral analysis on the cleaned optic flow output
    similar to cqt_all (consider removing/refactoring)

    Parameters
    ----------
    config_filepath (Path): path to the config file

    Returns
    -------

    """

    print(f'== Beginning Spectrogram Computation ==')
    tic_all = time.time()
    
    ## get parameters
    config = helpers.load_config(config_filepath)
    hop_length = config['cqt_hop_length']
    fmin_rough = config['cqt_fmin_rough']
    sr = config['cqt_sr']
    n_bins = config['cqt_n_bins']
    bins_per_octave = config['cqt_bins_per_octave']
    fmin = config['cqt_fmin']

    factors_np_positional = tca.load_factors(config_filepath, 'factors_positional')
    freqs_Sxx = helpers.load_data(config_filepath, 'path_freqs_Sxx')

    print(f'starting spectrogram calculation')
    tic = time.time()
    
    ## define positions traces to use
    input_sgram = np.single(np.squeeze(factors_np_positional[2][:,3]))

    ## make a single spectrogram to get some size parameters for preallocation
    Sxx_positional = librosa.cqt(np.squeeze(input_sgram), 
                                sr=sr, 
                                hop_length=hop_length, 
                                fmin=fmin, 
                                n_bins=n_bins, 
                                bins_per_octave=bins_per_octave, 
                                window=('hann'),
                                filter_scale=0.8)
    Sxx_positional = abs(Sxx_positional) * freqs_Sxx[:,None]
    # Sxx_positional = abs(Sxx_positional)
    # test = scipy.stats.zscore(Sxx_positional , axis=0)
    test_std = np.std(Sxx_positional , axis=0)
    test_sum = np.sum(Sxx_positional , axis=0)
    test = Sxx_positional / (test_std[None,:] )
    # test = (Sxx_positional) / (test_sum[None,:])
    # test = test - np.min(test , axis=0)[None,:]
    
    helpers.print_time('Spectrogram', time.time() - tic_all)

    test2 = scipy.stats.zscore(Sxx_positional , axis=1)
    test2 = test2 - np.min(test2 , axis=1)[:,None]

    plt.figure()
    plt.imshow(Sxx_positional, aspect='auto', cmap='hot', origin='lower')
    plt.figure()
    plt.imshow(test, aspect='auto', cmap='hot', origin='lower')
    plt.figure()
    plt.imshow(test2, aspect='auto', cmap='hot', origin='lower')

    helpers.create_nwb_group(config_filepath, 'CQT')
    helpers.create_nwb_ts(config_filepath, 'CQT', 'Sxx_positional', Sxx_positional)
    helpers.create_nwb_ts(config_filepath, 'CQT', 'Sxx_positional_norm', test)
    helpers.create_nwb_ts(config_filepath, 'CQT', 'Sxx_positional_normFactor', test2)
    
    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End spectrogram computation ==')
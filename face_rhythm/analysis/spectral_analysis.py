import time
import sys

import numpy as np
import scipy.stats
import librosa
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from face_rhythm.util import helpers
from face_rhythm.analysis import tca


def prepare_freqs(config_filepath):
    config = helpers.load_config(config_filepath)
    eps = 1.19209e-07 #float32 eps
    fmin_rough = config['CQT']['fmin_rough']
    sampling_rate = config['CQT']['sampling_rate']
    n_bins = config['CQT']['n_bins']

    bins_per_octave = int(np.round((n_bins) / np.log2((sampling_rate / 2) / fmin_rough)))
    fmin = ((sampling_rate / 2) / (2 ** ((n_bins) / bins_per_octave))) - (2 * eps)
    fmax = fmin * (2 ** ((n_bins) / bins_per_octave))

    freqs_Sxx = fmin * (2 ** ((np.arange(n_bins) + 1) / bins_per_octave))

    print(f'bins_per_octave: {round(bins_per_octave)} bins/octave')
    print(f'minimum frequency (fmin): {round(fmin, 3)} Hz')
    print(f'maximum frequency (fmax): {round(fmax, 8)} Hz')
    print(f'Nyquist                 : {sampling_rate / 2} Hz')
    print(f'number of frequencies   : {n_bins} bins')
    print(f'Frequencies: {np.round(freqs_Sxx, 3)}')
    plt.figure()
    plt.plot(freqs_Sxx)

    config['CQT']['bins_per_octave'] = bins_per_octave
    config['CQT']['fmin'] = fmin
    config['CQT']['fmax'] = fmax

    helpers.save_data(config_filepath, 'freqs_Sxx', freqs_Sxx)
    helpers.save_config(config, config_filepath)

def cqt_workflow(config_filepath, data_key):
    """
    computes spectral analysis on the cleaned optic flow output

    Args:
        config_filepath (Path): path to the config file
        data_key (str): data name on which to perform cqt

    Returns:

    """
    
    print(f'== Beginning Spectrogram Computation ==')
    tic_all = time.time()

    ## get parameters
    config = helpers.load_config(config_filepath)
    general = config['General']
    cqt = config['CQT']

    hop_length = cqt['hop_length']
    sampling_rate = cqt['sampling_rate']
    n_bins = cqt['n_bins']
    bins_per_octave = cqt['bins_per_octave']
    fmin = cqt['fmin']

    freqs_Sxx = helpers.load_data(config_filepath, 'freqs_Sxx')

    for session in general['sessions']:
        tic_session = time.time()

        positions_convDR_meanSub = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', data_key)
        ## define positions traces to use
        # input_sgram = np.single(np.squeeze(positions_new_sansOutliers))[:,:,:]
        input_sgram = np.single(np.squeeze(positions_convDR_meanSub))[:,:,:]

        ## make a single spectrogram to get some size parameters for preallocation
        Sxx = librosa.cqt(np.squeeze(input_sgram[0,0,:]),
                          sr=sampling_rate,
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
                                  sr=sampling_rate,
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

        helpers.create_nwb_group(session['nwb'], 'CQT')
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels', Sxx_allPixels,1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_norm', Sxx_allPixels_norm,1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_normFactor', Sxx_allPixels_normFactor,1.0)

        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End spectrogram computation ==')
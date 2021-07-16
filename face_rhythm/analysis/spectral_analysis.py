import time
import sys

import numpy as np
import scipy.stats
import librosa
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import gc

import pynwb
from numpy.linalg import norm



from face_rhythm.util import helpers


# def prepare_freqs(config_filepath):
#     config = helpers.load_config(config_filepath)
#     for session in config['General']['sessions']:
#         eps = 1.19209e-07 #float32 eps
#         fmin_rough = config['CQT']['fmin_rough']
#         sampling_rate = config['CQT']['sampling_rate']
#         n_bins = config['CQT']['n_bins']

#         bins_per_octave = int(np.round((n_bins) / np.log2((sampling_rate / 2) / fmin_rough)))
#         fmin = ((sampling_rate / 2) / (2 ** ((n_bins) / bins_per_octave))) - (2 * eps)
#         fmax = fmin * (2 ** ((n_bins) / bins_per_octave))

#         freqs_Sxx = fmin * (2 ** ((np.arange(n_bins) + 1) / bins_per_octave))

#         print(f'bins_per_octave: {round(bins_per_octave)} bins/octave')
#         print(f'minimum frequency (fmin): {round(fmin, 3)} Hz')
#         print(f'maximum frequency (fmax): {round(fmax, 8)} Hz')
#         print(f'Nyquist                 : {sampling_rate / 2} Hz')
#         print(f'number of frequencies   : {n_bins} bins')
#         print(f'Frequencies: {np.round(freqs_Sxx, 3)}')
#         plt.figure()
#         plt.plot(freqs_Sxx)

#         config['CQT']['bins_per_octave'] = bins_per_octave
#         config['CQT']['fmin'] = fmin
#         config['CQT']['fmax'] = fmax

#         helpers.save_config(config, config_filepath)
#         helpers.create_nwb_group(session['nwb'], 'CQT')
#         helpers.create_nwb_ts(session['nwb'], 'CQT', 'freqs_Sxx', freqs_Sxx, 1.0)

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

    for session in general['sessions']:
        tic_session = time.time()

        freqs_Sxx = helpers.load_nwb_ts(session['nwb'], 'CQT', 'freqs_Sxx')
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
                          tuning=0.0,
                          filter_scale=1,
                          norm=1,
                          sparsity=0.01,
                          window='hann',
                          scale=True,
                          pad_mode='reflect',
                          res_type=None,
                          dtype=None)

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

        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels', Sxx_allPixels,1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_norm', Sxx_allPixels_norm,1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_normFactor', Sxx_allPixels_normFactor,1.0)

        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)

        del Sxx, Sxx_allPixels, Sxx_allPixels_norm, Sxx_allPixels_normFactor, positions_convDR_meanSub, input_sgram

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End spectrogram computation ==')

    gc.collect()






##################################################
################ NEW RH 20210518 #################
##################################################

def get_q_filter_properties(sr,
                    fmin=None,
                    n_bins=84,
                    bins_per_octave=12,
                    window='hann',
                    filter_scale=0.5,
                    pad_fft=True,
                    norm=1,
    #                            dtype=<class 'numpy.complex64'>,
                    gamma=0,
                    plot_pref=True):
    
    freqs_Sxx = librosa.cqt_frequencies(n_bins, fmin,
                            bins_per_octave=bins_per_octave)

    filters = np.real(librosa.filters.constant_q(sr=sr,
                                                fmin=fmin,
                                                n_bins=n_bins,
                                                bins_per_octave=bins_per_octave,
                                                window=window,
                                                filter_scale=filter_scale,
                                                pad_fft=pad_fft,
                                                norm=norm,
                                #                            dtype=<class 'numpy.complex64'>,
                                                gamma=gamma,)[0])
    
    if plot_pref:
        fig, axs = plt.subplots(2, figsize=(6,8))

        axs[0].imshow(filters,
                        extent=[
                         -filters.shape[1]/(2*sr) , filters.shape[1]/(2*sr),
                            filters.shape[0], 0,
                         ],
                        aspect='auto',
                        cmap='bwr')
        ticks_toUse = np.arange(0,filters.shape[0] ,3)    
        axs[0].set_yticks(ticks_toUse)
        axs[0].set_yticklabels(np.round(freqs_Sxx[ticks_toUse],3))
        axs[0].set_ylabel('frequency of filter (Hz)')
        axs[0].set_xlabel('time (s)')
        axs[0].set_title('filters (real component)')

        axs[1].plot(freqs_Sxx)
        axs[1].set_ylabel('frequency of filter (Hz)')
        axs[1].set_xlabel('filter number')
        axs[1].set_title('frequencies')
    
    return freqs_Sxx


def prepare_freqs(config_filepath, plot_pref=True):
    config = helpers.load_config(config_filepath)
    for session in config['General']['sessions']: # why is this all in a for loop?

        hop_length      = config['CQT']['hop_length']
        fmin_rough      = config['CQT']['fmin_rough']
        fmax_rough      = config['CQT']['fmax_rough']
        sampling_rate   = config['CQT']['sampling_rate']
        n_bins          = config['CQT']['n_bins']
        filter_scale    = config['CQT']['filter_scale']
        gamma           = config['CQT']['gamma']


        f_nyquist = sampling_rate / 2
        n_octaves_rough = np.log2(f_nyquist / fmin_rough)
        bins_per_octave_rough = np.ceil(n_bins / n_octaves_rough)

        freqs_Sxx_all = get_q_filter_properties(
                                sampling_rate,
                                fmin=fmin_rough,
                                n_bins=n_bins,
                                bins_per_octave=bins_per_octave_rough,
                                window='hann',
                                filter_scale=filter_scale,
                                pad_fft=False,
                                norm=10,
#                            dtype=<class 'numpy.complex64'>,
                                gamma=gamma,
                                plot_pref=plot_pref,
                           )
        freq_idx_toUse = (freqs_Sxx_all <= fmax_rough) * (freqs_Sxx_all >= fmin_rough)
        freqs_Sxx_toUse = freqs_Sxx_all[freq_idx_toUse]
        fmin = freqs_Sxx_toUse[0]
        fmax = freqs_Sxx_toUse[-1]
        n_octaves = np.log2(f_nyquist / fmin)
        # bins_per_octave = int(np.round(n_bins / n_octaves))
        bins_per_octave = np.ceil(n_bins / n_octaves)

        print(f'octaves: {n_octaves} octaves')
        print(f'bins_per_octave: {bins_per_octave} bins/octave')
        print(f'minimum frequency (fmin): {round(fmin, 4)} Hz')
        print(f'maximum frequency (fmax): {round(fmax, 8)} Hz')
        print(f'Nyquist                 : {sampling_rate / 2} Hz')
        print(f'number of frequencies   : {n_bins} bins')

        # print(f'Frequencies: {np.round(freqs_Sxx, 3)}')


        config['CQT']['bins_per_octave'] = int(bins_per_octave)
        config['CQT']['n_octaves'] = float(n_octaves)
        config['CQT']['fmin'] = float(fmin)
        config['CQT']['fmax'] = float(fmax)
        config['CQT']['fmax'] = float(fmax)

        # print(config)
        helpers.save_config(config, config_filepath)
        helpers.create_nwb_group(session['nwb'], 'CQT')
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'freqs_Sxx_all', freqs_Sxx_all, 1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'freqs_Sxx_toUse', freqs_Sxx_toUse, 1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'freq_idx_toUse', freq_idx_toUse, 1.0)


def show_demo_spectrogram(config_filepath,
                        dot_toUse=0,
                        xy_toUse='x',
                        timeSeries_toUse='positions_convDR_meanSub',
                        show_filters=True, 
                        show_spectrogram=True,
                        dtype_to_estimate='float32'):
    
    if xy_toUse == 'x':
        xy_toUse = 1
    else:
        xy_toUse = 0
    
    config = helpers.load_config(config_filepath)
    nwb_path = config['General']['sessions'][0]['nwb']

    hop_length      = config['CQT']['hop_length']
    fmin_rough      = config['CQT']['fmin_rough']
    fmax_rough      = config['CQT']['fmax_rough']
    fmin            = config['CQT']['fmin']
    sampling_rate   = config['CQT']['sampling_rate']
    n_bins          = config['CQT']['n_bins']
    bins_per_octave = config['CQT']['bins_per_octave']
    filter_scale    = config['CQT']['filter_scale']
    gamma           = config['CQT']['gamma']
    spectrogram_exponent = config['CQT']['spectrogram_exponent']
    normalization_factor = config['CQT']['normalization_factor']

    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        test_timeSeries = np.array(nwbfile.processing['Face Rhythm']['Optic Flow'][timeSeries_toUse].data[dot_toUse, xy_toUse, :])
        n_dots = nwbfile.processing['Face Rhythm']['Optic Flow'][timeSeries_toUse].data.shape[0]
        freqs_Sxx = np.array(nwbfile.processing['Face Rhythm']['CQT']['freqs_Sxx_toUse'].data)
        freq_idx_toUse = np.array(nwbfile.processing['Face Rhythm']['CQT']['freq_idx_toUse'].data)

        print(test_timeSeries.shape)
        test_output = librosa.vqt(
                            test_timeSeries,
                            sr=sampling_rate,
                            hop_length=hop_length,
                            fmin=fmin,
                            n_bins=n_bins,
                            bins_per_octave=bins_per_octave,
                            gamma=gamma,
                            filter_scale=filter_scale,
                        #     tuning=0.0,
                        #     norm=1,
                            sparsity=0,
                        #     window="hann",
                        #     scale=True,
                        #     pad_mode="reflect",
                        #     res_type=None,
                        #     dtype=None,
        )

        # test_output = librosa.stft(
        #                     test_timeSeries,
        #                     n_fft=2048, hop_length=hop_length, win_length=None, window='hann', center=True, dtype=None, pad_mode='reflect'
        # )

        # test_output = mtaper_specgram(test_timeSeries)

        # plt.figure()
        # plt.imshow(test_output[2].T[:,2000:], aspect='auto')

        test_output = test_output[freq_idx_toUse,:]
        test_output_mag = np.array(np.abs(test_output) * freqs_Sxx[:,None]).astype(dtype_to_estimate)
        
        test_output_mag_norm = test_output_mag / norm(test_output_mag)
        test_normFactor = np.sum(test_output_mag_norm , axis=0)
        test_norm = test_output_mag_norm / ((normalization_factor * test_normFactor) + (1-normalization_factor))

        bitsize_for_estimation = 8 * np.random.rand(1).astype(dtype_to_estimate).nbytes
        estimated_size_of_Sxx_all = helpers.estimate_size_of_float_array( 
                                                numel=test_output.shape[0]*test_output.shape[1]*n_dots*2, 
                                                bitsize=bitsize_for_estimation)
        print(f'\nEstimated size of full spectrogram tensor: {round(estimated_size_of_Sxx_all/1000000000,5)} GB\n')
        
        fig, axs = plt.subplots(2, sharex=True, figsize=(8,8))
        axs[0].plot(np.arange(len(test_timeSeries))/sampling_rate , test_timeSeries)
        axs[0].set_xlabel('time (s)')
        axs[0].set_ylabel('positional displacement (pixels)')
                
        axs[1].imshow(
            np.abs(test_norm)**spectrogram_exponent, 
            # np.abs(test_output)**spectrogram_exponent, 
            extent= [0, len(test_timeSeries)/sampling_rate,
                    0, freqs_Sxx.shape[0]],
            aspect='auto',
            origin='lower',
            cmap='hot')
        
        ticks_toUse = np.arange(0,len(freqs_Sxx) ,3)    
        axs[1].set_yticks(ticks_toUse)
        axs[1].set_yticklabels(np.round(freqs_Sxx[ticks_toUse],3))
        axs[1].set_ylabel('frequency of filter (Hz)')
        axs[1].set_xlabel('time (s)')
        axs[1].set_title('filters (real component)')

def vqt_workflow(config_filepath, data_key, multicore_pref=True):
    """
    computes spectral analysis on the cleaned optic flow output

    Args:
        config_filepath (Path): path to the config file
        data_key (str): data name on which to perform cqt

    """
    
    print(f'== Beginning Spectrogram Computation ==')
    tic_all = time.time()

    ## get parameters
    config = helpers.load_config(config_filepath)
    general = config['General']
    cqt = config['CQT']

    hop_length      = config['CQT']['hop_length']
    fmin_rough      = config['CQT']['fmin_rough']
    fmax_rough      = config['CQT']['fmax_rough']
    fmin            = config['CQT']['fmin']
    sampling_rate   = config['CQT']['sampling_rate']
    n_bins          = config['CQT']['n_bins']
    bins_per_octave = config['CQT']['bins_per_octave']
    filter_scale    = config['CQT']['filter_scale']
    gamma           = config['CQT']['gamma']
    spectrogram_exponent = config['CQT']['spectrogram_exponent']
    normalization_factor = config['CQT']['normalization_factor']
    dtype_toUse = config['CQT']['dtype_toUse']
    


    for session in general['sessions']:
        tic_session = time.time()

        freq_idx_toUse = helpers.load_nwb_ts(session['nwb'], 'CQT', 'freq_idx_toUse')
        freqs_Sxx = helpers.load_nwb_ts(session['nwb'], 'CQT', 'freqs_Sxx_toUse')

        positions_convDR_meanSub = helpers.load_nwb_ts(session['nwb'], 'Optic Flow', data_key)
        ## define positions traces to use
        # input_sgram = np.single(np.squeeze(positions_new_sansOutliers))[:,:,:]
        input_sgram = (np.squeeze(positions_convDR_meanSub))[:,:,:]
        
        ## make a single spectrogram to get some size parameters for preallocation
        Sxx = librosa.vqt(np.squeeze(input_sgram[0,0,:]),
                            sr=sampling_rate,
                            hop_length=hop_length,
                            fmin=fmin,
                            n_bins=n_bins,
                            bins_per_octave=bins_per_octave,
                            gamma=gamma,
                            filter_scale=filter_scale,
                        #     tuning=0.0,
                        #     norm=1,
                            sparsity=0,
                        #     window="hann",
                        #     scale=True,
                        #     pad_mode="reflect",
                        #     res_type=None,
                        #     dtype=None,
                          )
        Sxx_shape = Sxx.shape

        # preallocation
        tic = time.time()

        if multicore_pref:
            def make_vqt_spectrogram(ii_dot):
                S_raw = np.zeros((2,Sxx_shape[0],Sxx_shape[1]), dtype=np.complex_)
                for ii in range(2):
                    S_raw[ii,...] = librosa.vqt(
                                        input_sgram[ii_dot,ii,:],
                                        sr=sampling_rate,
                                        hop_length=hop_length,
                                        fmin=fmin,
                                        n_bins=n_bins,
                                        bins_per_octave=bins_per_octave,
                                        gamma=gamma,
                                        filter_scale=filter_scale,
                                    #     tuning=0.0,
                                    #     norm=1,
                                    #     sparsity=0.01,
                                    #     window="hann",
                                    #     scale=True,
                                    #     pad_mode="reflect",
                                    #     res_type=None,
                                    #     dtype=None,
                                        )
                S_mag = np.array(
                    np.abs(S_raw[:, freq_idx_toUse, :]) * freqs_Sxx[None,:,None],
                    dtype=dtype_toUse)
                return S_mag
            print(f'Starting multicore pool')
            output_list = helpers.multithreading(make_vqt_spectrogram,
                                                range(input_sgram.shape[0]),
                                                workers=None)
            print(f'multicore elapsed time : {round(time.time() - tic , 2)} s. Now unpacking list into array.')
            Sxx_allPixels = np.array(output_list).transpose(0,2,3,1)
        
        else:
            Sxx_allPixels = np.array(np.zeros((input_sgram.shape[0] , len(freqs_Sxx) , Sxx.shape[1] , 2)), dtype=dtype_toUse)
            helpers.print_time('Preallocation completed', time.time() - tic_all)


            print(f'starting spectrogram calculation')
            tic = time.time()
            for ii in tqdm(range(input_sgram.shape[0]),total=Sxx_allPixels.shape[0]):

                ## iterated over x and y
                for jj in range(2):
                    tmp_input_sgram = np.squeeze(input_sgram[ii,jj,:])


                    tmp = librosa.vqt(np.squeeze(input_sgram[ii,jj,:]),
                                        sr=sampling_rate,
                                        hop_length=hop_length,
                                        fmin=fmin,
                                        n_bins=n_bins,
                                        bins_per_octave=bins_per_octave,
                                        gamma=gamma,
                                        filter_scale=filter_scale,
                                    #     tuning=0.0,
                                    #     norm=1,
                                    #     sparsity=0.01,
                                    #     window="hann",
                                    #     scale=True,
                                    #     pad_mode="reflect",
                                    #     res_type=None,
                                    #     dtype=None,
                                        )

                    ## normalization
                    tmp = np.array(
                        np.abs(tmp) * freqs_Sxx[:,None],
                        dtype=dtype_toUse)
            #         tmp = scipy.stats.zscore(tmp , axis=0)
            #         tmp = test - np.min(tmp , axis=0)[None,:]
            #         tmp = scipy.stats.zscore(tmp , axis=1)
            #         tmp = tmp - np.min(tmp , axis=1)[:,None]

                    Sxx_allPixels[ii,:,:,jj] = tmp[:, freq_idx_toUse, :]
            # Sxx_allPixels = Sxx_allPixels / np.std(Sxx_allPixels , axis=1)[:,None,:,:]

        ### Normalize the spectrograms so that each time point has a similar cumulative spectral amplitude across all dots (basically, sum of power of all frequencies from all dots at a particular time should equal one)
        ## hold onto the normFactor variable because you can use to it to undo the normalization after subsequent steps
        # Sxx_allPixels_normFactor = np.mean(np.sum(Sxx_allPixels , axis=1) , axis=0)
        # Sxx_allPixels_norm = Sxx_allPixels / Sxx_allPixels_normFactor[None,None,:,:]
        ##Sxx_allPixels_norm.shape

        print(f'normalizing spectrograms')
        Sxx_allPixels_normFactor = np.mean(np.sum(Sxx_allPixels ** spectrogram_exponent , axis=1) , axis=0)
        Sxx_allPixels_norm = Sxx_allPixels / ((normalization_factor * Sxx_allPixels_normFactor[None, None,...]) + (1-normalization_factor))
        # test_norm = test_output_mag_norm / ((normalization_factor * Sxx_allPixels_normFactor) + (1-normalization_factor))

        Sxx_xAxis = np.arange(input_sgram.shape[2])/sampling_rate

        print(f'completed spectrogram calculation')
        print('Info about Sxx_allPixels:\n')
        print(f'Shape: {Sxx_allPixels.shape}')
        print(f'Number of elements: {Sxx_allPixels.shape[0]*Sxx_allPixels.shape[1]*Sxx_allPixels.shape[2]*Sxx_allPixels.shape[3]}')
        print(f'Data type: {Sxx_allPixels.dtype}')
        print(f'size of Sxx_allPixels: {round(sys.getsizeof(Sxx_allPixels)/1000000000,3)} GB')
        helpers.print_time('Spectrograms computed', time.time() - tic)

        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels', Sxx_allPixels,1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_norm', Sxx_allPixels_norm,1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_allPixels_normFactor', Sxx_allPixels_normFactor,1.0)
        helpers.create_nwb_ts(session['nwb'], 'CQT', 'Sxx_xAxis', Sxx_xAxis,1.0)

        helpers.print_time(f'Session {session["name"]} completed', time.time() - tic_session)

        del Sxx, Sxx_allPixels, Sxx_allPixels_norm, Sxx_allPixels_normFactor, positions_convDR_meanSub, input_sgram

    helpers.print_time('total elapsed time', time.time() - tic_all)
    print(f'== End spectrogram computation ==')

    gc.collect()


def mtaper_specgram(
    signal,
    nw=2.5,
    ntapers=None,
    win_len=0.1,
    win_overlap=0.09,
    fs=int(192e3),
    clip=None,
    freq_res_frac=1,
    mode='psd',
    **kwargs
):
    """
    Multi-taper spectrogram
    RH 2021

        Args:
            signal (array type): Signal.
            nw (float): Time-bandwidth product
            ntapers (int): Number of tapers (None to set to 2 * nw -1)
            win_len (float): Window length in seconds
            win_overlap (float): Window overlap in seconds
            fs (float): Sampling rate in Hz
            clip (2-tuple of floats): Normalize amplitudes to 0-1 using clips (in dB)
            freq_res_frac (float): frequency resolution fraction. 
                                    generates nfft. If none then nfft=None,
                                    which makes nfft=win nfft=nperseg=len_samples. 
                                    else nfft = freq_resolution_frac * round(win_len * fs)
            mode (string): mode of the scipy.signal.spectrogram to use. Can be 'psd', 'complex', ‘magnitude’, ‘angle’, ‘phase’
            **kwargs: Additional arguments for scipy.signal.spectrogram
        Returns:
            f (ndarray): Frequency bin centers
            t (ndarray): Time indices
            sxx (ndarray): Spectrogram
    """
    len_samples = np.round(win_len * fs).astype("int")
    if freq_res_frac is None:
        nfft = None
    else:
        nfft = freq_res_frac*len_samples
    if ntapers is None:
        ntapers = int(nw * 2)
    overlap_samples = np.round(win_overlap * fs)
    sequences, r = scipy.signal.windows.dpss(
        len_samples, NW=nw, Kmax=ntapers, sym=False, norm=2, return_ratios=True
    )
    sxx_ls = None
    for sequence, weight in zip(sequences, r):
        f, t, sxx = scipy.signal.spectrogram(
            signal,
            fs=fs,
            window=sequence,
            nperseg=len_samples,
            noverlap=overlap_samples,
            nfft=nfft,
            detrend='constant',
            return_onesided=True, 
            scaling='density', 
            axis=-1, 
            mode=mode,
            **kwargs
        )
        if sxx_ls is None:
            sxx_ls = sxx * weight
        else:
            sxx_ls += np.abs(sxx * weight)
    sxx = sxx_ls / len(sequences)
    if clip is not None:
        sxx = 20 * np.log10(sxx)
        sxx = sxx - clip[0]
        sxx[sxx < 0] = 0
        sxx[sxx > (clip[1] - clip[0])] = clip[1] - clip[0]
        sxx /= clip[1] - clip[0]
    return f, t, sxx


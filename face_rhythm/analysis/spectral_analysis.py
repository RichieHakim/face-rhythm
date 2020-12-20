def cqt_all(config_filepath, positions_convDR_meanSub, freqs_Sxx):
    
    print(f'== Starting CQT spectrogram calculations ==')
    tic_all = time.time()

    ## define positions traces to use
    # input_sgram = np.single(np.squeeze(positions_new_sansOutliers))[:,:,:]
    input_sgram = np.single(np.squeeze(positions_convDR_meanSub))[:,:,:]

    ## get parameters
    config = helpers.load_config(config_filepath)
    hop_length = config['cqt_hop_length']
    fmin_rough = config['cqt_fmin_rough']
    sr = config['cqt_sr']
    n_bins = config['cqt_n_bins']
    bins_per_octave = config['cqt_bins_per_octave']
    fmin = config['cqt_fmin']

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
    print(f'preallocating')
    Sxx_allPixels = np.single(np.zeros((input_sgram.shape[0] , Sxx.shape[0] , Sxx.shape[1] , 2)))  
    print(f'preallocation done. Elapsed time: {round((time.time() - tic) , 2)} seconds')

    print(f'starting spectrogram calculation')
    for ii in range(input_sgram.shape[0]):
        ## progress tracking
        if ii%50 ==0:
            print(f'{ii} / {Sxx_allPixels.shape[0]}')
        elif ii==1:
            print(f'{ii} / {Sxx_allPixels.shape[0]}')

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
    print(f'== Spectrograms computed. Total elapsed time: {round((time.time() - tic_all)/60 , 2)} minutes ==')

    ### Normalize the spectrograms so that each time point has a similar cumulative spectral amplitude across all dots (basically, sum of power of all frequencies from all dots at a particular time should equal one)
    ## hold onto the normFactor variable because you can use to it to undo the normalization after subsequent steps
    Sxx_allPixels_normFactor = np.mean(np.sum(Sxx_allPixels , axis=1) , axis=0)
    Sxx_allPixels_norm = Sxx_allPixels / Sxx_allPixels_normFactor[None,None,:,:]
    Sxx_allPixels_norm.shape
    
    return Sxx_allPixels, Sxx_allPixels_norm, Sxx_allPixels_normFactor, tmp

def cqt_positions(config_filepath, factors_np_positional, freqs_Sxx):

    ## define positions traces to use
    input_sgram = np.single(np.squeeze(factors_np_positional[2][:,3]))

    ## get parameters
    config = helpers.load_config(config_filepath)
    hop_length = config['cqt_hop_length']
    fmin_rough = config['cqt_fmin_rough']
    sr = config['cqt_sr']
    n_bins = config['cqt_n_bins']
    bins_per_octave = config['cqt_bins_per_octave']
    fmin = config['cqt_fmin']

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

    test2 = scipy.stats.zscore(Sxx_positional , axis=1)
    test2 = test2 - np.min(test2 , axis=1)[:,None]
    
    return Sxx_positional, test, test2
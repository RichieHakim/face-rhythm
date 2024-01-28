"""
This file contains example pipelines for running the face_rhythm package. \n
In each case, a params dictionary is input, which must contain all the necessary parameters for the pipeline. \n
"""

def pipeline_basic(params):
    """
    This function runs the basic face_rhythm pipeline, similar to the
    interactive jupyter notebook: /notebooks/interactive_pipeline_basic.ipynb \n
    Note that the ROIs must be defined and saved as an ROIs.h5 file and then
    referenced in params['ROIs']['initialize']['path_file']. \n

    This pipeline performs the following steps: \n
    - Load video data \n
    - Load ROIs from file \n
    - Track points \n
    - Compute spectrograms \n
    - Perform tensor component analysis \n
    - Save results \n

    Args:
        params (dict): 
            Dictionary of parameters. See function for required fields. \n
            Also, /scripts/params_pipeline_basic.json contains an example
            parameters file. \n

    Returns:
        (dict):
            Dictionary containing the following keys: \n
            - path_config (str): Path to the config file \n
            - path_run_info (str): Path to the run_info file \n
            - directory_project (str): Path to the project directory \n
            - SEED (int): Random seed used \n
            - params (dict): Dictionary of parameters used \n
    """

    ########################################
    ## Start
    ########################################

    import face_rhythm as fr

    from pprint import pprint
    from pathlib import Path

    import cv2

    import numpy as np

    ## RESOURCE TRACKING
    # cpu_tracker = fr.helpers.CPU_Device_Checker()
    # cpu_tracker.track_utilization(
    #     interval=0.2,
    #     path_save=str(Path(directory_save) / 'cpu_tracker.csv'),
    # )
    # gpu_tracker = fr.helpers.NVIDIA_Device_Checker(device_index=0)
    # gpu_tracker.track_utilization(
    #     interval=0.2,
    #     path_save=str(Path(directory_save) / 'gpu_tracker.csv'),
    # )

    ## Initialize paths
    fr.util.system_info(verbose=True);

    SEED = _set_random_seed(
        seed=params['project']['random_seed'],
        deterministic=params['project']['random_seed'] is not None,
    )
    use_GPU = params['project']['use_GPU']


    directory_project = str(params['project']['directory_project'])
    directory_videos  = str(params['paths_videos']['directory_videos'])

    filename_videos_strMatch = params['paths_videos']['filename_videos_strMatch']

    path_config, path_run_info, directory_project = fr.project.prepare_project(
        directory_project=directory_project,
        overwrite_config=params['project']['overwrite_config'],  ## WARNING! CHECK THIS. If True, will overwrite existing config file!
        mkdir=True,
        initialize_visualization=params['project']['initialize_visualization'],
        verbose=params['project']['verbose'],
    )
    figure_saver = fr.util.Figure_Saver(
        path_config=path_config,
        **params['figure_saver'],
    )



    ########################################
    ## Prepare video data for point tracking
    ########################################

    if 'load_videos' in params['steps']:

        paths_videos = fr.helpers.find_paths(
            dir_outer=directory_videos,
            reMatch=filename_videos_strMatch,  ## string to use to search for files in directory. Uses regular expressions!
            depth=0,  ## how many folders deep to search
        )[:]

        pprint('Paths to videos:') if params['project']['verbose'] > 1 else None
        pprint(paths_videos, width=1000) if params['project']['verbose'] > 1 else None



        ## Make a `BufferedVideoReader` object for reading video file data

        videos = fr.helpers.BufferedVideoReader(
            paths_videos=paths_videos,
            **params['BufferedVideoReader']
        )

        ## Make a `Dataset_videos` object for referencing the raw video data

        data = fr.data_importing.Dataset_videos(
            bufferedVideoReader=videos,
            **params['Dataset_videos'],
        );

        ## Save the `Dataset_videos` object in the 'analysis_files' project folder

        data.save_config(path_config=path_config, overwrite=True, verbose=1)
        data.save_run_info(path_config=path_config, overwrite=True, verbose=1)
        data.save_run_data(path_config=path_config, overwrite=True, verbose=1)



    ########################################
    ## Define ROIs
    ########################################

    if 'ROIs' in params['steps']:

        ## Either select new ROIs (`select_mode='gui'`), or import existing ROIs (`path_file=path_to_ROIs.h5_file`).\
        ## Typically, you should make 1 or 2 ROIs. One for defining where the face points should be and one for cropping the frame.

        # %matplotlib notebook
        rois = fr.rois.ROIs(**params['ROIs']['initialize'])

        rois.make_points(
            rois=rois[params['ROIs']['make_points']['rois_points_idx']],
            point_spacing=params['ROIs']['make_points']['point_spacing'],
        ) if rois.point_positions is None else None

        ## Save the `ROIs` object in the 'analysis_files' project folder

        rois.save_config(path_config=path_config, overwrite=True, verbose=1)
        rois.save_run_info(path_config=path_config, overwrite=True, verbose=1)
        rois.save_run_data(path_config=path_config, overwrite=True, verbose=1)



        # ## visualize the ROIs

        # rois.plot_masks(data[0][0])



    ########################################
    # Point Tracking
    ########################################

    if 'point_tracking' in params['steps']:

        ## Prepare `PointTracker` object.\
        ## Set `visualize_video` to **`True`** to tune parameters until they look appropriate, then set to **`False`** to run the full dataset through at a much faster speed.
        ##
        ## Key parameters:
        ## - `point_spacing`: distance between points. Vary so that total number of points is appropriate.
        ## - `mesh_rigidity`: how rigid the mesh elasticity is. Vary so that points track well without drift.
        ## - `relaxation`: how quickly the points relax back to their home position. Vary so that points track well without dift.
        ## - `kwargs_method > winSize`: the spatial size of the optical flow calculation. Smaller is better but noisier, larger is less accurate but more robust to noise.
        ## - `params_outlier_handling > threshold_displacement`: point displacements above this value will result in freezing of the points.

        pt = fr.point_tracking.PointTracker(
            buffered_video_reader=videos,
            point_positions=rois.point_positions,
            rois_masks=rois[1],
            **params['PointTracker'],
        )

        ## Perform point tracking

        pt.track_points()



        ## Save the `PointTracker` object in 'analysis_files' project directory.\
        ## Using compression can reduce file sizes slightly but is very slow.

        pt.save_config(path_config=path_config, overwrite=True, verbose=1)
        pt.save_run_info(path_config=path_config, overwrite=True, verbose=2)
        pt.save_run_data(path_config=path_config, overwrite=True, use_compression=False, verbose=1)



        ## Clear some memory if needed. Optional.

        pt.cleanup()



    ########################################
    # Spectral Analysis
    ########################################

    if 'VQT' in params['steps']:

        ## Load the `PointTracker` data as a dictionary

        pt_data = fr.h5_handling.simple_load(str(Path(directory_project) / 'analysis_files' / 'PointTracker.h5'))

        ## Prepare `VQT_Analyzer` object.
        ##
        ## Key parameters:
        ## - `Q_lowF`:  Quality of the lowest frequency band of the spectrogram. Q value is number of oscillation periods.
        ## - `Q_highF`: Quality of the highest frequency band...
        ## - `F_min`: Lowest frequency band to use.
        ## - `F_max`: Highest frequency band to use.
        ## - `downsample_factor`: How much to downsample the spectrogram by in time.
        ## - `return_complex`: Whether or not to return the complex spectrogram. Generally set to False unless you want to try something fancy.

        Fs = fr.util.load_run_info_file(path_run_info)['Dataset_videos']['frame_rate']

        params['VQT_Analyzer']['params_VQT']['Fs_sample'] = Fs
        params['VQT_Analyzer']['params_VQT']['DEVICE_compute'] = fr.helpers.set_device(use_GPU=use_GPU)

        spec = fr.spectral_analysis.VQT_Analyzer(**params['VQT_Analyzer'])


        ## Look at a demo spectrogram of a single point.\
        ## Specify the point with the `idx_point` and `name_points` fields.\
        ## Note that the `pt_data['points_tracked']` dictionary holds subdictionaries withe numeric string names (ie `['0'], ['1']`) for each video.

        # demo_sepc = spec.demo_transform(
        #     points_tracked=pt_data['points_tracked'],
        #     point_positions=pt_data['point_positions'],
        #     idx_point=30,
        #     name_points='0',
        #     plot=False,
        # );

        ## Generate spectrograms

        spec.transform_all(
            points_tracked=pt_data['points_tracked'],
            point_positions=pt_data['point_positions'],
        )

        ## Save the `VQT_Analyzer` object in 'analysis_files' project directory.\
        ## Using compression can reduce file sizes slightly but is very slow.

        spec.save_config(path_config=path_config, overwrite=True, verbose=1)
        spec.save_run_info(path_config=path_config, overwrite=True, verbose=1)
        spec.save_run_data(path_config=path_config, overwrite=True, use_compression=False, verbose=1)

        ## Clear some memory if needed. Optional.

        spec.cleanup()



    ########################################
    # Decomposition
    ########################################

    if 'TCA' in params['steps']:

        ## Load the `VQT_Analyzer` data as a dictionary

        spec_data = fr.h5_handling.simple_load(str(Path(directory_project) / 'analysis_files' / 'VQT_Analyzer.h5'))

        ## Prepare `TCA` object, and then rearrange the data with the `.rearrange_data` method.
        ##
        ## Key parameters for `.rearrange_data`:
        ## - `names_dims_array`:  Enter the names of the dimensions of the spectrogram. Typically these are `'xy', 'points', 'frequency', 'time'`.
        ## - `names_dims_concat_array`: Enter any dimensions you wish to concatenate along other dimensions. Typically we wish to concatenate the `'xy'` dimension along the `'points'` dimension, so we make a list containing that pair as a tuple: `[('xy', 'points')]`.
        ## - `concat_complexDim`: If your input data are complex valued, then this can concatenate the complex dimension along another dimension.
        ## - `name_dim_dictElements`: The `data` argument is expected to be a dictionary of dictionaries of arrays, where the inner dicts are trials or videos. This is the name of what those inner dicts are. Typically `'trials'`.

        # spectrograms = spec_data['spectrograms']
        spectrograms = {key: np.abs(val) for key,val in list(spec_data['spectrograms'].items())[:]}

        tca = fr.decomposition.TCA(
            verbose=params['TCA']['verbose'],
        )

        tca.rearrange_data(
            data=spectrograms,
            **params['TCA']['rearrange_data'],
        )



        ## Fit TCA model.
        ##
        ## There are a few methods that can be used:
        ## - `'CP_NN_HALS'`: non-negative CP decomposition using the efficient HALS algorithm. This should be used in most cases.
        ## - `'CP'`: Standard CP decomposition. Use if input data are not non-negative (if you are using complex valued spectrograms or similar).
        ## - `'Randomized_CP'`: Randomized CP decomposition. Allows for large input tensors. If you are using huge tensors and you are memory constrained or want to run on a small GPU, this is your only option.
        ##
        ## If you have and want to use a CUDA compatible GPU:
        ## - Set `DEVICE` to `'cuda'`
        ## - GPU memory can be saved by setting `'init'` method to `'random'`. However, fastest convergence and highest accuracy typically come from `'init': 'svd'`.

        tca.fit(
            DEVICE=fr.helpers.set_device(use_GPU=use_GPU),
            **params['TCA']['fit'],
        )



        ## Rearrange the factors.\
        ## You can undo the concatenation that was done during `.rearrange_data`

        tca.rearrange_factors(**params['TCA']['rearrange_factors'])



        ## Save the `TCA` object in 'analysis_files' project directory.

        tca.save_config(path_config=path_config, overwrite=True, verbose=1)
        tca.save_run_info(path_config=path_config, overwrite=True, verbose=1)
        tca.save_run_data(path_config=path_config, overwrite=True, use_compression=False, verbose=1)



        ## Clear some memory if needed. Useful if you ran the fit on a GPU. Optional.

        tca._cleanup()


        # ## Plot factors

        # tca.plot_factors(
        #     figure_saver=None,
        #     show_figures=True,
        # )



        # ## Load the `TCA` data as a dictionary

        # tca_data = fr.h5_handling.simple_load(str(Path(directory_project) / 'analysis_files' / 'TCA.h5'))



    ########################################
    # Demo playback
    ########################################

    # ## Playback a video with points overlayed.\
    # ## Make sure you have a `BufferedVideoReader` object called `videos` made of your videos

    # idx_video_to_use = 0
    # idx_frames_to_use = np.arange(0,5000)

    # videos.method_getitem = 'by_video'

    # frame_visualizer = fr.visualization.FrameVisualizer(
    #     display=True,
    #     error_checking=True,
    # #     path_save=str(Path(directory_project) / 'visualizations' / 'point_tracking_demo.avi'),
    #     path_save=None,
    #     frame_height_width=videos.frame_height_width,
    #     frame_rate=240,
    #     point_sizes=3,
    #     points_colors=(0,255,255),
    #     alpha=0.3,
    # )

    # fr.visualization.play_video_with_points(
    #     bufferedVideoReader=videos[idx_video_to_use],
    #     frameVisualizer=frame_visualizer,
    #     points=list(pt_data['points_tracked'].values())[0],
    #     idx_frames=idx_frames_to_use,
    # )



    ########################################
    # Complete messages
    ########################################

    print(f'RUN COMPLETE')

    results = {
        'path_config': path_config,
        'path_run_info': path_run_info,
        'directory_project': directory_project,
        'SEED': SEED,
        'params': params,
    }

    return results


def _set_random_seed(seed=None, deterministic=False):
    """
    Set random seed for reproducibility.
    RH 2023

    Args:
        seed (int, optional):
            Random seed.
            If None, a random seed (spanning int32 integer range) is generated.
        deterministic (bool, optional):
            Whether to make packages deterministic.

    Returns:
        (int):
            seed (int):
                Random seed.
    """
    ### random seed (note that optuna requires a random seed to be set within the pipeline)
    import numpy as np
    seed = int(np.random.randint(0, 2**31 - 1, dtype=np.uint32)) if seed is None else seed

    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import cv2
    cv2.setRNGSeed(seed)

    ## Make torch deterministic
    torch.use_deterministic_algorithms(deterministic)
    ## Make cudnn deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    
    return seed
import cv2
import numpy as np
import h5py

from face_rhythm.util import helpers, set_roi, setup
from face_rhythm.optic_flow import optic_flow, clean_results, conv_dim_reduce
from face_rhythm.analysis import pca, spectral_analysis, tca

from pathlib import Path


def test_single_session_single_video():
    # SETUP
    run_name = 'single_session_single_video'
    project_path = Path('test_runs/'+run_name).resolve()
    video_path = Path('test_data/'+run_name).resolve()
    overwrite_config = False
    remote = False
    trials = False

    config_filepath = setup.setup_project(project_path, video_path, run_name, overwrite_config, remote, trials)

    # VIDEO LOAD
    config = helpers.load_config(config_filepath)
    config['Video']['session_prefix'] = 'session'
    config['Video']['print_filenames'] = True
    config['General']['overwrite_nwbs'] = True
    helpers.save_config(config, config_filepath)

    setup.prepare_videos(config_filepath)

    # ROI Selection
    config = helpers.load_config(config_filepath)
    config['ROI']['session_to_set'] = 0  # 0 indexed. Chooses the session to use
    config['ROI']['vid_to_set'] = 0  # 0 indexed. Sets the video to use to make an image
    config['ROI']['frame_to_set'] = 1  # 0 indexed. Sets the frame number to use to make an image
    config['ROI']['load_from_file'] = True  # if you've already run this and want to use the existing ROI, set to True
    helpers.save_config(config, config_filepath)
    #special line to just grab the points
    with h5py.File(Path('test_data/pts_all.h5'), 'r') as pt:
        pts_all = helpers.h5_to_dict(pt)
    for session in config['General']['sessions']:
        helpers.save_pts(session['nwb'], pts_all)

    # Optic Flow
    config = helpers.load_config(config_filepath)
    config['Optic']['vidNums_toUse'] = [0]
    config['Optic']['spacing'] = 16
    config['Optic']['showVideo_pref'] = False
    config['Video']['printFPS_pref'] = False
    config['Video']['fps_counterPeriod'] = 10
    config['Video']['dot_size'] = 1
    config['Video']['save_demo'] = False
    config['Video']['demo_len'] = 10
    config['Optic']['lk'] = {}
    config['Optic']['lk']['winSize'] = (15, 15)
    config['Optic']['lk']['maxLevel'] = 2
    config['Optic']['lk']['criteria'] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 3, 0.001)
    config['Optic']['recursive'] = False
    config['Optic']['recursive_relaxation_factor'] = 0.005
    config['Optic']['multithread'] = False
    helpers.save_config(config, config_filepath)

    optic_flow.optic_workflow(config_filepath)

    # Clean Up
    config = helpers.load_config(config_filepath)
    config['Clean']['outlier_threshold_positions'] = 25
    config['Clean']['outlier_threshold_displacements'] = 4
    config['Clean']['framesHalted_beforeOutlier'] = 4
    config['Clean']['framesHalted_afterOutlier'] = 2
    config['Clean']['relaxation_factor'] = 0.005
    helpers.save_config(config, config_filepath)

    clean_results.clean_workflow(config_filepath)

    # ConvDR
    config = helpers.load_config(config_filepath)
    pointInds_toUse = helpers.load_data(config_filepath, 'pointInds_toUse')
    config['CDR']['width_cosKernel'] = 48
    config['CDR']['num_dots'] = pointInds_toUse.shape[0]
    config['CDR']['spacing'] = 16
    config['CDR']['display_points'] = False
    config['CDR']['vidNum'] = 0
    config['CDR']['frameNum'] = 1
    config['CDR']['dot_size'] = 1
    config['CDR']['kernel_alpha'] = 0.3
    config['CDR']['kernel_pixel'] = 10
    config['CDR']['num_components'] = 3
    helpers.save_config(config, config_filepath)

    conv_dim_reduce.conv_dim_reduce_workflow(config_filepath)

    pca.pca_workflow(config_filepath, 'positions_convDR_absolute')

    # Positional TCA
    config = helpers.load_config(config_filepath)
    config['TCA']['device'] = tca.use_gpu(False)
    config['TCA']['pref_useGPU'] = False
    config['TCA']['rank'] = 4
    config['TCA']['init'] = 'random'
    config['TCA']['tolerance'] = 1e-06
    config['TCA']['verbosity'] = 0
    config['TCA']['n_iters'] = 10
    helpers.save_config(config, config_filepath)

    tca.positional_tca_workflow(config_filepath, 'positions_convDR_meanSub')

    # CQT

    config = helpers.load_config(config_filepath)
    config['CQT']['hop_length'] = 16
    config['CQT']['fmin_rough'] = 1.8
    config['CQT']['sampling_rate'] = config['Video']['Fs']
    config['CQT']['n_bins'] = 35
    helpers.save_config(config, config_filepath)

    spectral_analysis.prepare_freqs(config_filepath)

    spectral_analysis.cqt_workflow(config_filepath, 'positions_convDR_meanSub')

    # Spectral TCA
    config = helpers.load_config(config_filepath)
    config['TCA']['device'] = tca.use_gpu(False)
    config['TCA']['pref_useGPU'] = False
    config['TCA']['rank'] = 8
    config['TCA']['init'] = 'random'
    config['TCA']['tolerance'] = 1e-06
    config['TCA']['verbosity'] = 1
    config['TCA']['n_iters'] = 10
    helpers.save_config(config, config_filepath)

    tca.full_tca_workflow(config_filepath, 'positions_convDR_meanSub')



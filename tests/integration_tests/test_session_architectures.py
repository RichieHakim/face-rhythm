import cv2
import h5py
from matplotlib import  pyplot as plt

from face_rhythm.util import helpers, set_roi, setup
from face_rhythm.optic_flow import optic_flow, clean_results, conv_dim_reduce
from face_rhythm.analysis import pca, spectral_analysis, tca
from face_rhythm.visualize import videos, plots
from face_rhythm.comparisons import facemap, hog

from pathlib import Path
import shutil


def run_basic(run_name):
    project_path = Path('test_runs').resolve() / run_name
    video_path = Path('test_data').resolve() / run_name / 'session1'
    overwrite_config = True
    remote = True
    trials = False
    multisession = False

    config_filepath = setup.setup_project(project_path, video_path, run_name, overwrite_config, remote, trials,
                                          multisession)

    # VIDEO LOAD
    config = helpers.load_config(config_filepath)
    config['Video']['file_prefix'] = 'gmou06'
    config['Video']['print_filenames'] = True
    config['General']['overwrite_nwbs'] = True
    helpers.save_config(config, config_filepath)
    setup.prepare_videos(config_filepath)

    run_downstream(config_filepath)

def config_switch(run_name):
    project_path = Path('test_runs').resolve() / run_name
    video_path = Path('test_data').resolve() / run_name / 'session1'
    overwrite_config = True
    remote = True
    trials = False
    multisession = False

    return setup.setup_project(project_path, video_path, run_name, overwrite_config, remote, trials,
                                          multisession)


def run_multi(run_name):
    project_path = Path('test_runs/' + run_name).resolve()
    video_path = Path('test_data/' + run_name).resolve()
    overwrite_config = True
    remote = True
    trials = False
    multisession = True

    config_filepath = setup.setup_project(project_path, video_path, run_name, overwrite_config, remote, trials,
                                          multisession)

    # VIDEO LOAD
    config = helpers.load_config(config_filepath)
    config['Video']['session_prefix'] = 'session'
    config['Video']['print_filenames'] = True
    config['General']['overwrite_nwbs'] = True
    helpers.save_config(config, config_filepath)
    setup.prepare_videos(config_filepath)

    run_downstream(config_filepath)

def run_downstream(config_filepath):
    # ROI Selection
    config = helpers.load_config(config_filepath)
    config['ROI']['session_to_set'] = 0  # 0 indexed. Chooses the session to use
    config['ROI']['vid_to_set'] = 0  # 0 indexed. Sets the video to use to make an image
    config['ROI']['frame_to_set'] = 1  # 0 indexed. Sets the frame number to use to make an image
    config['ROI']['load_from_file'] = True  # if you've already run this and want to use the existing ROI, set to True
    helpers.save_config(config, config_filepath)
    # special line to just grab the points
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

    # Visualize
    config = helpers.load_config(config_filepath)
    config['Video']['demo_len'] = 10
    config['Video']['data_to_display'] = 'positions_cleanup_absolute'
    config['Video']['save_demo'] = True
    helpers.save_config(config, config_filepath)

    videos.visualize_points(config_filepath)

    # ConvDR
    config = helpers.load_config(config_filepath)
    config['CDR']['width_cosKernel'] = 48
    config['CDR']['num_dots'] = config['Optic']['num_dots']
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

    # Visualize
    config = helpers.load_config(config_filepath)
    config['Video']['demo_len'] = 10
    config['Video']['data_to_display'] = 'positions_convDR_absolute'
    config['Video']['save_demo'] = True
    helpers.save_config(config, config_filepath)

    videos.visualize_points(config_filepath)

    pca.pca_workflow(config_filepath, 'positions_convDR_absolute')

    config = helpers.load_config(config_filepath)
    config['PCA']['n_factors_to_show'] = 3
    helpers.save_config(config, config_filepath)

    plots.plot_pca_diagnostics(config_filepath)
    plt.close('all')

    # Visualize PCs
    config = helpers.load_config(config_filepath)
    config['Video']['factor_category_to_display'] = 'PCA'
    config['Video']['factor_to_display'] = 'factors_points'
    config['Video']['points_to_display'] = 'positions_convDR_absolute'
    config['Video']['demo_len'] = 10
    config['Video']['dot_size'] = 2
    config['Video']['save_demo'] = True
    helpers.save_config(config, config_filepath)

    videos.visualize_factor(config_filepath)

    # Positional TCA
    config = helpers.load_config(config_filepath)
    config['TCA']['pref_useGPU'] = False
    config['TCA']['rank'] = 4
    config['TCA']['init'] = 'random'
    config['TCA']['tolerance'] = 1e-06
    config['TCA']['verbosity'] = 0
    config['TCA']['n_iters'] = 100
    helpers.save_config(config, config_filepath)

    tca.positional_tca_workflow(config_filepath, 'positions_convDR_meanSub')

    config = helpers.load_config(config_filepath)
    config['TCA']['ftype'] = 'positional'
    helpers.save_config(config, config_filepath)

    plots.plot_tca_factors(config_filepath)
    plt.close('all')

    config = helpers.load_config(config_filepath)
    config['Video']['factor_category_to_display'] = 'TCA'
    config['Video']['factor_to_display'] = 'factors_positional_points'
    config['Video']['points_to_display'] = 'positions_convDR_absolute'
    config['Video']['demo_len'] = 10
    config['Video']['dot_size'] = 2
    config['Video']['save_demo'] = True
    helpers.save_config(config, config_filepath)

    videos.visualize_factor(config_filepath)

    # CQT
    config = helpers.load_config(config_filepath)
    config['CQT']['hop_length'] = 16
    config['CQT']['fmin_rough'] = 1.8
    config['CQT']['sampling_rate'] = config['Video']['Fs']
    config['CQT']['n_bins'] = 35
    helpers.save_config(config, config_filepath)

    spectral_analysis.prepare_freqs(config_filepath)

    spectral_analysis.cqt_workflow(config_filepath, 'positions_convDR_meanSub')

    config = helpers.load_config(config_filepath)
    config['CQT']['pixelNum_toUse'] = 10
    helpers.save_config(config, config_filepath)

    plots.plot_cqt(config_filepath)
    plt.close('all')

    # Spectral TCA
    config = helpers.load_config(config_filepath)
    config['TCA']['pref_useGPU'] = False
    config['TCA']['rank'] = 8
    config['TCA']['init'] = 'random'
    config['TCA']['tolerance'] = 1e-06
    config['TCA']['verbosity'] = 0
    config['TCA']['n_iters'] = 100
    helpers.save_config(config, config_filepath)

    tca.full_tca_workflow(config_filepath, 'positions_convDR_meanSub')

    config = helpers.load_config(config_filepath)
    config['TCA']['ftype'] = 'spectral'
    helpers.save_config(config, config_filepath)

    plots.plot_tca_factors(config_filepath)
    plt.close('all')

    config = helpers.load_config(config_filepath)
    config['Video']['factor_category_to_display'] = 'TCA'
    config['Video']['factor_to_display'] = 'factors_spectral_points'
    config['Video']['points_to_display'] = 'positions_convDR_absolute'
    config['Video']['demo_len'] = 10
    config['Video']['dot_size'] = 2
    config['Video']['save_demo'] = True
    helpers.save_config(config, config_filepath)

    videos.visualize_factor(config_filepath)

    config = helpers.load_config(config_filepath)
    config['Video']['factor_category_to_display'] = 'TCA'
    config['Video']['factor_to_display'] = 'factors_spectral_points'
    config['Video']['points_to_display'] = 'positions_convDR_absolute'
    config['Video']['start_vid'] = 0
    config['Video']['start_frame'] = 0
    config['Video']['demo_len'] = 10
    config['Video']['dot_size'] = 2
    config['Video']['save_demo'] = True
    config['Video']['factors_to_show'] = []
    config['Video']['show_alpha'] = True
    config['Video']['pulse_test_index'] = 0
    helpers.save_config(config, config_filepath)

    videos.face_with_trace(config_filepath)
    plt.close('all')

    # Comparisons
    config = helpers.load_config(config_filepath)
    config['Comps'] = {}
    config['Comps']['sbin'] = 4
    config['Comps']['ncomps'] = 100
    helpers.save_config(config, config_filepath)

    facemap.facemap_workflow(config_filepath)

    config = helpers.load_config(config_filepath)
    config['Comps']['sbin'] = 4
    config['Comps']['cell_size'] = 8
    helpers.save_config(config, config_filepath)

    hog.hog_workflow(config_filepath)

    # Cleanup
    shutil.rmtree(config['Paths']['project'])


def test_single_session_single_video():
    run_name = 'single_session_single_video'
    run_multi(run_name)


def test_single_session_multi_video():
    run_name = 'single_session_multi_video'
    run_multi(run_name)


def test_multi_session_single_video():
    run_name = 'multi_session_single_video'
    run_multi(run_name)


def test_multi_session_multi_video():
    run_name = 'multi_session_multi_video'
    run_multi(run_name)


def test_basic_single_video():
    run_name = 'single_session_single_video'
    run_basic(run_name)


def test_basic_multi_video():
    run_name = 'single_session_multi_video'
    run_basic(run_name)

def test_config_update():
    run_name = 'single_session_single_video'
    config_filepath = config_switch(run_name)
    config = helpers.load_config(config_filepath)
    old_project_path = config['Paths']['project']
    new_project_path = str(Path(old_project_path).parent / 'test')
    shutil.copytree(old_project_path, new_project_path)
    config_filepath = helpers.update_config(new_project_path, run_name)

    config = helpers.load_config(config_filepath)
    config['Video']['file_prefix'] = 'gmou06'
    config['Video']['print_filenames'] = True
    config['General']['overwrite_nwbs'] = True
    helpers.save_config(config, config_filepath)
    setup.prepare_videos(config_filepath)

    run_downstream(config_filepath)

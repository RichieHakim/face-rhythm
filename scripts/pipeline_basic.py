
#############################
## DEMO PARAMETERS
#############################

# params = {
#     'project': {
#         'directory_project': '/media/rich/bigSSD/analysis_data/face_rhythm/demo_faceRhythm_svoboda/fr_run_20221013_new_script1/',
#         'overwrite_config': False,
#         'initialize_visualization': False,
#         'verbose': 2,
#     },
#     'figure_saver': {
#         'format_save': ['png'],
#         'kwargs_savefig': {'bbox_inches': 'tight', 'pad_inches': 0.1, 'transparent': True, 'dpi': 300},
#         'overwrite': True,
#         'verbose': 2,
#     },    
#     'paths_videos': {
#         'directory_videos': '/media/rich/bigSSD/other lab data/Svoboda_lab/BCI34_2022-07-19/side/2022-07-19_13-34-06',
#         'filename_videos_strMatch': 'trial_.*mp4',  ## You can use regular expressions to search and match more complex strings
#         'depth': 2,  ## How many folders deep to search directory
#     },
#     'BufferedVideoReader': {
#         'buffer_size': 1000,
#         'prefetch': 1,
#         'posthold': 1,
#         'method_getitem': 'by_video',
#         'verbose': 1,
#     },
#     'Dataset_videos': {
#         'contiguous': False,
#         'frame_rate_clamp': 240,
#         'verbose': 2,
#     },
#     'ROIs': {
#         'select_mode': 'file',
#         'path_file': '/media/rich/bigSSD/analysis_data/face_rhythm/demo_faceRhythm_svoboda/fr_run_20221013_new_2/analysis_files/ROIs.h5',
#         'verbose': 2,
#     },
#     'PointTracker': {
#         'rois_points_idx': [0],
#         'rois_masks_idx': [1],
#         'contiguous': False,
#         'params_optical_flow': {
#             'method': 'lucas_kanade',
#             'point_spacing': 12,
#             'mesh_rigidity': 0.01,
#             'mesh_n_neighbors': 15,
#             'relaxation': 0.001,
#             'kwargs_method': {
#                 'winSize': [20,20],
#                 'maxLevel': 2,
#                 'criteria': [2, 0.03],
#             },
#         },
#         'visualize_video': False,
#         'params_visualization': {
#             'alpha': 0.2,
#             'point_sizes': 2,
#             'writer_cv2': None,
#         },
#         'params_outlier_handling': {
#             'threshold_displacement': 80,  ## Maximum displacement between frames, in pixels.
#             'framesHalted_before': 30,  ## Number of frames to halt tracking before a violation.
#             'framesHalted_after': 30,  ## Number of frames to halt tracking after a violation.
#         },
#         'verbose': 2,
#     },
#     'VQT_Analyzer': {
#         'params_VQT': {
#             'Fs_sample': 240, 
#             'Q_lowF': 2, 
#             'Q_highF': 8, 
#             'F_min': 1, 
#             'F_max': 30, 
#             'n_freq_bins': 40, 
#             'win_size': 901, 
#             'plot_pref': False, 
#             'downsample_factor': 20, 
#             'DEVICE_compute': 'cuda:0', 
#             'batch_size': 1000,
#             'return_complex': False, 
#             'progressBar': True
#         },
#         'normalization_factor': 0.95,
#         'spectrogram_exponent': 1.0,
#         'one_over_f_exponent': 0.5,
#         'verbose': 2,
#     },
#     'TCA': {
#         'verbose': 2,
#         'rearrange_data':{
#             'names_dims_array': ['xy', 'points', 'frequency', 'time'],
#             'names_dims_concat_array': [['xy', 'points']],
#             'concat_complexDim': False,
#             'name_dim_concat_complexDim': 'time',
#             'name_dim_dictElements': 'trials',
#             'method_handling_dictElements': 'concatenate',
#             'name_dim_concat_dictElements': 'time',
#             'idx_windows': None,
#             'name_dim_array_window': 'time',
#         },
#         'fit': {
#             'method': 'CP_NN_HALS',
#         #     method='CP',
#             'params_method': {
#                 'rank': 12, 
#                 'n_iter_max': 1000, 
#                 'init': 'random', 
#                 'svd': 'truncated_svd', 
#                 'tol': 1e-09, 
#         #         'nn_modes': [0,1], 
#                 'verbose': True, 
#             },
#             'DEVICE': 'cuda:0',
#             'verbose': 2,
#         },
#         'rearrange_factors': {
#             'undo_concat_complexDim': False,
#             'undo_concat_dictElements': True,
#         },
#     },
# }



########################################
## Import parameters from CLI
########################################

import os
print(f"script environment: {os.environ['CONDA_DEFAULT_ENV']}")


## Argparse --path_params, --directory_save
import argparse
parser = argparse.ArgumentParser(
    prog='Face-Rhythm Basic Pipeline',
    description='This script runs the basic pipeline using a json file containing the parameters.',
)
parser.add_argument(
    '--path_params',
    '-p',
    required=True,
    metavar='',
    type=str,
    default=None,
    help='Path to json file containing parameters.',
)
parser.add_argument(
    '--directory_save',
    '-d',
    required=False,
    metavar='',
    type=str,
    default=None,
    help="Directory to use as 'directory_project' and save results to. Overrides 'directory_project' field in parameters file.",
)
args = parser.parse_args()
path_params = args.path_params
directory_save = args.directory_save



## Checks for path_params and directory_save
from pathlib import Path

## Check path_params
### Check if path_params is valid
assert Path(path_params).exists(), f"Path to parameters file does not exist: {path_params}"
### Check if path is absolute. If not, convert to absolute path.
if not Path(path_params).is_absolute():
    path_params = Path(path_params).resolve()
    print(f"Warning: Input path_params is not absolute. Converted to absolute path: {path_params}")
### Warn if suffix is not json
print(f"Warning: suffix of path_params is not .json: {path_params}") if Path(path_params).suffix != '.json' else None
print(f"path_params: {path_params}")

## Check directory_save
### Check if directory_save is valid
if args.directory_save is not None:
    assert Path(args.directory_save).exists(), f"Path to directory_save does not exist: {args.directory_save}"
    ### Check if directory_save is absolute. If not, convert to absolute path.
    if not Path(args.directory_save).is_absolute():
        args.directory_save = Path(args.directory_save).resolve()
        print(f"Warning: Input directory_save is not absolute. Converted to absolute path: {args.directory_save}")
    ### Check that directory_save is a directory
    assert Path(args.directory_save).is_dir(), f"Input directory_save is not a directory: {args.directory_save}"
    ### Set directory_save
    print(f"directory_save: {directory_save}")

## Load parameters
import json
with open(path_params, 'r') as f:
    params = json.load(f)



########################################
## Start script
########################################

import face_rhythm as fr

from pprint import pprint
from pathlib import Path
import time

import cv2

import numpy as np

tic_start = time.time()

fr.util.get_system_versions(verbose=True);

directory_project = params['project']['directory_project'] if directory_save is None else directory_save
directory_videos  = params['paths_videos']['directory_videos']

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
    format_save=params['figure_saver']['format_save'],
    kwargs_savefig=params['figure_saver']['kwargs_savefig'],
    overwrite=params['figure_saver']['overwrite'],
    verbose=params['figure_saver']['verbose'],
)



########################################
## Prepare video data for point tracking
########################################

paths_videos = fr.helpers.find_paths(
    dir_outer=directory_videos,
    reMatch=filename_videos_strMatch,  ## string to use to search for files in directory. Uses regular expressions!
    depth=0,  ## how many folders deep to search
)[:3]

pprint('Paths to videos:') if params['project']['verbose'] > 1 else None
pprint(paths_videos, width=1000) if params['project']['verbose'] > 1 else None



## Make a `BufferedVideoReader` object for reading video file data

videos = fr.helpers.BufferedVideoReader(
#     video_readers=data.videos, 
    paths_videos=paths_videos,
    buffer_size=params['BufferedVideoReader']['buffer_size'],
    prefetch=params['BufferedVideoReader']['prefetch'],
    posthold=params['BufferedVideoReader']['posthold'],
    method_getitem=params['BufferedVideoReader']['method_getitem'],
    verbose=params['BufferedVideoReader']['verbose'],
)



## Make a `Dataset_videos` object for referencing the raw video data

data = fr.data_importing.Dataset_videos(
    bufferedVideoReader=videos,
#     paths_videos=paths_videos,
    contiguous=params['Dataset_videos']['contiguous'],
    frame_rate_clamp=params['Dataset_videos']['frame_rate_clamp'],
    verbose=params['Dataset_videos']['verbose'],
);



## Save the `Dataset_videos` object in the 'analysis_files' project folder

data.save_config(path_config=path_config, overwrite=True, verbose=1)
data.save_run_info(path_config=path_config, overwrite=True, verbose=1)
data.save_run_data(path_config=path_config, overwrite=True, verbose=1)



########################################
## Define ROIs
########################################

## Either select new ROIs (`select_mode='gui'`), or import existing ROIs (`path_file=path_to_ROIs.h5_file`).\
## Typically, you should make 1 or 2 ROIs. One for defining where the face points should be and one for cropping the frame.

# %matplotlib notebook
rois = fr.rois.ROIs(
#     select_mode='gui',
#     exampleImage=data[0][0],
    select_mode=params['ROIs']['select_mode'],
    path_file=params['ROIs']['path_file'],
    verbose=params['ROIs']['verbose'],
)



## Save the `ROIs` object in the 'analysis_files' project folder

rois.save_config(path_config=path_config, overwrite=True, verbose=1)
rois.save_run_info(path_config=path_config, overwrite=True, verbose=1)
rois.save_run_data(path_config=path_config, overwrite=True, verbose=1)



# ## visualize the ROIs

# rois.plot_masks(data[0][0])



########################################
# Point Tracking
########################################

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
#     buffered_video_reader=videos[:5],
    buffered_video_reader=videos,
    rois_points=[rois[ii] for ii in params['PointTracker']['rois_points_idx']],
    rois_masks=[rois[ii] for ii in params['PointTracker']['rois_masks_idx']],
    contiguous=params['PointTracker']['contiguous'],
    params_optical_flow={
        "method": params['PointTracker']['params_optical_flow']['method'],
        "point_spacing": params['PointTracker']['params_optical_flow']['point_spacing'],
        "mesh_rigidity": params['PointTracker']['params_optical_flow']['mesh_rigidity'],
        "mesh_n_neighbors": params['PointTracker']['params_optical_flow']['mesh_n_neighbors'],
        "relaxation": params['PointTracker']['params_optical_flow']['relaxation'],
        "kwargs_method": {
            "winSize": params['PointTracker']['params_optical_flow']['kwargs_method']['winSize'],
            "maxLevel": params['PointTracker']['params_optical_flow']['kwargs_method']['maxLevel'],
            "criteria": tuple([cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT] + list(params['PointTracker']['params_optical_flow']['kwargs_method']['criteria'])),
        },        
    },
    visualize_video=params['PointTracker']['visualize_video'],
    params_visualization={
                'alpha': params['PointTracker']['params_visualization']['alpha'],
                'point_sizes': params['PointTracker']['params_visualization']['point_sizes'],
                'writer_cv2': params['PointTracker']['params_visualization']['writer_cv2'],
    },
    params_outlier_handling = {
        'threshold_displacement': params['PointTracker']['params_outlier_handling']['threshold_displacement'],
        'framesHalted_before': params['PointTracker']['params_outlier_handling']['framesHalted_before'],
        'framesHalted_after': params['PointTracker']['params_outlier_handling']['framesHalted_after'],
    },
    verbose=params['PointTracker']['verbose'],
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



## Load the `PointTracker` data as a dictionary

pt_data = fr.h5_handling.simple_load(str(Path(directory_project) / 'analysis_files' / 'PointTracker.h5'))
pt_data.unlazy()



########################################
# Spectral Analysis
########################################

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

spec = fr.spectral_analysis.VQT_Analyzer(
    params_VQT={
        'Fs_sample': params['VQT_Analyzer']['params_VQT']['Fs_sample'],
        'Q_lowF': params['VQT_Analyzer']['params_VQT']['Q_lowF'],
        'Q_highF': params['VQT_Analyzer']['params_VQT']['Q_highF'],
        'F_min': params['VQT_Analyzer']['params_VQT']['F_min'],
        'F_max': params['VQT_Analyzer']['params_VQT']['F_max'],
        'n_freq_bins': params['VQT_Analyzer']['params_VQT']['n_freq_bins'],
        'win_size': params['VQT_Analyzer']['params_VQT']['win_size'],
        'plot_pref': params['VQT_Analyzer']['params_VQT']['plot_pref'],
        'downsample_factor': params['VQT_Analyzer']['params_VQT']['downsample_factor'],
        'DEVICE_compute': params['VQT_Analyzer']['params_VQT']['DEVICE_compute'],
        'batch_size': params['VQT_Analyzer']['params_VQT']['batch_size'],
        'return_complex': params['VQT_Analyzer']['params_VQT']['return_complex'],
        'progressBar': params['VQT_Analyzer']['params_VQT']['progressBar'],
    },
    normalization_factor=params['VQT_Analyzer']['normalization_factor'],
    spectrogram_exponent=params['VQT_Analyzer']['spectrogram_exponent'],
    one_over_f_exponent=params['VQT_Analyzer']['one_over_f_exponent'],
    verbose=params['VQT_Analyzer']['verbose'],
)



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



## Load the `VQT_Analyzer` data as a dictionary

spec_data = fr.h5_handling.simple_load(str(Path(directory_project) / 'analysis_files' / 'VQT_Analyzer.h5'))
spec_data.unlazy()



########################################
# Decomposition
########################################

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
    names_dims_array=params['TCA']['rearrange_data']['names_dims_array'],
    names_dims_concat_array=params['TCA']['rearrange_data']['names_dims_concat_array'],
    concat_complexDim=params['TCA']['rearrange_data']['concat_complexDim'],
    name_dim_concat_complexDim=params['TCA']['rearrange_data']['name_dim_concat_complexDim'],
    name_dim_dictElements=params['TCA']['rearrange_data']['name_dim_dictElements'],
    method_handling_dictElements=params['TCA']['rearrange_data']['method_handling_dictElements'],
    name_dim_concat_dictElements=params['TCA']['rearrange_data']['name_dim_concat_dictElements'],
    idx_windows=params['TCA']['rearrange_data']['idx_windows'],
    name_dim_array_window=params['TCA']['rearrange_data']['name_dim_array_window'],
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
    method=params['TCA']['fit']['method'],
    params_method={
        'rank': params['TCA']['fit']['params_method']['rank'],
        'n_iter_max': params['TCA']['fit']['params_method']['n_iter_max'],
        'init': params['TCA']['fit']['params_method']['init'],
        'svd': params['TCA']['fit']['params_method']['svd'],
        'tol': params['TCA']['fit']['params_method']['tol'],
        'verbose': params['TCA']['fit']['params_method']['verbose'],
    },
    DEVICE=params['TCA']['fit']['DEVICE'],
    verbose=params['TCA']['fit']['verbose'],
)



## Rearrange the factors.\
## You can undo the concatenation that was done during `.rearrange_data`

tca.rearrange_factors(
    undo_concat_complexDim=params['TCA']['rearrange_factors']['undo_concat_complexDim'],
    undo_concat_dictElements=params['TCA']['rearrange_factors']['undo_concat_dictElements'],
)



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



## Load the `TCA` data as a dictionary

tca_data = fr.h5_handling.simple_load(str(Path(directory_project) / 'analysis_files' / 'TCA.h5'))
tca_data.unlazy()



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
print(f'Project directory: {directory_project}')
print(f'Time elapsed: {time.time() - tic_start:.2f} seconds')
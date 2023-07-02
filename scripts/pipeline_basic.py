
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

## RESOURCE TRACKING
cpu_tracker = fr.helpers.CPU_Device_Checker()
cpu_tracker.track_utilization(
    interval=0.2,
    path_save=str(Path(directory_save) / 'cpu_tracker.csv'),
)
gpu_tracker = fr.helpers.NVIDIA_Device_Checker()
gpu_tracker.track_utilization(
    interval=0.2,
    path_save=str(Path(directory_save) / 'gpu_tracker.csv'),
)

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
    **params['figure_saver'],
)



########################################
## Prepare video data for point tracking
########################################

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



## Load the `PointTracker` data as a dictionary

pt_data = fr.h5_handling.simple_load(str(Path(directory_project) / 'analysis_files' / 'PointTracker.h5'))


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

params['VQT_Analyzer']['params_VQT']['Fs_sample'] = Fs
params['VQT_Analyzer']['params_VQT']['DEVICE_compute'] = fr.helpers.set_device(use_GPU=True)

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



## Load the `VQT_Analyzer` data as a dictionary

spec_data = fr.h5_handling.simple_load(str(Path(directory_project) / 'analysis_files' / 'VQT_Analyzer.h5'))



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
    DEVICE=fr.helpers.set_device(use_GPU=True),
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



## Load the `TCA` data as a dictionary

tca_data = fr.h5_handling.simple_load(str(Path(directory_project) / 'analysis_files' / 'TCA.h5'))



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

cpu_tracker.stop_tracking()
gpu_tracker.stop_tracking()
"""
Expects inputs deriving from a partial run of the main face-rhythm notebook. 

Expects the following inputs:
1. path to a .nwb file containing the initial dot coordinates
2. path to videos
3. path to a reference config file containing all the desired settings
4. path to output directory

- The reference config file can/should come from a local run of face-rhythm
- The .nwb file should come from a partial run of face-rhythm


This script will:
1. make a new config fig with the paths taken from the old config file for the following: .nwb, videos, config file, and outputs
"""




# path_configTemplate = '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/face_rhythm_20220513_movie3/configs/config_run.yaml'
# path_oldNWB         = '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/face_rhythm_20220513_movie3/data/sessionrun.nwb'

# dir_FRproject       = '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/batchRun'
# dir_videos          = '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/camera/'
# fileName_strMatch   = 'movie3'

import os
print(f"script environment: {os.environ['CONDA_DEFAULT_ENV']}")

import sys
path_script, path_params, dir_save = sys.argv

import json
with open(path_params, 'r') as f:
    params = json.load(f)

import shutil
from pathlib import Path
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));

# params_template = {
#     'dir_face_rhythm': dir_github,
#     'dir_videos': dir_videos,  ## directory containing the video(s) matching the fileName_strMatch
#     'fileName_strMatch' : 'movie3',
#     'name_FRproject': 'batchRun',

#     'path_configTemplate' : '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/face_rhythm_20220513_movie3/configs/config_run.yaml',
#     'path_oldNWB'         : '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/face_rhythm_20220513_movie3/data/sessionrun.nwb',
# }

dir_FRproject = str(Path(dir_save) / params['name_FRproject'])
path_configTemplate = params['path_configTemplate']
dir_videos = params['dir_videos']
fileName_strMatch = params['fileName_strMatch']
path_oldNWB = params['path_oldNWB']


import sys
dir_face_rhythm = params['dir_face_rhythm']
sys.path.append(dir_face_rhythm)

# from face_rhythm import ca2p_preprocessing, path_helpers, similarity, pickle_helpers, misc, featurization


# path_self, dir_FRproject, dir_videos, fileName_strMatch, path_oldNWB, path_configTemplate = sys.argv
# path_dispatcher_remote, dir_saveOutputs, path_script_remote, name_job, name_slurm, dir_videos = sys.argv

nameRun = '_batch'


def load_configFile(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)

def dump_nwb(nwb_path):
    """
    Print out nwb contents

    Args:
        nwb_path (str): path to the nwb file

    Returns:
    """
    import pynwb
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        for interface in nwbfile.processing['Face Rhythm'].data_interfaces:
            print(interface)
            time_series_list = list(nwbfile.processing['Face Rhythm'][interface].time_series.keys())
            for ii, time_series in enumerate(time_series_list):
                data_tmp = nwbfile.processing['Face Rhythm'][interface][time_series].data
                print(f"     {time_series}:    {data_tmp.shape}   ,  {data_tmp.dtype}   ,   {round((data_tmp.size * data_tmp.dtype.itemsize)/1000000000, 6)} GB")



path_configNew = str(Path(dir_FRproject) / 'configs' / ('config_'+nameRun+'.yaml'))

configTemplate = load_configFile(path_configTemplate)



from face_rhythm.util_old import helpers, setup

config_filepath = setup.setup_project(
    Path(dir_FRproject),
    Path(dir_videos),
    nameRun,
    overwrite_config=False,
    remote=True,
    trials=configTemplate['General']['trials'],
    multisession=configTemplate['General']['multisession'],
    update_paths=True
)


from face_rhythm.util_old import helpers, setup

config = helpers.load_config(config_filepath)
config['Video']['file_strMatch'] = fileName_strMatch # Set to '' to grab all vids in video_path. Set to 'session_prefix' if multisession.
config['Video']['sort_filenames']  = configTemplate['Video']['sort_filenames']
config['Video']['print_filenames'] = configTemplate['Video']['print_filenames']
config['General']['overwrite_nwbs'] = False
helpers.save_config(config, config_filepath)

setup.prepare_videos(config_filepath)


from face_rhythm.util_old import helpers, set_roi

config = helpers.load_config(config_filepath)
config['ROI']['session_to_set'] = 0 # 0 indexed. Chooses the session to use
config['ROI']['vid_to_set'] = 0 # 0 indexed. Sets the video to use to make an image
config['ROI']['frame_to_set'] = 1 # 0 indexed. Sets the frame number to use to make an image

config['ROI']['load_from_file'] = True # if you want to use the ROI from a previous session (different .nwb file), set to True and define path below
config['ROI']['path_to_oldNWB'] = path_oldNWB # if 'load_from_file' is true, define path to that .nwb file here
helpers.save_config(config, config_filepath)

frame, pts_all = set_roi.get_roi(config_filepath)

# Don't run this until you're done selecting
set_roi.save_roi(config_filepath, frame, pts_all)


from face_rhythm.optic_flow import optic_flow

config = helpers.load_config(config_filepath)

config['Optic']['vidNums_toUse'] = list(range(config['General']['sessions'][0]['num_vids'])) ## 0 indexing. Use this line of code to run all the videos in a particular session

config['Optic'] = configTemplate['Optic']
# config['Optic']['spacing'] = 99 ## This is the distance between points in the grid (both in x and y dims)
config['Optic']['showVideo_pref'] = True ## USE THIS TO TUNE PARAMETERS! Much faster when video is off. If 'remote' option chosen (from first cell block), video will be saved as file in project folder.


config['Video']['dot_size'] = 4 # for viewing purposes
config['Video']['save_video'] = False # Whether to save the demo video (true for remote users when showvideo is true)
config['Video']['demo_len'] = 1000 # used when remote users when show_video==True
config['Video']['fps_counterPeriod'] = 10 # number of frames between fps averaging calculation
config['Video']['printFPS_pref'] = False # option for whether fps should be printed in notebook
config['Video']['frames_to_ignore_pref'] = False # optional. If True, then a 'frames_to_ignore.npy' file must be in the video path. It must contain a boolean array of same length as the video


helpers.save_config(config, config_filepath)

optic_flow.optic_workflow(config_filepath)


from face_rhythm.optic_flow import clean_results

config = helpers.load_config(config_filepath)
config['Clean'] = configTemplate['Clean']
helpers.save_config(config, config_filepath)

clean_results.clean_workflow(config_filepath)


from face_rhythm.optic_flow import conv_dim_reduce

config = helpers.load_config(config_filepath)
config['CDR'] = configTemplate['CDR']
config['CDR']['num_dots'] = config['Optic']['num_dots']
config['CDR']['display_points'] = False # checkout the dots and overlayed filter

helpers.save_config(config, config_filepath)

conv_dim_reduce.conv_dim_reduce_workflow(config_filepath)

helpers.save_config(config, config_filepath)


from face_rhythm.analysis import spectral_analysis

config = helpers.load_config(config_filepath)
config['CQT'] = configTemplate['CQT']

helpers.save_config(config, config_filepath)

spectral_analysis.prepare_freqs(config_filepath)

spectral_analysis.show_demo_spectrogram(config_filepath,
                        dot_toUse=500,
                        xy_toUse='x',
                        timeSeries_toUse='positions_convDR_meanSub',
                        dtype_to_estimate=config['CQT']['dtype_toUse']
)


spectral_analysis.vqt_workflow(config_filepath, 
                               data_key='positions_convDR_meanSub',
                               multicore_pref=True,
                              )
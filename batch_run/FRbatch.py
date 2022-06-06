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


from pathlib import Path


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
# shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));

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
config_filepath = path_configNew


from face_rhythm.util import helpers
from face_rhythm.analysis import spectral_analysis

configTemplate = load_configFile(path_configTemplate)

config = helpers.load_config(config_filepath)
config['CQT'] = configTemplate['CQT']
print(config_filepath)
dump_nwb(config_filepath)
helpers.save_config(config, config_filepath)

spectral_analysis.vqt_workflow(config_filepath, 
                               data_key='positions_convDR_meanSub',
                               multicore_pref=True,
                              )
import os
print(f"script environment: {os.environ['CONDA_DEFAULT_ENV']}")

import sys
path_script, path_params, dir_save = sys.argv

import json
with open(path_params, 'r') as f:
    params = json.load(f)

import shutil
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));


import pynwb
import numpy as np

from pathlib import Path
import time

# params = {
#     'path_FRNWB': '/n/data1/hms/neurobio/sabatini/rich/analysis/faceRhythm/AEG21/2022_05_13/jobNum_0/batchRun/data/session_batch.nwb',
#     'fields_toSave': ['Sxx_allPixels', 'pts_spaced_convDR'],
#     'verbose': True,
# }

# dir_save = '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG22/2022_05_16/faceRhythm_npy'
# fields_toSave = ['Sxx_allPixels', 'pts_spaced_convDR']
# # fields_toSave = None
# verbose=True

with pynwb.NWBHDF5IO(params['path_FRNWB'], 'r') as io:
    nwbfile = io.read()
    keys_outer = nwbfile.fields['processing']['Face Rhythm'].data_interfaces.keys()
    for key_outer in keys_outer:
        keys_inner = list(nwbfile.fields['processing']['Face Rhythm'].data_interfaces[key_outer].time_series.keys())
        
        if params['fields_toSave'] is not None:
            keys_inner = list(np.array(keys_inner)[np.isin(np.array(keys_inner), np.array(params['fields_toSave']))])
        for key_inner in keys_inner:
            data = nwbfile.fields['processing']['Face Rhythm'].data_interfaces[key_outer].time_series[key_inner].data[:]
            
            path_save = str(Path(dir_save) / key_outer / (key_inner+'.npy'))
            Path(path_save).parent.mkdir(parents=True, exist_ok=True)
            
            if params['verbose']:
                print(f"saving: {key_outer}  >  {key_inner}  to  {path_save}  time: {time.ctime()}")
            np.save(path_save, data)
if params['verbose']:
    print(f'saving complete. time: {time.ctime()}')
    

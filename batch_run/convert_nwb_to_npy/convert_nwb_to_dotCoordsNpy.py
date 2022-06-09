# # ALWAYS RUN THIS CELL
# # widen jupyter notebook window
# from IPython.display import display, HTML
# display(HTML("<style>.container {width:95% !important; }</style>"))

# dir_github = '/media/rich/Home_Linux_partition/github_repos/'
# import sys
# sys.path.append(dir_github)

# %load_ext autoreload
# %autoreload 2
# from basic_neural_processing_modules import *

import numpy as np
import matplotlib.pyplot as plt
import pynwb

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


import sys
path_self, path_nwb, path_save = sys.argv
# path_nwb = '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/face_rhythm_20220513_movie3/data/sessionrun.nwb'
# path_save = '/media/rich/bigSSD/analysis_data/face_rhythm_paper/fig_4/2pRAM_motor_mapping/AEG21/2022_05_13/face_rhythm_20220513_movie3/data/dot_coords.npy'

with pynwb.NWBHDF5IO(path_nwb, 'r') as io:
    nwbfile = io.read()
    dot_coords = nwbfile.processing['Face Rhythm']['Optic Flow']['pointInds_toUse'].data[:].squeeze()

dump_nwb(path_nwb)

np.save(path_save, dot_coords)
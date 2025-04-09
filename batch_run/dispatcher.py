## Import general libraries
from pathlib import Path
import os
import sys
import copy

print(f"dispatcher environment: {os.environ['CONDA_DEFAULT_ENV']}")

from face_rhythm import util

path_self, \
path_script, \
dir_save, \
dir_videos, \
path_ROIs, \
name_job, \
name_slurm, \
name_env, \
name_wandb, \
wandb_api_key, \
    = sys.argv


# date = '20221011'

# path_script = f'/n/data1/hms/neurobio/sabatini/rich/github_repos/face-rhythm/scripts/run_pipeline_basic.py'
# dir_save = f'/n/data1/hms/neurobio/sabatini/rich/analysis/faceRhythm/{mouse}/run_20230701/'
# name_job = f'faceRhythm_{date}_'
# name_slurm = f'rh_{date}'
# name_env = f'/n/data1/hms/neurobio/sabatini/rich/virtual_envs/FR'

## set paths
Path(dir_save).mkdir(parents=True, exist_ok=True)


params_template = {
    "kwargs_wandb_init": {
        "project": "face_rhythm",
        "name": name_wandb,
        "dir_save": dir_save,
        "reinit": True,
    },
    "path_script": path_script,
    
    "params_script": {
        "steps": [
            # "load_videos",
            # "ROIs",
            # "point_tracking",
            "VQT",
            "TCA",
        ],
        "project": {
            "overwrite_config": True,
            "update_project_paths": True,
            "initialize_visualization": False,
            "use_GPU": True,
            "random_seed": 0,
            "verbose": 2,
        },
        "figure_saver": {
            "formats_save": [
                "png"
            ],
            "kwargs_savefig": {
                "bbox_inches": "tight",
                "pad_inches": 0.1,
                "transparent": True,
                "dpi": 300,
            },
            "overwrite": True,
            "verbose": 2
        },
        "paths_videos": {
            "directory_videos": dir_videos,
            "filename_videos_strMatch": "video1.*avi",
            # "filename_videos_strMatch": "test\.avi",
            "depth": 2,
        },
        "BufferedVideoReader": {
            "buffer_size": 1000,
            "prefetch": 1,
            "posthold": 1,
            "method_getitem": "by_video",
            "verbose": 1,
        },
        "Dataset_videos": {
            "contiguous": False,
            "frame_rate_clamp": None,
            "verbose": 2,
        },
        "ROIs": {
            "initialize":{
                "select_mode": "file",
                "path_file": path_ROIs,
                "verbose": 2,
            },
            "make_rois": {
                "rois_points_idx": [
                    0,
                ],
                "point_spacing": 10,
            },
        },
        "PointTracker": {
            "contiguous": True,
            "params_optical_flow": {
                "method": "lucas_kanade",
                "mesh_rigidity": 0.005,
                "mesh_n_neighbors": 80,
                "relaxation": 0.001,
                "kwargs_method": {
                    "winSize": [
                        80,
                        80,
                    ],
                    "maxLevel": 5,
                    "criteria": [
                        3, ## leave as 3
                        2,
                        0.0003,
                    ],
                },
            },
            "params_clahe": {
                "clipLimit": 40.0,
                "tileGridSize": [
                    120,
                    120,
                ],
            },
            "visualize_video": False,
            "params_visualization": {
                "alpha": 0.2,
                "point_sizes": 2,
            },
            "params_outlier_handling": {
                "threshold_displacement": 250,
                "framesHalted_before": 20,
                "framesHalted_after": 20,
            },
            "idx_start": 0,
            "verbose": 2,
        },
        "VQT_Analyzer": {
            "params_VQT": {
                'Fs_sample': 250,
                'Q_lowF': 3.0,
                'Q_highF': 10.0,
                'F_min': 0.5,
                'F_max': 60,
                'n_freq_bins': 30,
                'window_type': 'hann',
                'symmetry': 'center',
                'taper_asymmetric': True,
                'downsample_factor': 20,
                'padding': 'valid',
                'fft_conv': True,
                'fast_length': True,
                'take_abs': True,
                'filters': None, 
                'plot_pref': True,
            },
            "batch_size": 10,
            "normalization_factor": 0.6,
            "spectrogram_exponent": 1.0,
            "one_over_f_exponent": 0.5,
            "verbose": 2
        },
        "TCA": {
            "verbose": 2,
            "rearrange_data": {
                "names_dims_array": [
                    "xy",
                    "points",
                    "frequency",
                    "time",
                ],
                "names_dims_concat_array": [
                    [
                        "xy",
                        "points",
                    ]
                ],
                "concat_complexDim": False,
                "name_dim_concat_complexDim": "time",
                "name_dim_dictElements": "session",
                "method_handling_dictElements": "separate",
                "name_dim_concat_dictElements": "time",
                "idx_windows": None,
                "name_dim_array_window": "time",
            },
            "fit": {
                "method": "CP_NN_HALS",
                "params_method": {
                    "rank": 10,
                    "n_iter_max": 200,
                    "init": "random",
                    "svd": "truncated_svd",
                    "tol": 1e-09,
                    "random_state": 0,
                    "verbose": True,
                },
                "verbose": 2,
            },
            "rearrange_factors": {
                "undo_concat_complexDim": False,
                "undo_concat_dictElements": False,
            },
        },
    },
}


## make params dicts with grid swept values
params = copy.deepcopy(params_template)
params = [params]
# params = [container_helpers.deep_update_dict(params_template, ['db', 'save_path0'], str(Path(val).resolve() / (name_save+str(ii)))) for val in dir_save]
# params = [helpers.deep_update_dict(param, ['db', 'save_path0'], val) for param, val in zip(params_template, dirs_save_all)]
# params = container_helpers.flatten_list([[container_helpers.deep_update_dict(p, ['lr'], val) for val in [0.00001, 0.0001, 0.001]] for p in params])

# params_unchanging, params_changing = container_helpers.find_differences_across_dictionaries(params)


## notes that will be saved as a text file in the outer directory
notes = \
"""
First attempt
"""
with open(str(Path(dir_save) / 'notes.txt'), mode='a') as f:
    f.write(notes)



## copy script .py file to dir_save
import shutil
Path(dir_save).mkdir(parents=True, exist_ok=True)
print(f'Copying {path_script} to {str(Path(dir_save) / Path(path_script).name)}')
shutil.copyfile(path_script, str(Path(dir_save) / Path(path_script).name))



## save parameters to file
parameters_batch = {
    'params': params,
    # 'params_unchanging': params_unchanging,
    # 'params_changing': params_changing
}
import json
with open(str(Path(dir_save) / 'parameters_batch.json'), 'w') as f:
    json.dump(parameters_batch, f)

# with open(str(Path(dir_save) / 'parameters_batch.json')) as f:
#     test = json.load(f)
    
## change permissions of the data files
[os.system(f"chmod -R 777 {p}") for p in [dir_save, dir_videos, path_ROIs]]

## run batch_run function
paths_scripts = [path_script]
params_list = params
max_n_jobs=1
name_save=name_job


## define print log paths
paths_log = [str(Path(dir_save) / f'{name_save}{jobNum}' / 'print_log_%j.log') for jobNum in range(len(params))]

## Prepare call str and set environment variables
os.environ['WANDB_API_KEY'] = wandb_api_key
# Initialize a wandb run with minimal settings.
import wandb
wandb.init(project='face_rhythm', name=name_wandb, config=params[0], reinit=True)

## define slurm SBATCH parameters
# sbatch_config_list = \
# [f"""#!/usr/bin/bash
# #SBATCH --job-name={name_slurm}
# #SBATCH --output={path}
# #SBATCH --constraint=intel
# #SBATCH --partition=medium
# #SBATCH -c 16
# #SBATCH -n 1
# #SBATCH --mem=64GB
# #SBATCH --time=0-23:59:00

# unset XDG_RUNTIME_DIR

# cd /n/data1/hms/neurobio/sabatini/rich/

# date

# echo "loading modules"
# module load gcc/9.2.0

# echo "activating environment"
# source activate {name_env}

# echo "starting job"
# python "$@"
# """ for path in paths_log]

# # SBATCH --constraint=intel
# # SBATCH --gres=gpu:1,vram:23G
# # SBATCH --partition=gpu_requeue

# # SBATCH --partition=gpu_quad
# # SBATCH --gres=gpu:1,vram:31G

# # SBATCH --constraint=intel
# # SBATCH --partition=short





sbatch_config_list = \
[f"""#!/usr/bin/bash
#SBATCH --account=kempner_bsabatini_lab  # The account name for the job.
#SBATCH --job-name={name_slurm}          # Job name
#SBATCH --output={path}                  # File to write: STDOUT (and STDERR if --error is not used)
#SBATCH --partition=kempner_requeue      # Partition (job queue)
#SBATCH --gres=gpu:1                     # Number of GPUs
#SBATCH -c 16                            # Number of cores (-c) on one node
#SBATCH -n 1                             # Number of nodes (-n)
#SBATCH --mem=64GB                       # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --time=0-1:00:00                 # Runtime in D-HH:MM:SS
#SBATCH --requeue                        # Requeue the job if it is preempted
#SBATCH --export=WANDB_API_KEY           # Export the WANDB_API_KEY environment variable to the job

echo "Unsetting XDG_RUNTIME_DIR"
unset XDG_RUNTIME_DIR                    # This prevents an error with the conda environment

echo "activating environment"
source activate {name_env}

echo "starting job with call: python $@"
python "$@"
""" for path in paths_log]

#SBATCH --mail-type=FAIL                 # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --hold                           # Hold the job in the queue


util.batch_run(
    paths_scripts=paths_scripts,
    params_list=params_list,
    sbatch_config_list=sbatch_config_list,
    max_n_jobs=max_n_jobs,
    dir_save=str(dir_save),
    name_save=name_save,
    verbose=True,
)

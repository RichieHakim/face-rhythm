## Import general libraries
from pathlib import Path
import os
import sys
import copy

print(f"dispatcher environment: {os.environ['CONDA_DEFAULT_ENV']}")

from face_rhythm import helpers, util

path_self, path_script, dir_save, name_job, name_slurm, name_env = sys.argv


print(path_self, dir_save)

## set paths
Path(dir_save).mkdir(parents=True, exist_ok=True)


date = '20221010'

dir_save = f'/n/data1/hms/neurobio/sabatini/rich/analysis/faceRhythm/mouse_g1/run_20230107/'

params_template = {
    "project": {
        "directory_project": str(Path(dir_save).resolve() / f'fr_{date}'),
        "overwrite_config": False,
        "initialize_visualization": False,
        "verbose": 2
    },
    "figure_saver": {
        "format_save": [
            "png"
        ],
        "kwargs_savefig": {
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "transparent": True,
            "dpi": 300
        },
        "overwrite": True,
        "verbose": 2
    },
    "paths_videos": {
        "directory_videos": f"/n/data1/hms/neurobio/sabatini/rich/data/res2p/round_7_experiments/mouse_g1/{date}/camera_data/cam4/exp/",
        "filename_videos_strMatch": "cam4.*avi",
        "depth": 1
    },
    "BufferedVideoReader": {
        "buffer_size": 1000,
        "prefetch": 1,
        "posthold": 1,
        "method_getitem": "by_video",
        "verbose": 1
    },
    "Dataset_videos": {
        "contiguous": False,
        "frame_rate_clamp": None,
        "verbose": 2
    },
    "ROIs": {
        "select_mode": "file",
        "path_file": f"/n/data1/hms/neurobio/sabatini/rich/analysis/faceRhythm/mouse_g1/ROIs_{date}.h5",
        "verbose": 2,
        "rois_points_idx": [
            0
        ],
        "point_spacing": 9
    },
    "PointTracker": {
        "rois_masks_idx": [
            1
        ],
        "contiguous": False,
        "params_optical_flow": {
            "method": "lucas_kanade",
            "mesh_rigidity": 0.015,
            "mesh_n_neighbors": 15,
            "relaxation": 0.0010,
            "kwargs_method": {
                "winSize": [
                    28,
                    28
                ],
                "maxLevel": 2,
                "criteria": [
                    2,
                    0.03
                ]
            }
        },
        "visualize_video": False,
        "params_visualization": {
            "alpha": 0.2,
            "point_sizes": 2
        },
        "params_outlier_handling": {
            "threshold_displacement": 180,
            "framesHalted_before": 30,
            "framesHalted_after": 30
        },
        "verbose": 2
    },
    "VQT_Analyzer": {
        "params_VQT": {
            "Q_lowF": 2,
            "Q_highF": 8,
            "F_min": 1,
            "F_max": 30,
            "n_freq_bins": 40,
            "win_size": 701,
            "plot_pref": False,
            "downsample_factor": 20,
            "padding": "valid",
            "DEVICE_compute": "cuda:0",
            "batch_size": 10,
            "return_complex": False,
            "progressBar": True
        },
        "normalization_factor": 0.95,
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
                "time"
            ],
            "names_dims_concat_array": [
                [
                    "xy",
                    "points"
                ]
            ],
            "concat_complexDim": False,
            "name_dim_concat_complexDim": "time",
            "name_dim_dictElements": "session",
            "method_handling_dictElements": "separate",
            "name_dim_concat_dictElements": "time",
            "idx_windows": None,
            "name_dim_array_window": "time"
        },
        "fit": {
            "method": "CP_NN_HALS",
            "params_method": {
                "rank": 12,
                "n_iter_max": 1000,
                "init": "random",
                "svd": "truncated_svd",
                "tol": 1e-09,
                "verbose": True
            },
            "DEVICE": "cuda:0",
            "verbose": 2
        },
        "rearrange_factors": {
            "undo_concat_complexDim": False,
            "undo_concat_dictElements": False
        }
    }
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
shutil.copy2(path_script, str(Path(dir_save) / Path(path_script).name));



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


## run batch_run function
paths_scripts = [path_script]
params_list = params
max_n_jobs=1
name_save=name_job


## define print log paths
paths_log = [str(Path(dir_save) / f'{name_save}{jobNum}' / 'print_log_%j.log') for jobNum in range(len(params))]

## define slurm SBATCH parameters
sbatch_config_list = \
[f"""#!/usr/bin/bash
#SBATCH --job-name={name_slurm}
#SBATCH --output={path}
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --partition=gpu_requeue
#SBATCH -c 20
#SBATCH -n 1
#SBATCH --mem=32GB
#SBATCH --time=0-02:00:00

unset XDG_RUNTIME_DIR

cd /n/data1/hms/neurobio/sabatini/rich/

date

echo "loading modules"
module load gcc/9.2.0

echo "activating environment"
source activate {name_env}

echo "starting job"
python "$@"
""" for path in paths_log]

#SBATCH --gres=gpu:rtx6000:1
#SBATCH --partition=gpu_requeue


util.batch_run(
    paths_scripts=paths_scripts,
    params_list=params_list,
    sbatch_config_list=sbatch_config_list,
    max_n_jobs=max_n_jobs,
    dir_save=str(dir_save),
    name_save=name_save,
    verbose=True,
)

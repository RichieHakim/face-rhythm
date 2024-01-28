from pathlib import Path

import warnings
import pytest
import tempfile

from face_rhythm import helpers, util, h5_handling, pipelines


def test_pipeline_tracking_simple(dir_data_test):
    dir_temp = str(tempfile.TemporaryDirectory().name)
    dir_project = str(Path(dir_temp).resolve() / 'project')
    dir_inputs       = str(Path(dir_data_test).resolve() / 'inputs')
    dir_outputs_true = str(Path(dir_data_test).resolve() / 'outputs')

    defaults = util.get_default_parameters(
        directory_project=dir_project,
        directory_videos=str(Path(dir_inputs)),
        filename_videos_strMatch=r'demo.*',
        path_ROIs=str(Path(dir_inputs) / 'ROIs.h5'),
    )
    SEED = 0
    params_partial = {
            "steps": [
                "load_videos",
                "ROIs",
                "point_tracking",
                "VQT",
                "TCA",
            ],
            "project": {
                "overwrite_config": True,
                "update_project_paths": True,
                "initialize_visualization": False,
                "use_GPU": False,
                "random_seed": SEED,
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
                "depth": 0,
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
                    "verbose": 2,
                },
                "make_rois": {
                    "rois_points_idx": [
                        0,
                    ],
                    "point_spacing": 9,
                },
            },
            "PointTracker": {
                "contiguous": True,
                "params_optical_flow": {
                    "method": "lucas_kanade",
                    "mesh_rigidity": 0.025,
                    "mesh_n_neighbors": 8,
                    "relaxation": 0.0015,
                    "kwargs_method": {
                        "winSize": [
                            20,
                            20,
                        ],
                        "maxLevel": 2,
                        "criteria": [
                            3,
                            2,
                            0.03,
                        ],
                    },
                },
                "visualize_video": False,
                "params_visualization": {
                    "alpha": 0.2,
                    "point_sizes": 2,
                },
                "params_outlier_handling": {
                    "threshold_displacement": 150,
                    "framesHalted_before": 10,
                    "framesHalted_after": 10,
                },
                "verbose": 2,
            },
            "VQT_Analyzer": {
                "params_VQT": {
                    "Q_lowF": 4,
                    "Q_highF": 10,
                    "F_min": 1.0,
                    "F_max": 60,
                    "n_freq_bins": 36,
                    "win_size": 501,
                    "symmetry": 'center',
                    "taper_asymmetric": True,
                    "plot_pref": False,
                    "downsample_factor": 20,
                    "padding": "valid",
                    "batch_size": 10,
                    "return_complex": False,
                    "progressBar": True,
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
                        "random_state": SEED,
                        "verbose": True,
                    },
                    "verbose": 2,
                },
                "rearrange_factors": {
                    "undo_concat_complexDim": False,
                    "undo_concat_dictElements": False,
                },
            },
        }
    params = helpers.prepare_params(
        params=params_partial, 
        defaults=defaults,
        error_on_missing_keys=False,
    )
    results = pipelines.pipeline_basic(params)

    ## Check run_data equality
    print(f"Checking run_data equality")
    paths_rundata_true = helpers.find_paths(
        dir_outer=dir_outputs_true,
        reMatch=r'.*',
    )
    paths_rundata_relative = [str(Path(p).relative_to(dir_outputs_true)) for p in paths_rundata_true]
    paths_rundata_test = [str(Path(dir_project) / Path(p)) for p in paths_rundata_relative]

    def load_file(path):
        if Path(path).suffix == '.pkl':
            return helpers.pickle_load(path)
        elif Path(path).suffix == '.json':
            return helpers.json_load(path)
        elif Path(path).suffix == '.yaml':
            return helpers.yaml_load(path)
        elif Path(path).suffix == '.h5':
            return h5_handling.simple_load(path, return_dict=True)
        else:
            raise ValueError(f"Unknown file type: {Path(path).suffix}")

    checker = helpers.Equivalence_checker(
            kwargs_allclose={'rtol': 1e-5, 'equal_nan': True},
            assert_mode=False,
            verbose=1,
        )
    for path_test, path_true in zip(paths_rundata_test, paths_rundata_true):
        print(f"Loading run_data from {path_test}")
        data_test, data_true = load_file(path_test), load_file(path_true)
        print(f"run_data loaded. Checking equality")
        checks = checker(test=data_test, true=data_true)
        fails = [key for key, val in helpers.flatten_dict(checks).items() if val[0]==False]
        if len(fails) > 0:
            warnings.warn(f"run_data equality check failed for keys: {fails}")
        else:
            print(f"run_data equality check finished successfully")
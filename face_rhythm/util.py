"""Miscellaneous project utilities: FR_Module base class, config I/O, system info, batch launcher.

Contains the shared ``FR_Module`` base class (save config / run_info / run_data
for every pipeline stage), YAML helpers, matplotlib -> numpy array helpers, a
system-info collector used to snapshot the environment in run_info.json, and a
SLURM batch-run wrapper.
"""

from pathlib import Path
import re
import time
from datetime import datetime
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union
import PIL
import warnings

import yaml
import numpy as np

from . import h5_handling
from . import helpers


def get_default_parameters(
    path_defaults=None,
    directory_project=None,
    directory_videos=None,
    filename_videos_strMatch=None,
    path_ROIs=None,
):
    """
    Returns a dictionary of default parameters for running face-rhythm pipelines. RH 2023

    Args:
        path_defaults (Optional[str]):
            Path to a JSON file containing a parameters dictionary. If
            provided, parameters are loaded from this file. If ``None``, the
            built-in defaults are used. (Default is ``None``)
        directory_project (Optional[str]):
            Directory to use as the project directory. Passed through to
            ``fr.project.prepare_project``. (Default is ``None``)
        directory_videos (Optional[str]):
            Directory containing the videos. Passed through to
            ``fr.helpers.find_paths`` to discover video paths. (Default is
            ``None``)
        filename_videos_strMatch (Optional[str]):
            Regex that video filenames must match. Passed through to
            ``fr.helpers.find_paths`` to filter discovered videos. (Default is
            ``None``)
        path_ROIs (Optional[str]):
            Path to the file containing the ROIs. Used by ``fr.rois.ROIs``
            when running in ``'file'`` mode instead of ``'gui'`` mode.
            (Default is ``None``)

    Returns:
        (dict):
            params (dict):
                Dictionary containing the default (or loaded) parameters for
                each pipeline stage.
    """

    if path_defaults is not None:
        defaults = helpers.json_load(path_defaults)
    else:
        defaults = {
            "steps": [
                "load_videos",
                "ROIs",
                "point_tracking",
                "VQT",
                "TCA",
            ],
            "project": {
                "directory_project": directory_project,
                "overwrite_config": True,
                "update_project_paths": True,
                "initialize_visualization": False,
                "use_GPU": True,
                "random_seed": None,
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
                "directory_videos": directory_videos,
                "filename_videos_strMatch": filename_videos_strMatch,
                # "filename_videos_strMatch": "test\.avi",
                "depth": 1,
            },
            "BufferedVideoReader": {
                "buffer_size": 1000,
                "prefetch": 1,
                "posthold": 1,
                "method_getitem": "by_video",
                "backend": "torchcodec",
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
                    "point_spacing": 9,
                },
            },
            "PointTracker": {
                "contiguous": False,
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
                "frames_freeze": None,
                "relaxation_during_freeze_frames": True,
                "verbose": 2,
            },
            "VQT_Analyzer": {
                "params_VQT": {
                    'Fs_sample': 120,
                    'Q_lowF': 4.0,
                    'Q_highF': 10.0,
                    'F_min': 1.0,
                    'F_max': 60,
                    'n_freq_bins': 36,
                    'window_type': 'hann',
                    'symmetry': 'center',
                    'taper_asymmetric': True,
                    'downsample_factor': 20,
                    'padding': 'valid',
                    'fft_conv': True,
                    'fast_length': True,
                    'take_abs': True,
                    'filters': None, 
                    'plot_pref': False,
                },
                "batch_size": 10,
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
                        "random_state": None,
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

    return defaults


class FR_Module:
    """
    Superclass for all face-rhythm module classes. Provides shared helpers
    for saving ``run_data``, ``run_info``, and ``config`` files. RH 2022

    Attributes:
        run_info (Optional[dict]):
            Per-run metadata populated by the subclass. Saved by
            :meth:`save_run_info`.
        run_data (Optional[dict]):
            Per-run output data populated by the subclass. Saved by
            :meth:`save_run_data`.
        module_name (str):
            Name of the concrete subclass; used as the top-level key in the
            config and run_info files.
    """
    def __init__(self):
        """Initializes empty ``run_info`` and ``run_data`` and records the subclass name."""
        self.run_info = None
        self.run_data = None

        ## Get module name
        self.module_name = self.__class__.__name__


    def save_config(
        self,
        path_config=None,
        overwrite=True,
        verbose=1
    ):
        """
        Appends ``self.config`` to the ``config.yaml`` file. RH 2022

        ``self.config`` is created by the subclass and should contain all
        parameters used to run the module.

        Args:
            path_config (str):
                Path to the ``config.yaml`` file. (Default is ``None``)
            overwrite (bool):
                If ``True``, overwrites the existing field for this module
                inside ``config.yaml``. (Default is ``True``)
            verbose (int):
                Verbosity level. Either \n
                * ``0``: Silent.
                * ``1``: Print warnings.
                * ``2``: Print all info. \n
                (Default is ``1``)
        """
        ## Assert if self.config is not None
        assert self.config is not None, 'FR ERROR: self.config is None. Module likely did not run properly. Please set self.config before saving.'

        ## Assert that path_config is a string, exists, is a file, is a yaml file, and is named properly
        assert isinstance(path_config, str), "FR ERROR: path_config must be a string"
        assert Path(path_config).exists(), "FR ERROR: path_config must exist"
        assert Path(path_config).is_file(), "FR ERROR: path_config must be a file"
        assert Path(path_config).suffix == ".yaml", "FR ERROR: path_config must be a yaml file"
        assert Path(path_config).name == "config.yaml", "FR ERROR: path_config must be named config.yaml"

        config = load_config_file(path_config)
            
        ## Append self.config to module_name key in config.yaml
        if (self.module_name in config.keys()) and not overwrite:
            print(f"FR Warning: Not saving anything. Field exists in dictionary and overwrite==False. '{self.module_name}' is already a field in config.yaml.") if verbose > 0 else None
            return None
        elif (self.module_name in config.keys()) and overwrite:
            print(f"FR Warning: Overwriting field. '{self.module_name}' already in config.yaml.") if verbose > 0 else None
            config[self.module_name] = self.config
        else:
            print(f"FR: Adding '{self.module_name}' to config.yaml") if verbose > 1 else None
            config[self.module_name] = self.config

        ## Update the date_modified field
        config["general"]["date_modified"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ## Save config.yaml file
        print(f'FR: Saving config.yaml to {path_config}') if verbose > 1 else None
        with open(path_config, 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper, sort_keys=False)


    def save_run_info(
        self,
        path_run_info=None,
        path_config=None,
        overwrite=True,
        verbose=1
    ):
        """
        Appends ``self.run_info`` to the ``run_info.json`` file.

        Exactly one of ``path_run_info`` or ``path_config`` must be supplied.

        Args:
            path_run_info (Optional[str]):
                Path to the ``run_info.json`` file. If ``None``,
                ``path_config`` must be provided and must contain
                ``config['paths']['run_info']``. If the file does not exist,
                it will be created. (Default is ``None``)
            path_config (Optional[str]):
                Path to the ``config.yaml`` file. If ``None``,
                ``path_run_info`` must be provided. (Default is ``None``)
            overwrite (bool):
                If ``True``, overwrites the existing field for this module
                inside ``run_info.json``. (Default is ``True``)
            verbose (int):
                Verbosity level. Either \n
                * ``0``: Silent.
                * ``1``: Print warnings.
                * ``2``: Print all info. \n
                (Default is ``1``)
        """
        ## Assert self.run_info and self.run_data are not None
        assert self.run_info is not None, 'FR ERROR: self.run_info is None. Module likely did not run properly. Please set self.run_info before saving.'
        assert self.run_data is not None, 'FR ERROR: self.run_data is None. Module likely did not run properly. Please set self.run_data before saving.'

        ## Assert that either path_run_info or path_config must be a string, but not both
        assert (path_run_info is not None) and (path_config is None) or (path_run_info is None) and (path_config is not None), "FR ERROR: Either path_run_info or path_config must be specified as a string, but not both"
        ## Get the one that is not None
        path = path_run_info if path_run_info is not None else path_config

        ## Assert that path is a string, exists, is a file, is a json file, and is named properly
        assert isinstance(path, str), "FR ERROR: path_run_info must be a string"
        assert Path(path).exists(), "FR ERROR: path_run_info must exist"
        assert Path(path).is_file(), "FR ERROR: path_run_info must be a file"
        if path_run_info is not None:
            assert Path(path_run_info).name == "run_info.json", "FR ERROR: path_run_info must be named run_info.json"
        if path_config is not None:
            assert Path(path_config).name == "config.yaml", "FR ERROR: path_config must be named config.yaml"

        ## Set path_run_info. Get from config if path_run_info is None
        path_run_info = load_yaml_safe(path_config)["paths"]["run_info"] if path_run_info is None else path_run_info

        ## Check if file exists and load it if it does
        ## If directory to file does not exist, create it
        if Path(path_run_info).exists()==False:
            print(f'FR: No existing run_info.json file found in {path_run_info}. \n Creating new run_info.json at {path_run_info}') if verbose > 0 else None
            Path(path_run_info).parent.mkdir(parents=True, exist_ok=True)
            run_info = {}
        else:
            print(f'FR: Loading file {path_run_info}') if verbose > 1 else None
            run_info = helpers.json_load(path_run_info, mode='r')
            
        ## Append self.run_info to module_name key in run_info.json
        if (self.module_name in run_info.keys()) and not overwrite:
            print(f"FR Warning: Not saving anything. Field exists in dictionary and overwrite==False. '{self.module_name}' is already a field in run_info.json.") if verbose > 0 else None
        elif (self.module_name in run_info.keys()) and overwrite:
            print(f"FR Warning: Overwriting field. '{self.module_name}' is already a field in the run_info.json dictionary.") if verbose > 0 else None
            run_info[self.module_name] = self.run_info
        else:
            print(f"FR: Adding '{self.module_name}' field to run_info.json") if verbose > 1 else None
            run_info[self.module_name] = self.run_info
        
        ## Save run_info.json file
        print(f'FR: Saving run_info.json to {path_run_info}') if verbose > 1 else None
        helpers.json_save(run_info, path_run_info, mode='w')
            

    def save_run_data(
        self,
        path_run_data=None,
        path_config=None,
        overwrite=True,
        use_compression=False,
        track_order=True,
        verbose=1
    ):
        """
        Saves ``self.run_data`` to an ``.h5`` file under the project's
        ``analysis_files`` directory. RH 2022

        ``self.run_data`` is created by the subclass and should contain all
        the data generated by the module. Exactly one of ``path_run_data`` or
        ``path_config`` must be supplied. The project directory should
        already exist (use ``face_rhythm.project.prepare_project``).

        Args:
            path_run_data (Optional[str]):
                Path to the output ``.h5`` file. If ``None``, ``path_config``
                must be provided and must contain ``config['paths']['project']``.
                Resolved path will be
                ``<project>/analysis_files/<module_name>.h5``. If the file
                does not exist, it will be created. (Default is ``None``)
            path_config (Optional[str]):
                Path to the ``config.yaml`` file. If ``None``,
                ``path_run_data`` must be provided. (Default is ``None``)
            overwrite (bool):
                If ``True``, overwrites the existing ``.h5`` file. (Default
                is ``True``)
            use_compression (bool):
                If ``True``, uses compression when writing the ``.h5`` file.
                (Default is ``False``)
            track_order (bool):
                If ``True``, preserves insertion order of keys inside the
                ``.h5`` file. (Default is ``True``)
            verbose (int):
                Verbosity level. Either \n
                * ``0``: Silent.
                * ``1``: Print warnings.
                * ``2``: Print all info. \n
                (Default is ``1``)
        """
        ## Assert self.run_data is not None
        assert self.run_data is not None, 'FR ERROR: self.run_data is None. Module likely did not run properly. Please set self.run_data before saving.'

        ## Assert that either path_run_data or path_config must be a string, but not both
        assert (path_run_data is not None) and (path_config is None) or (path_run_data is None) and (path_config is not None), "FR ERROR: Either path_run_info or path_config must be specified as a string, but not both"
        ## If path_run_data is None, then path_config must be a string, exist, be a file, be a yaml file, and contain the project directory
        if path_run_data is None:
            assert isinstance(path_config, str), "FR ERROR: path_config must be a string"
            assert Path(path_config).exists(), "FR ERROR: path_config must exist"
            assert Path(path_config).is_file(), "FR ERROR: path_config must be a file"
            assert Path(path_config).suffix == ".yaml", "FR ERROR: path_config must be a yaml file"
            config = load_yaml_safe(path_config)
            assert 'project' in config['paths'].keys(), "FR ERROR: config['paths']['project'] must exist in path_config"
            path_run_data = str(Path(config['paths']['project']) / 'analysis_files' / f'{self.module_name}.h5')
            print(f"FR: Using project directory (config['paths']['project']) from config.yaml to make run_data path: {path_run_data}") if verbose > 1 else None

        ## Assert path_run_data is a string
        assert isinstance(path_run_data, str), "FR ERROR: path_run_data must be a string"
        if path_run_data is not None:
            print(f"FR WARNING: path_run_data file is expected to be named '{self.module_name+'.h5'}' if it is part of a project. Please make sure this is correct.") if verbose > 0 else None
        ## If a file exists and overwrite is False, then print a warning and cancel out
        ## If a file exists and overwrite is True, then print a warning and continue
        if Path(path_run_data).exists():
            if not overwrite:
                print(f'FR Warning: Not saving anything. File exists and overwrite==False. {path_run_data} already exists.') if verbose > 0 else None
                return None
            else:
                print(f'FR Warning: Overwriting file. File: {path_run_data} already exists.') if verbose > 0 else None


        ## Create directory if it does not exist
        if not Path(path_run_data).parent.exists():
            print(f'FR: Creating directory {Path(path_run_data).parent}') if verbose > 1 else None
            Path(path_run_data).parent.mkdir(parents=True)

        ## Try to save run_data to .h5 file. If we get an error that it failed because the ile is already open, then search for all open h5py.File objects and close them.
        print(f'FR: Saving run_data to {path_run_data}') if verbose > 1 else None
        try:
            h5_handling.simple_save(
                dict_to_save=self.run_data, 
                path=path_run_data, 
                use_compression=use_compression, 
                track_order=track_order,
                write_mode=('w' if overwrite else 'w-'), 
                verbose=verbose>1
            )
        except OSError as e:
            if re.search('Unable.*already open', str(e)):
                print(f'FR Warning: {path_run_data} is already open. Closing all open h5py.File objects and trying again.') if verbose > 0 else None
                h5_handling.close_all_h5()
                h5_handling.simple_save(
                    dict_to_save=self.run_data, 
                    path=path_run_data, 
                    use_compression=use_compression, 
                    track_order=track_order,
                    write_mode=('w' if overwrite else 'w-'), 
                    verbose=verbose>1
                )
            else:
                raise e
        
        ## Assert that the file exists
        assert Path(path_run_data).exists(), "FR ERROR: path_run_data must exist"
        ## Warn if it was not saved recently
        if (time.time() - Path(path_run_data).stat().st_mtime) > 1:
            print(f'FR Warning: Saving run_data may have failed. {path_run_data} was not saved recently.') if verbose > 0 else None

        

def load_yaml_safe(path, verbose=0):
    """
    Loads a YAML file, falling back to ``yaml.Loader`` if ``FullLoader`` fails.

    Args:
        path (str):
            Path to the ``.yaml`` file.
        verbose (int):
            Verbosity level. Higher values print more info. (Default is ``0``)

    Returns:
        (dict):
            data (dict):
                Parsed YAML file as a dictionary.
    """
    print(f'FR: Loading file {path}') if verbose > 1 else None
    try:
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except yaml.YAMLError:
        print(f'FR Warning: Failed to load {path} with Loader=yaml.FullLoader. A field is likely not yaml compatible. Trying with yaml.Loader.')
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.Loader)

def load_config_file(path, verbose=0):
    """
    Loads a ``config.yaml`` file as a dictionary.

    Args:
        path (str):
            Path to the ``config.yaml`` file.
        verbose (int):
            Verbosity level. Higher values print more info. (Default is ``0``)

    Returns:
        (dict):
            config (dict):
                Parsed ``config.yaml`` file as a dictionary.
    """
    return load_yaml_safe(path, verbose=verbose)
def load_run_info_file(path, verbose=0):
    """
    Loads a ``run_info.json`` file as a dictionary.

    Args:
        path (str):
            Path to the ``run_info.json`` file.
        verbose (int):
            Verbosity level. Higher values print more info. (Default is ``0``)

    Returns:
        (dict):
            run_info (dict):
                Parsed ``run_info.json`` file as a dictionary.
    """
    return helpers.json_load(path, mode='r')


class Saver_Viz_Base:
    """
    Superclass for saving visualizations (e.g. :class:`Figure_Saver`,
    :class:`Image_Saver`).

    Args:
        path_config (Optional[str]):
            Path to the ``config.yaml`` file. Optional if ``dir_save`` is
            specified. (Default is ``None``)
        dir_save (Optional[str]):
            Directory to save visualizations into. Optional if ``path_config``
            is specified. (Default is ``None``)
        formats_save (List[str]):
            File formats to save visualizations as. Valid values depend on
            the saving method used by the subclass. (Default is ``['png']``)
        kwargs_method (Dict[str, Any]):
            Keyword arguments forwarded to the underlying save method.
            (Default is ``{}``)
        overwrite (bool):
            If ``True``, overwrites existing files. (Default is ``False``)
        verbose (int):
            Verbosity level. Either \n
            * ``0``: Silent.
            * ``1``: Print warnings.
            * ``2``: Print warnings and info. \n
            (Default is ``1``)

    Attributes:
        path_config (Optional[str]):
            Stored path to the ``config.yaml`` file.
        dir_save (str):
            Resolved directory used for saving outputs.
        formats_save (List[str]):
            Stored list of file formats.
        kwargs_method (Dict[str, Any]):
            Stored keyword arguments forwarded to the save method.
        overwrite (bool):
            Stored overwrite flag.
        verbose (int):
            Stored verbosity level.
    """
    def __init__(
        self,
        path_config: str=None,
        dir_save: str=None,
        formats_save: list=['png'],
        kwargs_method: dict={},
        overwrite: bool=False,
        verbose: int=1,
    ):
        """Initializes the saver, validates inputs, and ensures ``dir_save`` exists."""
        ## Validate inputs
        assert isinstance(path_config, str) or isinstance(dir_save, str), "FR ERROR: Either path_config or dir_save must be specified as a string."
        if path_config is not None:
            assert Path(path_config).exists(), "FR ERROR: path_config must exist"
        if isinstance(formats_save, str):
            formats_save = [formats_save]
        assert isinstance(formats_save, list), "FR ERROR: formats_save must be a list"
        assert all([isinstance(f, str) for f in formats_save]), "FR ERROR: formats_save must be a list of strings"

        ## Set attributes
        self.path_config = path_config
        self.dir_save = dir_save
        self.formats_save = formats_save
        self.kwargs_method = kwargs_method
        self.overwrite = overwrite
        self.verbose = verbose

        ## Load config file
        self.dir_save = str(Path(load_config_file(self.path_config)['paths']['project']) / 'visualizations') if dir_save is None else dir_save

        ## Create directory if it does not exist
        if not Path(self.dir_save).exists():
            Path(self.dir_save).mkdir(parents=True, exist_ok=True)
            print(f'FR: Created directory {self.dir_save}') if verbose > 0 else None

    def _save_single(
        self,
        name_save: str,
        obj_save: object,
        fn_save: callable,
        kwargs_method: dict={},
        format_save: str=None,
    ):
        """
        Saves a single visualization.

        Args:
            name_save (str):
                Name of the file to save the visualization as (without
                extension).
            obj_save (object):
                Object to save (e.g. a figure or image array).
            fn_save (Callable):
                Function used to save the visualization. Must accept the
                kwargs ``obj_save``, ``path_save``, ``format_save``, and
                ``kwargs_method``.
            kwargs_method (Dict[str, Any]):
                Keyword arguments forwarded to ``fn_save``. (Default is ``{}``)
            format_save (Optional[str]):
                File format to save the visualization as. If ``None``, the
                default format is used. (Default is ``None``)
        """
        ## Validate inputs
        assert isinstance(name_save, str), "FR ERROR: name_save must be a string"
        assert isinstance(obj_save, object), "FR ERROR: obj_save must be an object"
        assert callable(fn_save), "FR ERROR: fn_save must be callable"
        assert isinstance(kwargs_method, dict), "FR ERROR: kwargs_method must be a dictionary"
        assert isinstance(format_save, str), "FR ERROR: format_save must be a string"

        ## Set kwargs_method
        kwargs_method = {**self.kwargs_method, **kwargs_method}

        ## Set and prepare path to save
        path_save = str(Path(self.dir_save).resolve() / f"{name_save}.{format_save}")
        helpers.prepare_filepath_for_saving(path_save, mkdir=True, allow_overwrite=self.overwrite)

        ## assert that fn_save has the correct kwargs
        args_fn_save = inspect.getfullargspec(fn_save).args
        assert all([k in args_fn_save for k in ['obj_save', 'path_save', 'format_save']]), "FR ERROR: fn_save must have args: ['obj_save', 'path_save', 'format_save']"

        ## Save visualization
        fn_save(
            obj_save=obj_save,
            path_save=path_save,
            format_save=format_save,
            kwargs_method=kwargs_method,
        )

    def _inherit_from_attrs(self, vars, attrs):
        """
        Yields each value in ``vars``, falling back to the matching attribute
        on ``self`` when the value is ``None``.

        Args:
            vars (List[Any]):
                Candidate values supplied at the call site.
            attrs (List[str]):
                Attribute names on ``self`` to use as fallback values.

        Yields:
            (Any):
                value (Any):
                    Either the original value or the corresponding attribute.
        """
        for var, attr in zip(vars, attrs):
            if var is None:
                assert hasattr(self, attr), f"FR ERROR: {attr} must be specified in either the constructor or the method call"
                var = getattr(self, attr)
            yield var

    def __repr__(self):
        """Returns a string representation of the saver and its key attributes."""
        return f"Figure_Saver(path_config={self.path_config}, dir_save={self.dir_save}, formats_save={self.formats_save}, kwargs_method={self.kwargs_method}, overwrite={self.overwrite}, verbose={self.verbose})"


class Figure_Saver(Saver_Viz_Base):
    """
    Saves matplotlib figures to disk in one or more file formats. RH 2022

    Args:
        path_config (Optional[str]):
            Path to the ``config.yaml`` file. If ``None``, ``dir_save`` must
            be specified. (Default is ``None``)
        dir_save (Optional[str]):
            Directory to save the figure into. Used when ``path_config`` is
            ``None``. (Default is ``None``)
        formats_save (List[str]):
            File formats to save the figure as. Common choices are
            ``'png'``, ``'svg'``, ``'eps'``, and ``'pdf'``. (Default is
            ``['png']``)
        kwargs_savefig (Dict[str, Any]):
            Keyword arguments forwarded to ``matplotlib.figure.Figure.savefig``.
            (Default is ``{'bbox_inches': 'tight', 'pad_inches': 0.1,
            'transparent': True, 'dpi': 300}``)
        overwrite (bool):
            If ``True``, overwrites existing files. (Default is ``False``)
        verbose (int):
            Verbosity level. Either \n
            * ``0``: Silent.
            * ``1``: Print warnings.
            * ``2``: Print warnings and info. \n
            (Default is ``1``)

    Attributes:
        kwargs_savefig (Dict[str, Any]):
            Stored ``savefig`` keyword arguments.
    """
    def __init__(
        self,
        path_config: str=None,
        dir_save: str=None,
        formats_save: list=['png'],
        kwargs_savefig: dict={
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'transparent': True,
            'dpi': 300,
        },
        overwrite: bool=False,
        verbose: int=1,
    ):
        """Initializes the figure saver and stores ``kwargs_savefig``."""
        ## Initialize super
        super().__init__(
            path_config=path_config,
            dir_save=dir_save,
            formats_save=formats_save,
            overwrite=overwrite,
            verbose=verbose,
        )

        ## Set kwargs_savefig
        self.kwargs_savefig = kwargs_savefig

        self.__call__ = self.save_figure

    def save_figure(
        self,
        fig,
        name_save: str=None,
        dir_save: str=None,
        formats_save: str=None,
        kwargs_savefig: dict=None,
    ):
        """
        Saves a single matplotlib figure to one or more file formats.

        Args:
            fig (matplotlib.figure.Figure):
                Figure to save.
            name_save (Optional[str]):
                Name of the file to save the figure as (without extension).
                If ``None``, the figure's label is used. (Default is ``None``)
            dir_save (Optional[str]):
                Directory to save the figure into. If ``None``, the
                directory stored on the instance is used. (Default is
                ``None``)
            formats_save (Optional[Union[str, List[str]]]):
                File format(s) to save the figure as. If ``None``, the
                formats stored on the instance are used. (Default is
                ``None``)
            kwargs_savefig (Optional[Dict[str, Any]]):
                Keyword arguments forwarded to
                ``matplotlib.figure.Figure.savefig``. If ``None``, the
                stored kwargs are used. (Default is ``None``)
        """
        import matplotlib

        ## Set missing inputs
        name_save = name_save if name_save is not None else fig.get_label()
        name_save = 'fig' if len(name_save) == 0 else name_save        
        dir_save, formats_save, kwargs_savefig = self._inherit_from_attrs(
            vars=[dir_save, formats_save, kwargs_savefig], 
            attrs=['dir_save', 'formats_save', 'kwargs_savefig'],
        )
        formats_save = [formats_save] if not isinstance(formats_save, list) else formats_save

        ## Validate inputs
        assert isinstance(fig, matplotlib.figure.Figure), "FR ERROR: fig must be a matplotlib.figure.Figure"

        ## Save figure
        fn_save = lambda obj_save, path_save, format_save, kwargs_method: obj_save.savefig(path_save, format=format_save, **kwargs_method)
        for format_save in formats_save:
            self._save_single(
                name_save=name_save,
                obj_save=fig,
                fn_save=fn_save,
                kwargs_method=kwargs_savefig,
                format_save=format_save,
            )

class Image_Saver(Saver_Viz_Base):
    """
    Saves images and animated GIFs to disk using PIL. RH 2022

    Args:
        path_config (Optional[str]):
            Path to the ``config.yaml`` file. If ``None``, ``dir_save`` must
            be specified. (Default is ``None``)
        dir_save (Optional[str]):
            Directory to save the image into. Used when ``path_config`` is
            ``None``. (Default is ``None``)
        formats_save (List[str]):
            File formats to save the image as. Common choices are
            ``'png'``, ``'jpg'``, and ``'tif'``. (Default is ``['png']``)
        kwargs_PIL_save (Dict[str, Any]):
            Keyword arguments forwarded to ``PIL.Image.Image.save``.
            (Default is ``{}``)
        overwrite (bool):
            If ``True``, overwrites existing files. (Default is ``False``)
        verbose (int):
            Verbosity level. Either \n
            * ``0``: Silent.
            * ``1``: Print warnings.
            * ``2``: Print warnings and info. \n
            (Default is ``1``)

    Attributes:
        kwargs_PIL_save (Dict[str, Any]):
            Stored ``PIL.Image.save`` keyword arguments.
    """
    def __init__(
        self,
        path_config: str=None,
        dir_save: str=None,
        formats_save: list=['png'],
        kwargs_PIL_save: dict={
        },
        overwrite: bool=False,
        verbose: int=1,
    ):
        """Initializes the image saver and stores ``kwargs_PIL_save``."""
        ## Initialize super
        super().__init__(
            path_config=path_config,
            dir_save=dir_save,
            formats_save=formats_save,
            overwrite=overwrite,
            verbose=verbose,
        )

        ## Set kwargs_PIL_save
        self.kwargs_PIL_save = kwargs_PIL_save

        self.__call__ = self.save_image

    def save_image(
        self,
        array_image,
        name_save: str=None,
        dir_save: str=None,
        formats_save: str=None,
        kwargs_PIL_save: dict=None,
    ):
        """
        Saves a single image array as one or more files using PIL.

        Args:
            array_image (np.ndarray):
                Image to save. shape: *(H, W)* or *(H, W, C)* with ``C`` in
                ``{1, 3}``. If ``dtype`` is float, values must lie in
                ``[0, 1]`` and will be scaled by ``255`` and cast to *uint8*.
                If ``dtype`` is int, values must lie in ``[0, 255]`` and will
                be cast to *uint8*.
            name_save (Optional[str]):
                Name of the file to save the image as (without extension).
                If ``None``, ``'image'`` is used. (Default is ``None``)
            dir_save (Optional[str]):
                Directory to save the image into. If ``None``, the directory
                stored on the instance is used. (Default is ``None``)
            formats_save (Optional[Union[str, List[str]]]):
                File format(s) to save the image as. If ``None``, the
                formats stored on the instance are used. (Default is ``None``)
            kwargs_PIL_save (Optional[Dict[str, Any]]):
                Keyword arguments forwarded to ``PIL.Image.Image.save``. If
                ``None``, the stored kwargs are used. (Default is ``None``)
        """

        ## Set missing inputs
        name_save = name_save if name_save is not None else 'image'
        dir_save, formats_save, kwargs_PIL_save = self._inherit_from_attrs(
            vars=[dir_save, formats_save, kwargs_PIL_save],
            attrs=['dir_save', 'formats_save', 'kwargs_PIL_save'],
        )
        formats_save = [formats_save] if not isinstance(formats_save, list) else formats_save
        
        ## Validate inputs
        array_image = self._prepare_array_image(array_image)

        ## Save image
        for format_save in formats_save:
            self._save_single(
                name_save=name_save,
                obj_save=array_image,
                fn_save=self._fn_save_single_image,
                kwargs_method=kwargs_PIL_save,
                format_save=format_save,
            )
        
    def save_gif(
        self,
        array_images,
        name_save: str=None,
        dir_save: str=None,
        frame_rate: float=5.0,
        loop: int=True,
        optimize: bool=True,
        kwargs_PIL_save: dict=None,
    ):
        """
        Saves a sequence of images as an animated GIF using PIL.

        Args:
            array_images (List[np.ndarray]):
                List of frames to save. Each frame has shape *(H, W)* or
                *(H, W, C)* with ``C`` in ``{1, 3}``.
            name_save (Optional[str]):
                Name of the file to save the GIF as (without extension). If
                ``None``, ``'image'`` is used. (Default is ``None``)
            dir_save (Optional[str]):
                Directory to save the GIF into. If ``None``, the directory
                stored on the instance is used. (Default is ``None``)
            frame_rate (float):
                Playback frame rate in frames per second. (Default is ``5.0``)
            loop (Union[int, bool]):
                Number of times the GIF should loop. ``True`` loops forever.
                (Default is ``True``)
            optimize (bool):
                If ``True``, applies PIL's GIF size optimization. (Default
                is ``True``)
            kwargs_PIL_save (Optional[Dict[str, Any]]):
                Keyword arguments forwarded to ``PIL.Image.Image.save``. If
                ``None``, the stored kwargs are used. (Default is ``None``)
        """
        ## Set missing inputs
        name_save = name_save if name_save is not None else 'image'
        dir_save, kwargs_PIL_save = self._inherit_from_attrs(
            vars=[dir_save, kwargs_PIL_save],
            attrs=['dir_save', 'kwargs_PIL_save'],
        )
        formats_save = ['gif']
        
        ## Validate inputs
        assert isinstance(array_images, list), "FR ERROR: array_images must be a list"
        
        kwargs_PIL_save['optimize'] = optimize

        kwargs_method = {
            'frame_rate': frame_rate,
            'loop': loop,
            'kwargs_PIL_save': kwargs_PIL_save,
        }

        ## Save gif
        for format_save in formats_save:
            self._save_single(
                name_save=name_save,
                obj_save=array_images,
                fn_save=self._fn_save_gif,
                kwargs_method=kwargs_method,
                format_save=format_save,
            )

    def _fn_save_single_image(self, obj_save, path_save, format_save, kwargs_method):
        """
        Converts a 3D ``np.ndarray`` with ``shape[-1] in {1, 3}`` to a
        ``PIL.Image.Image`` and writes it to disk.

        Args:
            obj_save (np.ndarray):
                Image array. shape: *(H, W, C)* with ``C`` in ``{1, 3}``,
                dtype: *uint8*.
            path_save (str):
                Output file path.
            format_save (str):
                File format string. Aliases ``'jpg'`` -> ``'JPEG'`` and
                ``'tif'`` -> ``'TIFF'`` are applied.
            kwargs_method (Dict[str, Any]):
                Keyword arguments forwarded to ``PIL.Image.Image.save``.
        """
        format_LUT = {
            'jpg': 'JPEG',
            'tif': 'TIFF',
        }
        format_save = format_LUT.get(format_save, format_save)
        obj_save = PIL.Image.fromarray(obj_save, mode='RGB') if obj_save.shape[-1] == 3 else PIL.Image.fromarray(obj_save, mode='L')
        obj_save.save(path_save, format=format_save, **kwargs_method)

    def _fn_save_gif(self, obj_save, path_save, format_save, kwargs_method):
        """
        Saves a list of image arrays as an animated GIF using
        :func:`face_rhythm.helpers.save_gif`.

        Args:
            obj_save (List[np.ndarray]):
                Frames to save. Each has shape *(H, W, C)* with ``C`` in
                ``{1, 3}``, dtype: *uint8*.
            path_save (str):
                Output file path.
            format_save (str):
                File format string (``'gif'``).
            kwargs_method (Dict[str, Any]):
                Dictionary with keys ``'frame_rate'``, ``'loop'``, and
                ``'kwargs_PIL_save'`` forwarded to the GIF backend.
        """
        helpers.save_gif(
            array=obj_save, 
            path=path_save, 
            frameRate=kwargs_method['frame_rate'],
            loop=kwargs_method['loop'],
            backend='PIL',
            kwargs_backend=kwargs_method['kwargs_PIL_save'],
        )

        
    
    def _prepare_array_image(self, array_image):
        """
        Normalizes an input image to a 3D ``uint8`` array with channel
        dimension last.

        Args:
            array_image (np.ndarray):
                Input image. shape: *(H, W)* or *(H, W, C)*. Floats must lie
                in ``[0, 1]``; ints must lie in ``[0, 255]``.

        Returns:
            (np.ndarray):
                array_image (np.ndarray):
                    Prepared image. shape: *(H, W, C)*, dtype: *uint8*.
        """
        ## Validate inputs
        assert isinstance(array_image, np.ndarray), "FR ERROR: array_image must be a numpy.ndarray"
        assert array_image.ndim in [2, 3], "FR ERROR: array_image must be a 2D or 3D numpy.ndarray"

        ## Prepare array_image
        if array_image.ndim == 2:
            array_image = np.expand_dims(array_image, axis=-1)

        if np.issubdtype(array_image.dtype, np.floating):
            assert np.all((0 <= array_image) & (array_image <= 1)), "FR ERROR: images must be between 0 and 1"
            array_image = (array_image * 255).astype(np.uint8)
        elif np.issubdtype(array_image.dtype, np.integer):
            assert np.all((0 <= array_image) & (array_image <= 255)), "FR ERROR: images must be between 0 and 255"
            array_image = array_image.astype(np.uint8)
        else:
            raise ValueError("FR ERROR: array_image.dtype must be float or int")

        return array_image

    

def system_info(verbose: bool = False,) -> Dict:
    """
    Collects information about the OS, CPU, RAM, GPU, and key Python
    packages, and optionally prints it. RH 2022

    Args:
        verbose (bool):
            If ``True``, prints each section to stdout as it is collected.
            (Default is ``False``)

    Returns:
        (Dict):
            versions (Dict):
                Dictionary containing the system snapshot. Keys include
                ``'datetime'``, ``'face_rhythm'``, ``'operating_system'``,
                ``'cpu_info'``, ``'user'``, ``'ram'``, ``'gpu_info'``,
                ``'conda_env'``, ``'python'``, ``'gcc'``, ``'torch'``,
                ``'cuda'``, ``'cudnn'``, ``'torch_devices'``, and ``'pkgs'``.
    """
    ## Operating system and version
    import platform
    def try_fns(fn):
        try:
            return fn()
        except Exception:
            return None
    fns = {key: val for key, val in platform.__dict__.items() if (callable(val) and key[0] != '_')}
    operating_system = {key: try_fns(val) for key, val in fns.items() if (callable(val) and key[0] != '_')}
    print(f'== Operating System ==: {operating_system["uname"]}') if verbose else None

    ## CPU info
    try:
        import cpuinfo
        import multiprocessing as mp
        # cpu_info = cpuinfo.get_cpu_info()
        cpu_n_cores = mp.cpu_count()
        cpu_brand = cpuinfo.cpuinfo.CPUID().get_processor_brand(cpuinfo.cpuinfo.CPUID().get_max_extension_support())
        cpu_info = {'n_cores': cpu_n_cores, 'brand': cpu_brand}
        if 'flags' in cpu_info:
            cpu_info['flags'] = 'omitted'
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get cpu info. Got error: {e}')
        cpu_info = 'Error: Failed to get'
    print(f'== CPU Info ==: {cpu_info}') if verbose else None

    ## RAM
    import psutil
    ram = psutil.virtual_memory()
    print(f'== RAM ==: {ram}') if verbose else None

    ## User
    import getpass
    user = getpass.getuser()

    ## GPU
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        gpu_info = {gpu.id: gpu.__dict__ for gpu in gpus}
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get gpu info. Got error: {e}')
        gpu_info = 'Error: Failed to get'
    print(f'== GPU Info ==: {gpu_info}') if verbose else None
    
    ## Conda Environment
    import os
    if 'CONDA_DEFAULT_ENV' not in os.environ:
        conda_env = 'None'
    else:
        conda_env = os.environ['CONDA_DEFAULT_ENV']
    print(f'== Conda Environment ==: {conda_env}') if verbose else None

    ## Python
    import sys
    python_version = sys.version.split(' ')[0]
    print(f'== Python Version ==: {python_version}') if verbose else None

    ## GCC
    import subprocess
    try:
        gcc_version = subprocess.check_output(['gcc', '--version']).decode('utf-8').split('\n')[0].split(' ')[-1]
    except Exception as e:
        warnings.warn(f'RH WARNING: unable to get gcc version. Got error: {e}')
        gcc_version = 'Faled to get'
    print(f'== GCC Version ==: {gcc_version}') if verbose else None
    
    ## PyTorch
    import torch
    torch_version = str(torch.__version__)
    print(f'== PyTorch Version ==: {torch_version}') if verbose else None
    ## CUDA
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        torch_devices = [f'device {i}: Name={torch.cuda.get_device_name(i)}, Memory={torch.cuda.get_device_properties(i).total_memory / 1e9} GB' for i in range(torch.cuda.device_count())]
        print(f"== CUDA Version ==: {cuda_version}, CUDNN Version: {cudnn_version}, Number of Devices: {torch.cuda.device_count()}, Devices: {torch_devices}, ") if verbose else None
    else:
        cuda_version = None
        cudnn_version = None
        torch_devices = None
        print('== CUDA is not available ==') if verbose else None

    ## all packages in environment
    from importlib.metadata import distributions
    pkgs_dict = {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"] is not None}

    ## face_rhythm
    import face_rhythm
    import time
    face_rhythm_version = face_rhythm.__version__
    from importlib.metadata import distribution
    face_rhythm_dist = distribution("face_rhythm")
    face_rhythm_location = str(face_rhythm_dist._path.parent) if hasattr(face_rhythm_dist, '_path') else os.path.dirname(face_rhythm.__file__)
    face_rhythm_fileDate = time.ctime(os.path.getctime(face_rhythm_location))
    face_rhythm_stuff = {'version': face_rhythm_version, 'date_installed': face_rhythm_fileDate}
    print(f'== face_rhythm Version ==: {face_rhythm}') if verbose else None
    print(f'== face_rhythm date installed ==: {face_rhythm_fileDate}') if verbose else None

    ## get datetime
    from datetime import datetime
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    versions = {
        'datetime': dt,
        'face_rhythm': face_rhythm_stuff,
        'operating_system': operating_system,
        'cpu_info': cpu_info,  ## This is the slow one.
        'user': user,
        'ram': ram,
        'gpu_info': gpu_info,
        'conda_env': conda_env,
        'python': python_version,
        'gcc': gcc_version,
        'torch': torch_version,
        'cuda': cuda_version,
        'cudnn': cudnn_version,
        'torch_devices': torch_devices,
        'pkgs': pkgs_dict,
    }

    return versions

def batch_run(
    paths_scripts,
    params_list,
    sbatch_config_list,
    max_n_jobs=2,
    dir_save=None,
    name_save='jobNum_',
    verbose=True,
):
    r"""
    Submits a batch of SLURM jobs that each run a Python script with a
    parameter file. Adapted from BNPM. RH 2021

    A typical workflow is to sweep one script over a list of parameter
    dictionaries: each entry in ``params_list`` is written to its own job
    directory as ``params.json``, the corresponding SBATCH script is
    materialized, and ``sbatch`` is invoked. Variants with multiple scripts
    or multiple SBATCH configs are also supported -- any of
    ``paths_scripts``, ``params_list``, and ``sbatch_config_list`` may have
    length ``1`` (broadcast) or length ``n_jobs``.

    Args:
        paths_scripts (List[str]):
            Paths to the Python scripts to run. Length must be ``1`` or
            ``n_jobs``. Each script should accept the kwargs
            ``--path_params`` and ``--directory_save`` injected by this
            function.
        params_list (List[Dict[str, Any]]):
            Parameter dictionaries, one per job. Length must be ``1`` or
            ``n_jobs``. Each dictionary is written as ``params.json`` inside
            its job directory and its path is passed to the script.
        sbatch_config_list (List[str]):
            SBATCH script bodies, one per job. Length must be ``1`` or
            ``n_jobs``. Each string must contain the literal ``python "$@"``
            on its final command line; this is replaced with the resolved
            ``python <script> --path_params <...> --directory_save <...>``
            invocation before being written to disk.
        max_n_jobs (Optional[int]):
            Safety cap on the number of jobs that may be submitted. If the
            inferred ``n_jobs`` exceeds this value, a ``ValueError`` is
            raised. Set to ``None`` to disable the cap. (Default is ``2``)
        dir_save (Union[str, pathlib.Path]):
            Outer directory under which each job's subdirectory is created.
            Created if it does not exist. Must be supplied -- there is no
            sensible default. (Default is ``None``)
        name_save (Union[str, List[str]]):
            Base name for each job's subdirectory; the job index is always
            appended. If a string, it is reused for every job; if a list,
            it must have ``n_jobs`` items. (Default is ``'jobNum_'``)
        verbose (bool):
            If ``True``, prints a status line per submitted job. (Default
            is ``True``)
    """
    import json
    import os
    import shutil

    ## dir_save has no sensible default; caller must provide an explicit output directory.
    if dir_save is None:
        raise ValueError("dir_save must be provided")

    # make sure the arguments are matched in length
    n_jobs = max(len(paths_scripts), len(params_list), len(sbatch_config_list))
    if max_n_jobs is not None:
        if n_jobs > max_n_jobs:
            raise ValueError(f'Too many jobs requested: max_n_jobs={n_jobs} > n_jobs={max_n_jobs}')

    def rep_inputs(item, n_jobs):
        if len(item)==1 and (n_jobs>1):
            return helpers.Lazy_repeat_item(item[0], pseudo_length=n_jobs)
        else:
            return item

    paths_scripts      = rep_inputs(paths_scripts,   n_jobs)
    params_list        = rep_inputs(params_list,  n_jobs)
    sbatch_config_list = rep_inputs(sbatch_config_list, n_jobs)
    name_save          = rep_inputs([name_save], n_jobs)

    # setup the save path
    Path(dir_save).mkdir(parents=True, exist_ok=True)
    dir_save = Path(dir_save).resolve()

    # run the jobs
    for ii in range(n_jobs):
        dir_save_job = dir_save / f'{name_save[ii]}{ii}'
        dir_save_job.mkdir(parents=True, exist_ok=True)

        # save the script
        path_script_job = dir_save_job / Path(paths_scripts[ii]).name
        shutil.copyfile(paths_scripts[ii], path_script_job);

        # save the parameters        
        path_params_job = dir_save_job / 'params.json'
        with open(path_params_job, 'w') as f:
            json.dump(params_list[ii], f)

        # Prepare the sbatch_config
        ## assert the search term 'python "$@"' is in the sbatch_config_list
        assert 'python "$@"' in sbatch_config_list[ii], "FR ERROR: sbatch_config_list must contain 'python \"$@\"' at the end"
        ## Replace the "$@" with the arguments
        sbatch_config_list[ii] = sbatch_config_list[ii].replace(
            'python "$@"', 
            f'python {path_script_job} --path_params {path_params_job} --directory_save {dir_save_job}'
        )
        # save the shell scripts
        save_path_sbatchConfig = dir_save_job / 'sbatch_config.sh'
        with open(save_path_sbatchConfig, 'w') as f:
            f.write(sbatch_config_list[ii])

        # run the job
        if verbose:
            print(f'Submitting job: {name_save[ii]} {ii}')
        # ! sbatch --job-name=${name_save}_${ii} --output=${dir_save_job}/log.txt --error=${dir_save_job}/err.txt --time=${sbatch_config_list[ii]["time"]} --mem=${sbatch_config_list[ii]["mem"]} --cpus-per-task=${sbatch_config_list[ii]["cpus"]} --wrap="${paths_scripts[ii]} ${params_list[ii]} ${sbatch_config_list[ii]} ${dir_save_job}"
        # os.system(f'sbatch {save_path_sbatchConfig} {path_script_job} --path_params {path_params_job} --directory_save {dir_save_job}')
        os.system(f'sbatch {save_path_sbatchConfig}')

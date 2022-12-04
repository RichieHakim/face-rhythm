from pathlib import Path
import re
import time
from datetime import datetime

import yaml

from . import h5_handling
from . import helpers

class FR_Module:
    """
    The superclass for all of the Face Rhythm module classes.
    Allows for saving run_data, run_info, and config files.
    RH 2022
    """
    def __init__(self):
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
        Appends the self.config dictionary to the config.yaml file.
        This dictionary is created by the subclass and should contain
         all the parameters used to run the module.
        RH 2022

        Args:
            path_config (str):
                Path to config.yaml file.
            overwrite (bool):
                If True, overwrites fields within the config.yaml file.
            verbose (int):
                Verbosity level. 0 is silent. 1 is print warnings. 2 is print all.
        """
        ## Assert if self.config is not None
        assert self.config is not None, 'FR ERROR: self.config is None. Module likely did not run properly. Please set self.config before saving.'

        ## Assert that path_config is a string, exists, is a file, is a yaml file, and is named properly
        assert isinstance(path_config, str), "FR ERROR: path_config must be a string"
        assert Path(path_config).exists(), "FR ERROR: path_config must exist"
        assert Path(path_config).is_file(), "FR ERROR: path_config must be a file"
        assert Path(path_config).suffix == ".yaml", "FR ERROR: path_config must be a yaml file"
        assert Path(path_config).name == "config.yaml", "FR ERROR: path_config must be named config.yaml"

        config = load_yaml_safe(path_config)
            
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
        Appends the self.run_info dictionary to the run_info.yaml file.

        Args:
            path_run_info (str):
                Path to run_info.yaml file.
                Optional. If None, then path_config must be provided, and must
                 contain: config['paths']['project'].
                If the file does not exist, it will be created.
            path_config (str):
                Path to config.yaml file.
                Optional. If None, then path_run_info must be provided.
            overwrite (bool):
                If True, overwrites fields within the run_info.yaml file.
            verbose (int):
                Verbosity level. 0 is silent. 1 is print warnings. 2 is print all.
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
        use_compression=True,
        track_order=True,
        verbose=1
    ):
        """
        Appends the self.run_data dictionary to a .h5 file in the
         .../project/analaysis_files/'object method name'.h5.
        The self.run_data dictionary is created by the subclass and should contain
         all the data generated by the module.
        The project directory should already exist and can be created using
         the face_rhythm.project.prepare_project function.
        RH 2022

        Args:
            path_run_data (str):
                Path to .h5 file.
                Optional. If None, then path_config must be provided, and must
                 contain: config['paths']['project']
                If the file does not exist, it will be created.
            path_config (str):
                Path to config.yaml file.
                Optional. If None, then path_run_data must be provided.
                Should contain: config['paths']['project']. path_run_data will be:
                 .../config['paths']['project']/analysis_files/'object method name'.h5
            overwrite (bool):
                If True, overwrites fields within the .h5 file.
            use_compression (bool):
                If True, uses compression when saving the .h5 file.
            track_order (bool):
                If True, tracks the order of the data in the .h5 file.
            verbose (int):
                Verbosity level. 
                0: silent
                1: print warnings
                2: print all info
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
            assert Path(path_run_data).name == self.module_name+'.h5', f"FR ERROR: path_run_data must be named {self.module_name+'.h5'}"
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
    loads yaml file

    Args:
        path (str): 
            path to .yaml file

    Returns:
        (dict): 
            yaml file as a dictionary

    """
    print(f'FR: Loading file {path}') if verbose > 1 else None
    try:
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except:
        print(f'FR Warning: Failed to load {path} with Loader=yaml.FullLoader. A field is likely not yaml compatible. Trying with yaml.Loader.')
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.Loader)

def load_config_file(path, verbose=0):
    """
    Loads config.yaml file

    Args:
        path (str): 
            path to config.yaml file

    Returns:
        (dict): 
            config.yaml file as a dictionary

    """
    return load_yaml_safe(path, verbose=verbose)
def load_run_info_file(path, verbose=0):
    """
    Loads run_info.json file

    Args:
        path (str): 
            path to run_info.json file

    Returns:
        (dict): 
            run_info.json file as a dictionary

    """
    return helpers.json_load(path, mode='r')


class Figure_Saver:
    """
    Class for saving figures
    RH 2022
    """
    def __init__(
        self,
        path_config: str=None,
        dir_save: str=None,
        format_save: list=['png'],
        kwargs_savefig: dict={
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'transparent': True,
            'dpi': 300,
        },
        overwrite: bool=False,
        verbose: int=1,
    ):
        """
        Initializes Figure_Saver object

        Args:
            path_config (str):
                Path to config.yaml file. If None, then path_save must
                be specified.
            dir_save (str):
                Directory to save the figure. Used if path_config is None.
                Must be specified if path_config is None.
            format (list of str):
                Format(s) to save the figure. Default is 'png'.
                Others: ['png', 'svg', 'eps', 'pdf']
            overwrite (bool):
                If True, then overwrite the file if it exists.
            kwargs_savefig (dict):
                Keyword arguments to pass to fig.savefig().
            verbose (int):
                Verbosity level.
                0: No output.
                1: Warning.
                2: All info.
        """
        self._path_config = path_config
        ## Get dir_save from path_config if it is not specified
        assert (self._path_config is not None) or (dir_save is not None), "FR ERROR: path_config or dir_save must be specified"
        self.dir_save = str(Path(load_config_file(self._path_config)['paths']['project']) / 'visualizations') if dir_save is None else dir_save

        assert isinstance(format_save, list), "FR ERROR: format_save must be a list of strings"
        assert all([isinstance(f, str) for f in format_save]), "FR ERROR: format_save must be a list of strings"
        self.format_save = format_save

        assert isinstance(kwargs_savefig, dict), "FR ERROR: kwargs_savefig must be a dictionary"
        self.kwargs_savefig = kwargs_savefig

        self.overwrite = overwrite
        self.verbose = verbose

    def save(
        self,
        fig,
        name_file: str=None,
    ):
        """
        Save the figures.

        Args:
            fig (matplotlib.figure.Figure):
                Figure to save.
            name_file (str):
                Name of the file to save. If None, then the name of 
                the figure is used.
        """
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure), "FR ERROR: fig must be a matplotlib.figure.Figure"
        ## Get figure title
        if name_file is None:
            titles = [a.get_title() for a in fig.get_axes() if a.get_title() != '']
            name_file = '.'.join(titles)
        path_save = [str(Path(self.dir_save) / (name_file + '.' + f)) for f in self.format_save]

        ## Save figure
        for path, form in zip(path_save, self.format_save):
            if Path(path).exists():
                if self.overwrite:
                    print(f'FR Warning: Overwriting file. File: {path} already exists.') if self.verbose > 0 else None
                else:
                    print(f'FR Warning: Not saving anything. File exists and overwrite==False. {path} already exists.') if self.verbose > 0 else None
                    return None
            print(f'FR: Saving figure {path} as format(s): {form}') if self.verbose > 1 else None
            fig.savefig(path, format=form, **self.kwargs_savefig)

    def save_all(
        self,
        figs,
        names_files: str=None,
    ):
        """
        Save all figures in a list.

        Args:
            figs (list of matplotlib.figure.Figure):
                Figures to save.
            name_file (str):
                Name of the file to save. If None, then the name of 
                the figure is used.
        """
        import matplotlib.pyplot as plt
        assert isinstance(figs, list), "FR ERROR: figs must be a list of matplotlib.figure.Figure"
        assert all([isinstance(fig, plt.Figure) for fig in figs]), "FR ERROR: figs must be a list of matplotlib.figure.Figure"
        for fig, name_file in zip(figs, names_files):
            self.save(fig, name_file=name_file)

    def __call__(
        self,
        fig,
        name_file: str=None,
    ):
        """
        Calls save() method.
        """
        self.save(fig, name_file=name_file)

    def __repr__(self):
        return f"Figure_Saver(path_config={self._path_config}, dir_save={self.dir_save}, format={self.format_save}, overwrite={self.overwrite}, kwargs_savefig={self.kwargs_savefig}, verbose={self.verbose})"
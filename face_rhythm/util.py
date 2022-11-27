from pathlib import Path
import re
import time

import yaml

from . import h5_handling

class FR_Module:
    def __init__(self):
        self.run_info = None
        self.run_data = None

        ## Get module name
        self.module_name = self.__class__.__name__


    def save_run_info(
        self, 
        path_run_info=None, 
        path_config=None,
        overwrite=True, 
        verbose=1
    ):
        """
        Appends the self.run_info dictionary to the run_info.yaml file.
        RH 2022

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

        ## Assert that path is a string, exists, is a file, is a yaml file, and is named properly
        assert isinstance(path, str), "FR ERROR: path_run_info must be a string"
        assert Path(path).exists(), "FR ERROR: path_run_info must exist"
        assert Path(path).is_file(), "FR ERROR: path_run_info must be a file"
        assert Path(path).suffix == ".yaml", "FR ERROR: path_run_info must be a yaml file"
        if path_run_info is not None:
            assert Path(path_run_info).name == "run_info.yaml", "FR ERROR: path_run_info must be named run_info.yaml"
        if path_config is not None:
            assert Path(path_config).name == "config.yaml", "FR ERROR: path_config must be named config.yaml"

        ## Set path_run_info. Get from config if path_run_info is None
        path_run_info = load_yaml_safe(path_config)["paths"]["run_info"] if path_run_info is None else path_run_info

        ## Check if file exists and load it if it does
        ## If directory to file does not exist, create it
        if Path(path_run_info).exists()==False:
            print(f'FR: No existing run_info.yaml file found in {path_run_info}. \n Creating new run_info.yaml at {path_run_info}') if verbose > 0 else None
            Path(path_run_info).parent.mkdir(parents=True, exist_ok=True)
            run_info = {}
        else:
            print(f'FR: Loading file {path_run_info}') if verbose > 1 else None
            run_info = load_yaml_safe(path_run_info)
            
        ## Append self.run_info to module_name key in run_info.yaml
        if (self.module_name in run_info.keys()) and not overwrite:
            print(f"FR Warning: Not saving anything. Field exists in dictionary and overwrite==False. '{self.module_name}' is already a field in run_info.yaml.") if verbose > 0 else None
        elif (self.module_name in run_info.keys()) and overwrite:
            print(f"FR Warning: Overwriting field. '{self.module_name}' is already a field in the run_info.yaml dictionary.") if verbose > 0 else None
            run_info[self.module_name] = self.run_info
        else:
            print(f"FR: Adding '{self.module_name}' field to run_info.yaml") if verbose > 1 else None
            run_info[self.module_name] = self.run_info

        ## Save run_info.yaml file
        print(f'FR: Saving run_info.yaml to {path_run_info}') if verbose > 1 else None
        with open(path_run_info, 'w') as f:
            yaml.dump(run_info, f, Dumper=yaml.Dumper, sort_keys=False)


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

        ## Save config.yaml file
        print(f'FR: Saving config.yaml to {path_config}') if verbose > 1 else None
        with open(path_config, 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper, sort_keys=False)
            

    def save_run_data(
        self, 
        path_run_data=None, 
        path_config=None,
        overwrite=True, 
        use_compression=True,
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
            h5_handling.simple_save(dict_to_save=self.run_data, path=path_run_data, use_compression=use_compression, write_mode=('w' if overwrite else 'w-'), verbose=verbose>1)
        except OSError as e:
            if re.search('Unable.*already open', str(e)):
                print(f'FR Warning: {path_run_data} is already open. Closing all open h5py.File objects and trying again.') if verbose > 0 else None
                h5_handling.close_all_h5()
                h5_handling.simple_save(dict_to_save=self.run_data, path=path_run_data, use_compression=use_compression, write_mode=('w' if overwrite else 'w-'), verbose=verbose>1)
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
    Loads run_info.yaml file

    Args:
        path (str): 
            path to run_info.yaml file

    Returns:
        (dict): 
            run_info.yaml file as a dictionary

    """
    return load_yaml_safe(path, verbose=verbose)
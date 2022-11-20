
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
                If the file does not exist, it will be created.
            overwrite (bool):
                If True, overwrites fields within the run_info.yaml file.
            verbose (int):
                Verbosity level. 0 is silent. 1 is print warnings. 2 is print all.
        """
        import yaml
        from pathlib import Path

        ## Assert if self.run_info and self.run_data are not None
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
            print(f'FR Warning: {self.module_name} already in run_info.yaml. Not overwriting.') if verbose > 0 else None
        elif (self.module_name in run_info.keys()) and overwrite:
            print(f'FR Warning: {self.module_name} already in run_info.yaml. Overwriting.') if verbose > 0 else None
            run_info[self.module_name] = self.run_info
        else:
            print(f'FR: Adding {self.module_name} to run_info.yaml') if verbose > 1 else None
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
                If the file does not exist, it will be created.
            overwrite (bool):
                If True, overwrites fields within the config.yaml file.
            verbose (int):
                Verbosity level. 0 is silent. 1 is print warnings. 2 is print all.
        """
        import yaml
        from pathlib import Path

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
            print(f'FR Warning: {self.module_name} already in config.yaml. Not overwriting.') if verbose > 0 else None
        elif (self.module_name in config.keys()) and overwrite:
            print(f'FR Warning: {self.module_name} already in config.yaml. Overwriting.') if verbose > 0 else None
            config[self.module_name] = self.config
        else:
            print(f'FR: Adding {self.module_name} to config.yaml') if verbose > 1 else None
            config[self.module_name] = self.config

        ## Save config.yaml file
        print(f'FR: Saving config.yaml to {path_config}') if verbose > 1 else None
        with open(path_config, 'w') as f:
            yaml.dump(config, f, Dumper=yaml.Dumper, sort_keys=False)
            


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
    import yaml
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
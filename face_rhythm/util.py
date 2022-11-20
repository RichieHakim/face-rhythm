
class FR_Module:
    def __init__(self):
        self.run_info = None
        self.run_data = None

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
        """
        import yaml
        from pathlib import Path

        ## Assert if self.run_info and self.run_data are not None
        assert self.run_info is not None, 'FR: self.run_info is None. Module likely did not run properly. Please set self.run_info before saving.'
        assert self.run_data is not None, 'FR: self.run_data is None. Module likely did not run properly. Please set self.run_data before saving.'

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
        path_run_info = load_config_file(path_config)["paths"]["run_info"] if path_run_info is None else path_run_info

        ## Get module name
        module_name = self.__class__.__name__

        ## Check if file exists and load it if it does
        ## If directory to file does not exist, create it
        if Path(path_run_info).exists()==False:
            print(f'FR: No existing run_info.yaml file found in {path_run_info}. \n Creating new run_info.yaml at {path_run_info}') if verbose > 0 else None
            Path(path_run_info).parent.mkdir(parents=True, exist_ok=True)
            run_info = {}
        else:
            print(f'FR: Loading file {path_run_info}') if verbose > 1 else None
            try:
                with open(path_run_info, 'r') as f:
                    run_info = yaml.load(f, Loader=yaml.FullLoader)
            except:
                print(f'FR Warning: Failed to load {path_run_info} with Loader=yaml.FullLoader. A field is likely not yaml compatible. Trying with yaml.Loader.')
                with open(path_run_info, 'r') as f:
                    run_info = yaml.load(f, Loader=yaml.Loader)
            
        ## Append self.run_info to module_name key in run_info.yaml
        if (module_name in run_info.keys()) and not overwrite:
            print(f'FR Warning: {module_name} already in run_info.yaml. Not overwriting.') if verbose > 0 else None
        elif (module_name in run_info.keys()) and overwrite:
            print(f'FR Warning: {module_name} already in run_info.yaml. Overwriting.') if verbose > 0 else None
            run_info[module_name] = self.run_info
        else:
            print(f'FR: Adding {module_name} to run_info.yaml') if verbose > 1 else None
            run_info[module_name] = self.run_info

        ## Save run_info.yaml file
        print(f'FR: Saving run_info.yaml to {path_run_info}') if verbose > 1 else None
        with open(path_run_info, 'w') as f:
            yaml.dump(run_info, f, Dumper=yaml.Dumper, sort_keys=False)




def load_config_file(path_config):
    """
    loads config file

    Args:
        path_config (str): 
            path to config.yaml file

    Returns:
        config (dict): 
            config file as a dictionary

    """
    import yaml
    with open(path_config, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)
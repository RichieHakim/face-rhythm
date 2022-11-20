
class FR_Module:
    def __init__(self):
        pass

    def save_run_info(
        self, 
        path_run_info, 
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

        ## Get module name
        module_name = self.__class__.__name__

        ## Check if file exists and load it if it does
        ## If directory to file does not exist, create it
        if Path(path_run_info).exists()==False:
            print(f'FR: No existing run_info.yaml file found in {path_run_info}. \n Creating new run_info.yaml at {path_run_info}') if verbose > 0 else None
            Path(path_run_info).parent.mkdir(parents=True, exist_ok=True)
            run_info = {}
        else:
            print(f'FR: {path_run_info} exists. Loading file.') if verbose > 1 else None
            try:
                with open(path_run_info, 'r') as f:
                    run_info = yaml.load(f, Loader=yaml.FullLoader)
            except:
                print(f'FR Warning: loading {path_run_info} with yaml.FullLoader. A field is likely not yaml compatible. Trying with yaml.Loader.')
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
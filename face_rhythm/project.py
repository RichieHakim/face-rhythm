import yaml
from pathlib import Path

from datetime import datetime

from . import helpers, util


def prepare_project(
    directory_project='./',
    overwrite_config=False,
    update_project_paths=False,
    mkdir=True,
    initialize_visualization=True,
    verbose=1,
):
    """
    Prepares the project folder and data folder (if they don't exist)
    Creates the config file (if it doesn't exist or overwrite requested)
    Returns path to the config file

    Args:
        directory_project (str): 
            Path to the project. 
            If './' is passed, the current working directory is used
        overwrite_config (bool): 
            Whether to overwrite the ENTIRE config file with a brand
             new config file.
            If False, update_project_paths can still be set to True.
        update_project_paths (bool):
            If True, then will update the following within the config.yaml
             file to reflect the current project directory (directory_project):
                - paths > project: directory_project/
                - paths > config: directory_project/config.yaml
                - paths > run_info: directory_project/run_info.json
            If overwrite_config is True, then this is ignored.
        mkdir (bool):
            Whether to create the project directory if it doesn't exist
        initialize_visualization (bool):
            Whether to initialize cv2.imshow visualization. If on a server,
             this should be set to False.
        verbose (int):
            Verbosity level.
            0: No output
            1: Warnings
            2: Info

    Returns:
        path_config (str):
            path to the config file
        path_run_info (str):
            path to the run info file
        directory_project (str):
            path to the project directory
    """
    ## initialize cv2.imshow
    if initialize_visualization:
        print('Initializing cv2.imshow') if verbose > 1 else None
        helpers.prepare_cv2_imshow()

    def _create_config_file():
        """
        Creates a config.yaml file.
        """
        contents_basic = {
            'general': {
                'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'date_modified': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'path_created': path_config,
            },
            'paths': {
                'project': directory_project,
                'config': path_config,
                'run_info': path_run_info,
            },
        }

        ## Write to file with overwriting
        with open(path_config, 'w') as f:
            yaml.dump(contents_basic, f, sort_keys=False)
    
    ## Check if directory_project exists
    if mkdir:
        if not Path(directory_project).exists():
            print(f"FR: Creating project directory: {directory_project}") if verbose > 1 else None
            Path(directory_project).mkdir(parents=True, exist_ok=True)
    else:
        raise FileNotFoundError(f"FR ERROR: directory_project does not exist: {directory_project}")

    path_config = str(Path(directory_project) / 'config.yaml')
    path_run_info = str(Path(directory_project) / 'run_info.json')
    ## Check if project exists
    if (Path(path_config)).exists():
        print(f'FR: Found config.yaml file at {path_config}') if verbose > 1 else None
        if overwrite_config:
            print(f'FR: Overwriting config.yaml file at {path_config}') if verbose > 0 else None
            _create_config_file()
        elif update_project_paths:
            print(f'FR: Updating project paths in config.yaml file at {path_config}') if verbose > 0 else None
            config = util.load_config_file(path_config)
            config['paths']['project'] = directory_project
            config['paths']['config'] = path_config
            config['paths']['run_info'] = path_run_info
            config['general']['date_modified'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(path_config, 'w') as f:
                yaml.dump(config, f, Dumper=yaml.Dumper, sort_keys=False)
    else:
        _create_config_file()
        print(f'FR: No existing config.yaml file found in {directory_project}. \n Creating new config.yaml at {Path(directory_project) / "config.yaml"}')
        

    ## Make sure project folders exist
    (Path(directory_project) / 'analysis_files').mkdir(parents=True, exist_ok=True)
    (Path(directory_project) / 'visualizations').mkdir(parents=True, exist_ok=True)

    return path_config, path_run_info, directory_project

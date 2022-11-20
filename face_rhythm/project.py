import yaml
from pathlib import Path

from datetime import datetime

from .helpers import get_system_versions

def prepare_project(
    directory_project='./',
    overwrite_config=False,
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
            whether to overwrite the config

    Returns:
        config_filepath (str):
            path to the current config
    """
    def _create_config_file(path):
        """
        Creates a config.yaml file.
        """
        contents_basic = {
            'general': {
                'date_created': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'date_modified': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'system_versions': get_system_versions(),
                'path_created': path,
            },
            'paths': {
                'project': directory_project,
                'config': path,
            },
        }

        ## Write to file with overwriting
        with open(path, 'w') as f:
            yaml.dump(contents_basic, f)

    path_config = str(Path(directory_project) / 'config.yaml')
    ## Check if project exists
    if (Path(path_config)).exists():
        print(f'FR: Found config.yaml file at {path_config}') if verbose > 1 else None
        if overwrite_config:
            print(f'FR: Overwriting config.yaml file at {path_config}') if verbose > 0 else None
            _create_config_file(path=path_config)
    else:
        _create_config_file(path=path_config)
        print(f'FR: No existing config.yaml file found in {directory_project}. \n Creating new config.yaml at {Path(directory_project) / "config.yaml"}')
        

    ## Make sure project folders exist
    (Path(directory_project) / 'analysis_files').mkdir(parents=True, exist_ok=True)
    (Path(directory_project) / 'visualizations').mkdir(parents=True, exist_ok=True)

    return path_config

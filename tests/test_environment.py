import sys
from pathlib import Path
import re

def test_python_version(
    look_for_verion_in_environmentYaml=True, 
    filename_environmentYaml='environment.yml',
    fallback_version=3.11,
    verbose=1,
):
    """
    Test python version.
    Either use the version specified in environment.yaml file or
     falls back to the version specified in fallback_version.
    RH 2022

    Args:
        look_for_verion_in_environmentYaml (bool):
            If True, look for the version in environment.yaml file.
            If False, use the version specified in fallback_version.
        filename_environmentYaml (str):
            The name of the environment.yaml file.
        fallback_version (float):
            The version to use if look_for_verion_in_environmentYaml is False.
        verbose (int):
            If 0, do not print anything.
            If 1, print warnings.
            If 2, print all below and info.

    Returns:
        success (bool): 
            True if the version is correct, False otherwise.
    """
    ## find path to repo and environment.yml files
    path_repo = str(Path(__file__).parent.parent.parent)

    ## check if environment.yml exists. If not, warn user and 
    ##  use fallback version
    if look_for_verion_in_environmentYaml:
        path_envYaml = str(Path(path_repo) / filename_environmentYaml)
        if not Path(path_envYaml).exists():
            print(f'FR Warning: {path_envYaml} does not exist. Using fallback version {fallback_version}') if verbose > 0 else None
            version_test = fallback_version
        else:
            ## Read environment.yml file. Find string that contains python version.
            ## Extract version number and convert to float
            with open(path_envYaml, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if re.search(' python=', line):
                        version_test = float(line.split('=')[1])
                        print(f'FR: Found environment specification for PYTHON VERSION {version_test} from line: {line} in {path_envYaml}') if verbose > 1 else None
                        break
    else:
        version_test = fallback_version

    ## Test if python version and subversion are correct
    version_system = float(f'{sys.version_info.major}.{sys.version_info.minor}')
    print(f'FR: PYTHON VERSION system: {version_system}') if verbose > 1 else None
    if version_test != version_system:
        print(f'FR Error: PYTHON VERSION {version_system} does not match specification: {version_test}. Please check your environment.')
        return False
    print(f'FR: PYTHON VERSION on system: {version_system} matches specification: {version_test}') if verbose > 1 else None
    # return True


def test_torch_cuda(verbose=1, device='cuda'):
    """
    Test to see if torch can do operations on GPU if CUDA is available.
    RH 2022

    Args:
        verbose (int):
            If 0, do not print anything.
            If 1, print warnings.
            If 2, print all below and info.
        device (str):
            The device to use. Default is 'cuda'.
    """
    import torch
    ## Check if CUDA is available
    if torch.cuda.is_available():
        print(f'FR: CUDA is available. Environment using PyTorch version: {torch.__version__}') if verbose > 1 else None
        arr = torch.rand(1000, 100).to(device)
        arr2 = torch.rand(100, 1000).to(device)
        arr3 = (arr @ arr2).mean().cpu().numpy()
        print(f'FR: Torch can do basic operations on GPU. Environment using PyTorch version: {torch.__version__}. Result of operations: {arr3}') if verbose > 1 else None

    else:
        print(f'FR Error: CUDA is not available. Environment using PyTorch version: {torch.__version__}') if verbose > 0 else None
        # return True
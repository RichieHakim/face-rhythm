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
        use_compression=False,
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
        dir_save: str=None,
    ):
        """
        Save the figures.

        Args:
            fig (matplotlib.figure.Figure):
                Figure to save.
            name_file (str):
                Name of the file to save. If None, then the name of 
                the figure is used.
            dir_save (str):
                Directory to save the figure. If None, then the directory
                 specified in the initialization is used.
        """
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure), "FR ERROR: fig must be a matplotlib.figure.Figure"

        ## Get dir_save
        dir_save = self.dir_save if dir_save is None else str(Path(dir_save).resolve())

        ## Get figure title
        if name_file is None:
            titles = [a.get_title() for a in fig.get_axes() if a.get_title() != '']
            name_file = '.'.join(titles)
        path_save = [str(Path(dir_save) / (name_file + '.' + f)) for f in self.format_save]

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

    def save_batch(
        self,
        figs,
        names_files: str=None,
        dir_save: str=None,
    ):
        """
        Save all figures in a list.

        Args:
            figs (list of matplotlib.figure.Figure):
                Figures to save.
            name_file (str):
                Name of the file to save. If None, then the name of 
                the figure is used.
            dir_save (str):
                Directory to save the figure. If None, then the directory
                 specified in the initialization is used.
        """
        import matplotlib.pyplot as plt
        assert isinstance(figs, list), "FR ERROR: figs must be a list of matplotlib.figure.Figure"
        assert all([isinstance(fig, plt.Figure) for fig in figs]), "FR ERROR: figs must be a list of matplotlib.figure.Figure"

        ## Get dir_save
        dir_save = self.dir_save if dir_save is None else str(Path(dir_save).resolve())

        for fig, name_file in zip(figs, names_files):
            self.save(fig, name_file=name_file, dir_save=dir_save)

    def __call__(
        self,
        fig,
        name_file: str=None,
        dir_save: str=None,
    ):
        """
        Calls save() method.
        """
        self.save(fig, name_file=name_file, dir_save=dir_save)

    def __repr__(self):
        return f"Figure_Saver(path_config={self._path_config}, dir_save={self.dir_save}, format={self.format_save}, overwrite={self.overwrite}, kwargs_savefig={self.kwargs_savefig}, verbose={self.verbose})"



def get_system_versions(verbose=False):
    """
    Checks the versions of various important softwares.
    Prints those versions
    RH 2022

    Args:
        verbose (bool): 
            Whether to print the versions

    Returns:
        versions (dict):
            Dictionary of versions
    """
    ## Operating system and version
    import platform
    operating_system = str(platform.system()) + ': ' + str(platform.release()) + ', ' + str(platform.version()) + ', ' + str(platform.machine()) + ', node: ' + str(platform.node()) 
    print(f'Operating System: {operating_system}') if verbose else None

    ## Conda Environment
    import os
    conda_env = os.environ['CONDA_DEFAULT_ENV']
    print(f'Conda Environment: {conda_env}') if verbose else None

    ## Python
    import sys
    python_version = sys.version.split(' ')[0]
    print(f'Python Version: {python_version}') if verbose else None

    ## GCC
    import subprocess
    gcc_version = subprocess.check_output(['gcc', '--version']).decode('utf-8').split('\n')[0].split(' ')[-1]
    print(f'GCC Version: {gcc_version}') if verbose else None
    
    ## PyTorch
    import torch
    torch_version = str(torch.__version__)
    print(f'PyTorch Version: {torch_version}') if verbose else None
    ## CUDA
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        print(f"\
CUDA Version: {cuda_version}, \
CUDNN Version: {torch.backends.cudnn.version()}, \
Number of Devices: {torch.cuda.device_count()}, \
Devices: {[f'device {i}: Name={torch.cuda.get_device_name(i)}, Memory={torch.cuda.get_device_properties(i).total_memory / 1e9} GB' for i in range(torch.cuda.device_count())]}, \
") if verbose else None
    else:
        cuda_version = None
        print('CUDA is not available') if verbose else None

    ## Numpy
    import numpy
    numpy_version = numpy.__version__
    print(f'Numpy Version: {numpy_version}') if verbose else None

    ## OpenCV
    import cv2
    opencv_version = cv2.__version__
    print(f'OpenCV Version: {opencv_version}') if verbose else None
    # print(cv2.getBuildInformation())

    ## face-rhythm
    import face_rhythm
    faceRhythm_version = face_rhythm.__version__
    print(f'face-rhythm Version: {faceRhythm_version}') if verbose else None

    versions = {
        'face-rhythm_version': faceRhythm_version,
        'operating_system': operating_system,
        'conda_env': conda_env,
        'python_version': python_version,
        'gcc_version': gcc_version,
        'torch_version': torch_version,
        'cuda_version': cuda_version,
        'numpy_version': numpy_version,
        'opencv_version': opencv_version,
    }

    return versions


def batch_run(paths_scripts, 
                params_list, 
                sbatch_config_list, 
                max_n_jobs=2,
                dir_save='/n/data1/hms/neurobio/sabatini/rich/analysis/', 
                name_save='jobNum_', 
                verbose=True,
                ):
    r"""
    FROM BNPM
    Run a batch of jobs.
    Workflow 1: run a single script over a sweep of parameters
        - Make a script that takes in the set of parameters
           you wish to sweep over as variables.
        - Prepend the script to take in string arguments
           pointing to a param_config file (maybe a dict).
           See paths_scripts Arg below for details.
        - Save the script in .py file.
        - In a new script, call this function (batch_run)
        - A new job will be run for each item in params_list
            - Each job will make a new directory, and within
               it will save (1) a .json file containing the 
               parameters used, and (2) the .sh file that 
               was run.
        - Save output files using the 'dir_save' argument

    Alternative workflows where you have multiple different
     scripts or different config files are also possible.

    RH 2021

    Args:
        paths_scripts (List):
            - List of script paths to run.
            - List can contain either 1 or n_jobs items.
            - Each script must save its results it's own way
               using a relative path (see 'dir_save' below)
            - Each script should contain the following to handle
               input arguments specified by the user and this
               function, DEMO:
                ```
                import sys
                    path_script, path_params, dir_save = sys.argv
                
                import json
                with open(path_params, 'r') as f:
                    params = json.load(f)
                ```                
            - It's also good practice to save the script .py file
               within dir_save DEMO:
                ```
                import shutil
                shutil.copy2(
                    path_script, 
                    str(Path(dir_save) / Path(path_script).name)
                    );
                ```
        params_list (List):
            - Parameters (arguments) to be used
            - List can contain either 1 or n_jobs items.
            - Each will be saved as a .json file (so nothing too big)   
            - Will be save into each inner/job directory and the path
               will be passed to the script for each job.
        sbatch_config_list (List):
            - List of string blocks containing the arguments to 
               pass for each job/script.
            - List can contain either 1 or n_jobs items.
            - Must contain: python "$@" at the bottom (to take in 
               arguments), and raw string must have '\n' to signify
               line breaks.
               Demo: '#!/usr/bin/bash
                    #SBATCH --job-name=python_01
                    #SBATCH --output=jupyter_logs/python_01_%j.log
                    #SBATCH --partition=priority
                    #SBATCH -c 1
                    #SBATCH -n 1
                    #SBATCH --mem=1GB
                    #SBATCH --time=0-00:00:10

                    unset XDG_RUNTIME_DIR

                    cd /n/data1/hms/neurobio/sabatini/rich/

                    date

                    echo "loading modules"
                    module load gcc/9.2.0 cuda/11.2

                    echo "activating environment"
                    source activate ROI_env

                    echo "starting job"
                    python "$@" '
        max_n_jobs (int):
            - Maximum number of jobs that can be called
            - Used as a safety precaution
            - Be careful that params_list has the right len
        dir_save (str or Path):
            - Outer directory to save results to.
            - Will be created if it does not exist.
            - Will be populated by folders for each job
            - Will be sent to the script for each job as the
               third argument. See paths_scripts demo for details.
        name_save (str or List):
            - Name of each job (used as inner directory name)
            - If str, then will be used for all jobs 
            - Job iteration always appended to the end.
            - If List, then must have len(params_list) items.
        verbose (bool):
            - Whether or not to print progress
    """
    import json
    import os
    import shutil

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
        # save the shell scripts
        save_path_sbatchConfig = dir_save_job / 'sbatch_config.sh'
        with open(save_path_sbatchConfig, 'w') as f:
            f.write(sbatch_config_list[ii])
        # save the script
        path_script_job = dir_save_job / Path(paths_scripts[ii]).name
        shutil.copy2(src=paths_scripts[ii], dst=str(path_script_job));
        # save the parameters        
        path_params_job = dir_save_job / 'params.json'
        with open(path_params_job, 'w') as f:
            json.dump(params_list[ii], f)
    
        # run the job
        if verbose:
            print(f'Submitting job: {name_save[ii]} {ii}')
        # ! sbatch --job-name=${name_save}_${ii} --output=${dir_save_job}/log.txt --error=${dir_save_job}/err.txt --time=${sbatch_config_list[ii]["time"]} --mem=${sbatch_config_list[ii]["mem"]} --cpus-per-task=${sbatch_config_list[ii]["cpus"]} --wrap="${paths_scripts[ii]} ${params_list[ii]} ${sbatch_config_list[ii]} ${dir_save_job}"
        os.system(f'sbatch {save_path_sbatchConfig} {path_script_job} --path_params {path_params_job} --directory_save {dir_save_job}')

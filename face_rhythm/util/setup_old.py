from distutils.command.config import config
import cv2
import torch
import numpy as np

import yaml
from pathlib import Path

from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO

from face_rhythm.util import helpers

def prepare_cv2_imshow():
    """
    This function is necessary because cv2.imshow() 
     can crash the kernel if called after importing 
     av and decord.
    RH 2022
    """
    import cv2
    test = np.zeros((1,300,400,3))
    for frame in test:
        cv2.putText(frame, "Prepping CV2", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, "Calling this figure allows cv2.imshow ", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "to work without crashing if this function", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "is called before importing av and decord", (10,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('startup', frame)
        cv2.waitKey(100)
    cv2.destroyWindow('startup')


def setup_project(project_path, sessions_path, run_name, overwrite_config, remote, trials, multisession, update_paths=False):
    """
    Creates the project folder and data folder (if they don't exist)
    Creates the config file (if it doesn't exist or overwrite requested)
    Returns path to the config file

    Args:
        project_path (Path): path to the project (usually ./)
        sessions_path (Path): path to the session folders and videos
        run_name (str): name for this current run of Face Rhythm
        overwrite_config (bool): whether to overwrite the config
        remote (bool): whether running on remote
        trials (bool): whether using a trial structure for the recordings

    Returns:
        config_filepath (str): path to the current config
    """
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / 'configs').mkdir(parents=True, exist_ok=True)
    (project_path / 'data').mkdir(parents=True, exist_ok=True)
    (project_path / 'viz').mkdir(parents=True, exist_ok=True)
    sessions_path.mkdir(parents=True, exist_ok=True)
    config_filepath = project_path / 'configs' / f'config_{run_name}.yaml'
    if not config_filepath.exists() or overwrite_config:
        generate_config(config_filepath, project_path, sessions_path, remote, trials, multisession, run_name)
    elif update_paths:
        config = helpers.load_config(config_filepath)
        config['Paths']['project'] = str(project_path)
        config['Paths']['video'] = str(sessions_path)
        config['Paths']['data'] = str(project_path / 'data')
        config['Paths']['viz'] = str(project_path / 'viz')
        config['Paths']['config'] = str(config_filepath)

        for i_sesh, session in enumerate(config['General']['sessions']):
            path_nwb = str(project_path / 'data' / (session['name'] + config['General']['run_name'] + '.nwb'))
            config['General']['sessions'][i_sesh]['nwb'] = path_nwb
            print(f'Updated path to nwb file: {path_nwb}')
        with open(str(config_filepath), 'w') as f:
            yaml.safe_dump(config, f)
        
        # display(config)
        
        print(f'Updated path to config file: {config_filepath}')

    print_important_versions()

    # return config_filepath


    if not remote:  ## forgive me father for I have sinned. This hack is necessary to make the code work on servers.
        ###############################################################################
        ## This block of code is used to initialize cv2.imshow
        ## This is necessary because importing av and decord 
        ##  will cause cv2.imshow to fail unless it is initialized.
        ## Obviously, this should be commented out when running on
        ##  systems that do not support cv2.imshow like servers.
        ## Also be sure to import BNPM before importing most other
        ##  modules.
        test = np.zeros((1,300,400,3))
        for frame in test:
            cv2.putText(frame, "Prepping CV2", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, "Calling this figure allows cv2.imshow ", (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, "to run after importing av and decord", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow('test', frame)
            cv2.waitKey(100)
        cv2.destroyAllWindows()
        ###############################################################################


    return config_filepath


def print_important_versions():
    """
    Checks the versions of various important softwares.
    Prints those versions
    RH 2022

    Args:

    Returns:

    """
    ## Conda Environment
    import os
    print(f'Conda Environment: ' + os.environ['CONDA_DEFAULT_ENV'])

    ## Python
    import sys
    print(f'Python Version: ' + sys.version.split('|')[0])

    ## GCC
    import subprocess
    print(f'GCC Version: ' + subprocess.check_output(['gcc', '--version']).decode('utf-8').split('\n')[0])
    
    ## PyTorch
    print(f'PyTorch Version: ' + torch.__version__)

    ## Numpy
    print(f'Numpy Version: ' + np.__version__)

    ## OpenCV
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') # script currently works with v4.4.0
    print(f'OpenCV version: {major_ver}.{minor_ver}.{subminor_ver}')
    # print(cv2.getBuildInformation())

    ## Decord
    import decord
    print(f'Decord Version: ' + decord.__version__)


def generate_config(config_filepath, project_path, sessions_path, remote, trials, multisession, run_name):
    """
    Generates bare config file with just basic info

    Args:
        config_filepath (Path): path to config file
        project_path (Path): path to the project (usually ./)
        sessions_path (Path): path to the session folders and videos
        remote (bool): whether running on remote
        trials (bool): whether using a trial structure for the recordings
        multisession (bool): whether we'll be handling multiple sessions
        run_name (str): name for this current run of Face Rhythm

    Returns:
    """

    basic_config = {'General': {},
                    'Video': {},
                    'Paths': {},
                    'ROI': {},
                    'Optic': {},
                    'Clean': {},
                    'CDR': {},
                    'PCA': {},
                    'CQT': {},
                    'TCA': {},
                    'Comps':{}}
    basic_config['Paths']['project'] = str(project_path)
    basic_config['Paths']['video'] = str(sessions_path)
    basic_config['Paths']['data'] = str(project_path / 'data')
    basic_config['Paths']['viz'] = str(project_path / 'viz')
    basic_config['Paths']['config'] = str(config_filepath)
    basic_config['General']['remote'] = remote
    basic_config['General']['trials'] = trials
    basic_config['General']['multisession'] = multisession
    basic_config['General']['run_name'] = run_name

    demo_path = project_path / 'viz' / 'demos'
    demo_path.mkdir(parents=True, exist_ok=True)
    basic_config['Video']['demos'] = str(demo_path)
    positional_path = project_path / 'viz' / 'positional'
    positional_path.mkdir(parents=True, exist_ok=True)
    basic_config['TCA']['dir_positional'] = str(positional_path)
    spectral_path = project_path / 'viz' / 'spectral'
    spectral_path.mkdir(parents=True, exist_ok=True)
    basic_config['TCA']['dir_spectral'] = str(spectral_path)

    with open(str(config_filepath), 'w') as f:
        yaml.safe_dump(basic_config, f)

class NoFilesError(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, folder, pattern):
        self.message = f'No files found in {folder} with pattern {pattern}'
        super().__init__(self.message)

class NoFoldersError(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, folder, pattern):
        self.message = f'No folders found in {folder} with pattern {pattern}'
        super().__init__(self.message)

def import_videos(config_filepath):
    """
    Loop over one folder and find all videos of interest

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """

    config = helpers.load_config(config_filepath)
    paths = config['Paths']
    video = config['Video']
    general = config['General']
    general['sessions'] = []

    path_vids = []
    session = {'name': 'session', 'videos': []}
    for vid in Path(paths['video']).iterdir():
        if video['file_strMatch'] in str(vid.name):
            if vid.suffix in ['.avi', '.mp4','.mov','.MOV', '.m4v']:
                # session['videos'].append(str(vid))
                path_vids.append(str(vid))
        elif vid.name in ['trial_indices.npy'] and general['trials']:
            session['trial_inds'] = str(vid)
            trial_inds = np.load(session['trial_inds'])
            session['num_trials'] = trial_inds.shape[0]
            session['trial_len'] = trial_inds.shape[1]
        elif vid.name in ['frames_to_ignore.npy']:
            session['frames_to_ignore'] = str(vid)
    
    if video['sort_filenames']:
        path_vids.sort()
    
    session['videos'] = path_vids

    general['sessions'].append(session)
    helpers.save_config(config, config_filepath)

    if len(session['videos']) == 0:
        raise NoFilesError(paths['video'], video['file_strMatch'])


def import_videos_multisession(config_filepath):
    """
    Loop over all sessions and find all videos for each session

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """

    config = helpers.load_config(config_filepath)
    paths = config['Paths']
    video = config['Video']
    general = config['General']
    general['sessions'] = []

    for path in Path(paths['video']).iterdir():
        if path.is_dir() and video['session_prefix'] in str(path.name):
            session = {'name': path.stem, 'videos': []}
            for vid in path.iterdir():
                if vid.suffix in ['.avi', '.mp4','.MOV','.mov']:
                    session['videos'].append(str(vid))
                elif vid.suffix in ['.npy'] and general['trials']:
                    session['trial_inds'] = str(vid)
                    trial_inds = np.load(session['trial_inds'])
                    session['num_trials'] = trial_inds.shape[0]
                    session['trial_len'] = trial_inds.shape[1]
            general['sessions'].append(session)
    helpers.save_config(config, config_filepath)

    if len(general['sessions']) == 0:
        raise NoFoldersError(paths['video'], video['session_prefix'])




def print_session_report(session):
    """
    Prints a simple report of all the session data

    Args:
        session (dict): session dictionary

    Returns:

    """

    print(f'Current Session: {session["name"]}')
    print(f'number of videos: {session["num_vids"]}')
    print(f'number of frames per video (roughly): {session["frames_per_video"]}')
    print(f'number of frames in ALL videos (roughly): {session["frames_total"]}')


def get_video_data(config_filepath):
    """
    get info on the imported video(s): num of frames, video height and width, framerate

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """
    config = helpers.load_config(config_filepath)
    general = config['General']
    video = config['Video']

    for session in general['sessions']:
        session['num_vids'] = len(session['videos'])
        vid_lens = np.ones(session['num_vids'])
        for i, vid_path in enumerate(session['videos']):
            vid_reader = cv2.VideoCapture(vid_path)
            vid_lens[i] = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        session['vid_lens'] = vid_lens.astype(int).tolist()
        session['frames_total'] = int(sum(session['vid_lens']))
        session['frames_per_video'] = int(session['frames_total'] / session['num_vids'])
        print_session_report(session)

        if video['print_filenames']:
            print(f'\n {np.array(session["videos"]).transpose()}')

    video['Fs'] = vid_reader.get(cv2.CAP_PROP_FPS)  ## Sampling rate (FPS). Manually change here if necessary
    print(f'Sampling rate pulled from video file metadata:   {round(video["Fs"], 3)} frames per second')

    vid_reader.set(1, 1)
    ok, frame = vid_reader.read()
    video['height'] = frame.shape[0]
    video['width'] = frame.shape[1]

    helpers.save_config(config, config_filepath)


def create_nwbs(config_filepath):
    """
    Create one nwb per session. This file will be used for all future data storage

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """

    config = helpers.load_config(config_filepath)
    general = config['General']
    paths = config['Paths']

    for session in general['sessions']:
        session['nwb'] = str(Path(paths['data']) / (session['name'] + general['run_name'] + '.nwb'))
        if not general['overwrite_nwbs'] and Path(session['nwb']).exists():
            print(f'nwb for {session["name"]} already exists, not overwriting')
            print('set config["General"]["overwrite_nwbs"]=True for otherwise')
            continue

        nwbfile = NWBFile(session_description=f'face rhythm data',
                          identifier=f'{session["name"]}',
                          session_start_time=datetime.now(tzlocal()),
                          file_create_date=datetime.now(tzlocal()))

        nwbfile.create_processing_module(name='Face Rhythm',
                                         description='all face rhythm related data')

        with NWBHDF5IO(session['nwb'], 'w') as io:
            io.write(nwbfile)

    helpers.save_config(config, config_filepath)


def prepare_videos(config_filepath):
    """
    Collects key video information and stores in the config

    Args:
        config_filepath (Path): path to the config file

    Returns:

    """
    config = helpers.load_config(config_filepath)
    if config['General']['multisession']:
        import_videos_multisession(config_filepath)
    else:
        import_videos(config_filepath)
    get_video_data(config_filepath)
    create_nwbs(config_filepath)
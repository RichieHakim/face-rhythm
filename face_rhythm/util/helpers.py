import yaml
import cv2
import torch
import sys
import numpy as np
import os
import os.path
import h5py

from pathlib import Path


def setup_project(project_path, video_path, session_name, overwrite_config):
    """
    Creates the project folder and data folder (if they don't exist)
    Creates the config file (if it doesn't exist or overwrite requested)
    Returns path to the config file

    Parameters
    ----------
    config_filepath (Path): path to config file

    Returns
    -------

    """
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / 'configs').mkdir(parents=True, exist_ok=True)
    (project_path / 'data').mkdir(parents=True, exist_ok=True)
    (project_path / 'viz').mkdir(parents=True, exist_ok=True)
    video_path.mkdir(parents=True, exist_ok=True)
    config_filepath = project_path / 'configs' / f'config_{session_name}.yaml'
    if not config_filepath.exists() or overwrite_config:
        generate_config(config_filepath, project_path, video_path)

    version_check(config_filepath)
    return config_filepath


def version_check(config_filepath):
    """
    Checks the versions of various important softwares.
    Prints those versions
    OS versioning check obsolete, to remove
    
    Parameters
    ----------
    config_filepath (Path): path to config file
    
    Returns
    -------
    
    """
    ### find version of openCV
    # script currently works with v4.4.0
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    print(f'OpenCV version: {major_ver}.{minor_ver}.{subminor_ver}')
    # print(cv2.getBuildInformation())

    ### find version of pytorch
    print(f'Pytorch version: {torch.__version__}')

    ## prep stuff
    ## find slash type of operating system

    if sys.platform == 'linux':
        slash_type = '/'
        print('Autodetected operating system: Linux. Using "/" for directory slashes')
    elif sys.platform == 'win32':
        slash_type = '\\'
        print(f'Autodetected operating system: Windows. Using "{slash_type}{slash_type}" for directory slashes')
    elif sys.platform == 'darwin':
        slash_type = '/'
        print(
            "What computer are you running this on? I haven't tested it on OSX or anything except windows and ubuntu.")
        print('Autodetected operating system: OSX. Using "/" for directory slashes')

    config = load_config(config_filepath)
    config['slash_type'] = slash_type
    save_config(config, config_filepath)


def load_config(config_filepath):
    """
    Loads config file into memory
    
    Parameters
    ----------
    config_filepath (Path): path to config file
    
    Returns
    -------
    config (dict) : actual config dict
    
    """
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_filepath):
    """
    Dumps config file to yaml
    
    Parameters
    ----------
    config (dict): config dict
    config_filepath (Path): path to config file
    
    Returns
    -------
    
    """
    with open(config_filepath, 'w') as f:
        yaml.safe_dump(config, f)


def generate_config(config_filepath, project_path, video_path):
    """
    Generates bare config file with just basic info
    
    Parameters
    ----------
    config_filepath (Path): path to config file
    
    Returns
    -------
    
    """

    basic_config = {'path_project': str(project_path),
                    'path_video': str(video_path),
                    'dir_vid': str(video_path),
                    'path_data': str(project_path / 'data'),
                    'path_viz': str(project_path / 'viz'),
                    'path_config': str(config_filepath)}

    with open(str(config_filepath), 'w') as f:
        yaml.safe_dump(basic_config, f)


def import_videos(config_filepath):
    """
    Define the directory of videos
    Import the videos as read objects
    
    Prints those versions
    
    Parameters
    ----------
    config_filepath (Path): path to the config file 
    
    Returns
    -------
    
    """

    config = load_config(config_filepath)
    multiple_files_pref = config['multiple_files_pref']
    dir_vid = config['dir_vid']
    fileName_vid_prefix = config['fileName_vid_prefix']
    fileName_vid = config['fileName_vid']
    slash_type = config['slash_type']

    if multiple_files_pref:
        ## first find all the files in the directory with the file name prefix
        fileNames_allInPathWithPrefix = []
        for ii in os.listdir(dir_vid):
            if os.path.isfile(os.path.join(dir_vid, ii)) and fileName_vid_prefix in ii:
                fileNames_allInPathWithPrefix.append(ii)
        numVids = len(fileNames_allInPathWithPrefix)

        ## make a variable containing all of the file paths
        path_vid_allFiles = list()
        for ii in range(numVids):
            path_vid_allFiles.append(f'{dir_vid}{slash_type}{fileNames_allInPathWithPrefix[ii]}')

    else:  ## Single file import
        path_vid = f'{dir_vid}{slash_type}{fileName_vid}'
        path_vid_allFiles = list()
        path_vid_allFiles.append(path_vid)
        numVids = 1

    config['numVids'] = numVids
    path_vid_allFiles = sorted(path_vid_allFiles)

    config['path_vid_allFiles'] = path_vid_allFiles

    save_config(config, config_filepath)


def get_video_data(config_filepath):
    """
    get info on the imported video(s): num of frames, video height and width, framerate
    
    Parameters
    ----------
    config_filepath (Path): path to the config file 
    
    Returns
    -------
    
    """

    config = load_config(config_filepath)
    multiple_files_pref = config['multiple_files_pref']
    path_vid_allFiles = config['path_vid_allFiles']
    numVids  = config['numVids']
    print_fileNames_pref = config['print_fileNames_pref']

    if multiple_files_pref:
        path_vid = path_vid_allFiles[0]
        video = cv2.VideoCapture(path_vid_allFiles[0])
        numFrames_firstVid = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        numFrames_allFiles = np.ones(numVids) * np.nan  # preallocation
        for ii in range(numVids):
            video = cv2.VideoCapture(path_vid_allFiles[ii])
            numFrames_allFiles[ii] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        numFrames_total_rough = np.uint64(sum(numFrames_allFiles))
        numFrames_allFiles = numFrames_allFiles.tolist()
        print(f'number of videos: {numVids}')
        print(f'number of frames in FIRST video (roughly):  {numFrames_firstVid}')
        print(f'number of frames in ALL videos (roughly):   {numFrames_total_rough}')
    else:
        video = cv2.VideoCapture(path_vid_allFiles[0])
        numFrames_onlyVid = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        numFrames_total_rough = numFrames_onlyVid
        numFrames_allFiles = numFrames_total_rough
        print(f'number of frames in ONLY video:   {numFrames_onlyVid}')

    Fs = video.get(cv2.CAP_PROP_FPS)  ## Sampling rate (FPS). Manually change here if necessary
    print(f'Sampling rate pulled from video file metadata:   {round(Fs, 3)} frames per second')

    if print_fileNames_pref:
        print(f'\n {np.array(path_vid_allFiles).transpose()}')

    video.set(1, 1)
    ok, frame = video.read()
    vid_height = frame.shape[0]
    vid_width = frame.shape[1]

    config['numFrames_total_rough'] = int(numFrames_total_rough)
    config['vid_Fs'] = Fs
    config['numFrames_allFiles'] = numFrames_allFiles
    config['vid_height'] = vid_height
    config['vid_width'] = vid_width

    save_config(config, config_filepath)


def save_data(config_filepath, save_name, data_to_save):
    """
    save an npy file with data

    Parameters
    ----------
    config_filepath (Path): path to the config file
    save_name (str): name of the object to be saved
    data_to_save (np.ndarray): (usually) a numpy array

    Returns
    -------

    """

    config = load_config(config_filepath)
    save_dir = Path(config['path_data'])
    save_path = save_dir / f'{save_name}.npy'
    np.save(save_path, data_to_save, allow_pickle=True)
    config[f'path_{save_name}'] = str(save_path)
    save_config(config, config_filepath)


def save_h5(config_filepath, save_name, data_dict):
    """
    save an h5 file from a data dictionary

    Parameters
    ----------
    config_filepath (Path): path to the config file
    save_name (str): name of the object to be saved
    data_dict (dict): dict of numpy arrays

    Returns
    -------

    """
    config = load_config(config_filepath)
    save_dir = Path(config['path_data'])
    save_path = save_dir / f'{save_name}.h5'
    to_write = h5py.File(save_path, 'w')
    dict_to_h5(data_dict, to_write)
    to_write.close()
    config[f'path_{save_name}'] = str(save_path)
    save_config(config, config_filepath)


def load_h5(config_filepath, data_key):
    """
    load an h5 file into a data dictionary
    proceed with caution given that this loads the entire h5 file into mem

    Parameters
    ----------
    config_filepath (Path): path to the config file
    save_name (str): name of the object to be saved
    data_dict (dict): dict of numpy arrays

    Returns
    -------

    """
    config = load_config(config_filepath)
    return h5_to_dict(config[data_key])


def load_data(config_filepath, data_key):
    """
    load an npy file with data

    Parameters
    ----------
    config_filepath (Path): path to the config file
    data_key (str): config key for the target data

    Returns
    -------
    data (np.ndarray): (usually) an np array with data

    """

    config = load_config(config_filepath)
    return np.load(config[data_key], allow_pickle=True)


def print_time(action, time):
    """
    prints the time adjusted for hours/minutes/seconds based on length

    Parameters
    ----------
    action (str): description of the completed action
    time (float): elapsed time

    Returns
    -------

    """

    hour = 60 * 60
    minute = 60
    if time > hour:
        reported_time = time / hour
        unit = 'hours'
    elif time > 2 * minute:
        reported_time = time / minute
        unit = 'minutes'
    else:
        reported_time = time
        unit = 'seconds'
    reported_time = round(reported_time, 2)
    print(f'{action}. Elapsed time: {reported_time} {unit}')


def h5_to_dict(h5file, path='/'):
    '''
    Reads all contents from h5 and returns them in a nested dict object.

    Parameters
    ----------
    h5file (str): path to h5 file
    path (str): path to group within h5 file

    Returns
    -------
    ans (dict): dictionary of all h5 group contents
    '''

    ans = {}

    if type(h5file) is str:
        with h5py.File(h5file, 'r') as f:
            ans = h5_to_dict(f, path)
            return ans

    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = h5_to_dict(h5file, path + key + '/')
    return ans


def dict_to_h5(data_dict, h5):
    '''
    Quick and dirty dict dumper to h5

    Parameters
    ----------
    data_dict (dict): dictionary (potentially nested) of data!
    h5 (h5py.File): h5 File (or Group) to populate

    Returns
    -------
    '''

    for key, item in data_dict.items():
        if isinstance(item, dict):
            group = h5.create_group(key)
            dict_to_h5(item, group)
        else:
            h5.create_dataset(key, data=item)

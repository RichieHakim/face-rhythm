import yaml
import cv2
import torch
import numpy as np

import h5py
import types

from pathlib import Path

from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO
from hdmf.backends.hdf5.h5tools import H5DataIO
import pynwb
from pynwb.behavior import BehavioralTimeSeries


def setup_project(project_path, video_path, run_name, overwrite_config, remote, trials):
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
    config_filepath = project_path / 'configs' / f'config_{run_name}.yaml'
    if not config_filepath.exists() or overwrite_config:
        generate_config(config_filepath, project_path, video_path, remote, trials)

    version_check(config_filepath)
    return config_filepath


def version_check(config_filepath):
    """
    Checks the versions of various important softwares.
    Prints those versions
    
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


def generate_config(config_filepath, project_path, video_path, remote, trials):
    """
    Generates bare config file with just basic info
    
    Parameters
    ----------
    config_filepath (Path): path to config file
    
    Returns
    -------
    
    """

    basic_config = {'General':{},
                    'Video':{},
                    'Paths':{},
                    'ROI':{},
                    'Optic':{},
                    'Clean':{},
                    'CDR':{},
                    'PCA':{},
                    'CQT':{},
                    'TCA':{}}
    basic_config['Paths']['project'] = str(project_path)
    basic_config['Paths']['video'] = str(video_path)
    basic_config['Paths']['data'] = str(project_path / 'data')
    basic_config['Paths']['viz'] = str(project_path / 'viz')
    basic_config['Paths']['config'] = str(config_filepath)
    basic_config['General']['remote'] = remote
    basic_config['General']['trials'] = trials

    with open(str(config_filepath), 'w') as f:
        yaml.safe_dump(basic_config, f)


def import_videos(config_filepath):
    """
    Find all videos

    Parameters
    ----------
    config_filepath (Path): path to the config file

    Returns
    -------

    """

    config = load_config(config_filepath)
    paths = config['Paths']
    video = config['Video']
    general = config['General']
    general['sessions'] = []

    for path in Path(paths['video']).iterdir():
        if path.is_dir() and video['session_prefix'] in str(path):
            session = {'name': path.stem, 'videos': []}
            for vid in path.iterdir():
                if vid.suffix in ['.avi', '.mp4']:
                    session['videos'].append(str(vid))
                elif vid.suffix in ['.npy'] and general['trials']:
                    session['trial_inds'] = str(vid)
                    trial_inds = np.load(session['trial_inds'])
                    session['num_trials'] = trial_inds.shape[0]
                    session['trial_len'] = trial_inds.shape[1]
            general['sessions'].append(session)

    save_config(config, config_filepath)


def print_session_report(session):
    print(f'Current Session: {session["name"]}')
    print(f'number of videos: {session["num_vids"]}')
    print(f'number of frames per video (roughly): {session["frames_per_video"]}')
    print(f'number of frames in ALL videos (roughly): {session["frames_total"]}')


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
    general = config['General']
    video = config['Video']

    for session in general['sessions']:
        session['num_vids'] = len(session['videos'])
        vid_lens = np.ones(session['num_vids'])
        for i, vid_path in enumerate(session['videos']):
            vid_reader = cv2.VideoCapture(vid_path)
            vid_lens[i] = int(vid_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        session['vid_lens'] = vid_lens.tolist()
        session['frames_total'] = int(sum(session['vid_lens']))
        session['frames_per_video'] = int(session['frames_total']/session['num_vids'])
        print_session_report(session)

        if video['print_filenames']:
            print(f'\n {np.array(session["videos"]).transpose()}')

    video['Fs'] = vid_reader.get(cv2.CAP_PROP_FPS)  ## Sampling rate (FPS). Manually change here if necessary
    print(f'Sampling rate pulled from video file metadata:   {round(video["Fs"], 3)} frames per second')

    vid_reader.set(1, 1)
    ok, frame = vid_reader.read()
    video['height'] = frame.shape[0]
    video['width'] = frame.shape[1]

    save_config(config, config_filepath)


def create_nwbs(config_filepath):
    """
    Create one nwb per session. This file will be used for all future data storage

    Parameters
    ----------
    config_filepath (Path): path to the config file

    Returns
    -------

    """

    config = load_config(config_filepath)
    general = config['General']
    paths = config['Paths']

    for session in general['sessions']:
        session['nwb'] = str(Path(paths['data']) / (session['name']+ '.nwb'))

        nwbfile = NWBFile(session_description=f'face rhythm data',
                          identifier=f'{session["name"]}',
                          session_start_time=datetime.now(tzlocal()),
                          file_create_date=datetime.now(tzlocal()))

        nwbfile.create_processing_module(name='Face Rhythm',
                                         description='all face rhythm related data')

        with NWBHDF5IO(session['nwb'], 'w') as io:
            io.write(nwbfile)

    save_config(config, config_filepath)


def create_nwb(config_filepath):
    """
    Create the nwb for one video. This file will be used for all future data storage

    Parameters
    ----------
    config_filepath (Path): path to the config file

    Returns
    -------

    """
    config = load_config(config_filepath)
    session_name = config['fileName_vid_prefix']
    data_dir = Path(config['path_data'])
    out_file = data_dir / (session_name + '.nwb')

    nwbfile = NWBFile(session_description=f'face rhythm data',
                      identifier=f'{session_name}',
                      session_start_time=datetime.now(tzlocal()),
                      file_create_date=datetime.now(tzlocal()))

    nwbfile.create_processing_module(name='Face Rhythm',
                                     description='all face rhythm related data')

    with NWBHDF5IO(str(out_file), 'w') as io:
        io.write(nwbfile)

    config['path_nwb'] = str(out_file)
    save_config(config, config_filepath)


def create_nwb_group(nwb_path, group_name):
    """
    Create an NWB BehavioralTimeSeries for grouping data

    Parameters
    ----------
    config_filepath (Path): path to the config file
    group_name (str): name of group to be created

    Returns
    -------

    """
    with NWBHDF5IO(nwb_path,'a') as io:
        nwbfile = io.read()
        if group_name in nwbfile.processing['Face Rhythm'].data_interfaces.keys():
            return
        new_group = BehavioralTimeSeries(name=group_name)
        nwbfile.processing['Face Rhythm'].add(new_group)
        io.write(nwbfile)


def create_nwb_ts(nwb_path, group_name, ts_name, data, Fs):
    """
    Create a new TimeSeries for data to write

    Parameters
    ----------
    config_filepath (Path): path to the config file
    group_name (str): name of group to write to
    ts_name (str): name of new ts
    data (np.array): data to be written

    Returns
    -------

    """
    print(f'Saving {ts_name} in Group {group_name}')
    with NWBHDF5IO(nwb_path, 'a') as io:
        nwbfile = io.read()
        maxshape = tuple(None for dim in data.shape)
        new_ts = pynwb.TimeSeries(name=ts_name,
                                  data=H5DataIO(np.moveaxis(data,-1,0), maxshape=maxshape),
                                  unit='mm',
                                  rate=Fs)
        if ts_name not in nwbfile.processing['Face Rhythm'][group_name].time_series:
            nwbfile.processing['Face Rhythm'][group_name].add_timeseries(new_ts)
        else:
            ts = nwbfile.processing['Face Rhythm'][group_name].get_timeseries(ts_name)
            ts.data.resize(new_ts.data.shape)
            ts.data[()] = new_ts.data


def load_nwb_ts(nwb_path, group_name, ts_name):
    """
    Create a new TimeSeries for data to write

    Parameters
    ----------
    config_filepath (Path): path to the config file
    group_name (str): name of group to write to
    ts_name (str): name of ts

    Returns
    -------

    """
    with NWBHDF5IO(nwb_path, 'a') as io:
        nwbfile = io.read()
        return np.moveaxis(nwbfile.processing['Face Rhythm'][group_name][ts_name].data[()],0,-1)

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
    save_dir = Path(config['Paths']['data'])
    save_path = save_dir / f'{save_name}.npy'
    np.save(save_path, data_to_save, allow_pickle=True)
    config['Paths'][save_name] = str(save_path)
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
    paths = config['Paths']
    save_dir = Path(paths['data'])
    save_path = save_dir / f'{save_name}.h5'
    to_write = h5py.File(save_path, 'w')
    dict_to_h5(data_dict, to_write)
    to_write.close()
    paths[save_name] = str(save_path)
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
    return h5_to_dict(config['Paths'][data_key])


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
    return np.load(config['Paths'][data_key], allow_pickle=True)


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

def dump_nwb(nwb_path):
    io = pynwb.NWBHDF5IO(nwb_path, 'r')
    nwbfile = io.read()
    for interface in nwbfile.processing['Face Rhythm'].data_interfaces:
        print(interface)
        time_series_list = list(nwbfile.processing['Face Rhythm'][interface].time_series.keys())
        for ii, time_series in enumerate(time_series_list):
            print(f"     {time_series}:    {nwbfile.processing['Face Rhythm'][interface][time_series].data.shape}   ,  {nwbfile.processing['Face Rhythm'][interface][time_series].data.dtype}")


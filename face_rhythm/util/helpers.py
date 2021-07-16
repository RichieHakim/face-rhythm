import yaml
import numpy as np

import h5py

from pathlib import Path

from pynwb import NWBHDF5IO
from hdmf.backends.hdf5.h5tools import H5DataIO
import pynwb
from pynwb.behavior import BehavioralTimeSeries

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def load_config(config_filepath):
    """Loads config file into memory
    
    Args:
        config_filepath (str): path to config file
    
    Returns:
        config (dict): actual config dict
    
    """
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, config_filepath):
    """
    Dumps config file to yaml
    Args:
        config (dict): config dict
        config_filepath (str): path to config file
    Returns:
    """
    serialized_object_string = ''
    try:
        serialized_object_string = yaml.safe_dump(config)
    except yaml.YAMLError as ex:
        print("An error has occurred while trying to save FR config:\n", ex)
    if serialized_object_string:
        with open(config_filepath, 'w') as f:
            f.write(serialized_object_string)


def create_nwb_group(nwb_path, group_name):
    """
    Create an NWB BehavioralTimeSeries for grouping data

    Args:
        config_filepath (Path): path to the config file
        group_name (str): name of group to be created

    Returns:

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

    Args:
        config_filepath (Path): path to the config file
        group_name (str): name of group to write to
        ts_name (str): name of new ts
        data (np.array): data to be written
        Fs (float): frequency of the data being written

    Returns:

    """
    print(f'Saving {ts_name} in Group {group_name}')
    with NWBHDF5IO(nwb_path, 'a') as io:
        nwbfile = io.read()
        maxshape = tuple(None for dim in data.shape)
        new_ts = pynwb.TimeSeries(name=ts_name,
                                  data=H5DataIO(data, maxshape=maxshape),
                                  unit='mm',
                                  rate=Fs)
        if ts_name not in nwbfile.processing['Face Rhythm'][group_name].time_series:
            nwbfile.processing['Face Rhythm'][group_name].add_timeseries(new_ts)
        else:
            ts = nwbfile.processing['Face Rhythm'][group_name].get_timeseries(ts_name)
            ts.data.resize(new_ts.data.shape)
            ts.data[()] = new_ts.data
        io.write(nwbfile)


def load_nwb_ts(nwb_path, group_name, ts_name):
    """
    Create a new TimeSeries for data to write

    Args:
        config_filepath (Path): path to the config file
        group_name (str): name of group to write to
        ts_name (str): name of ts

    Returns:

    """
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        return nwbfile.processing['Face Rhythm'][group_name][ts_name].data[()]


def save_data(config_filepath, save_name, data_to_save):
    """
    save an npy file with data

    Args:
        config_filepath (Path): path to the config file
        save_name (str): name of the object to be saved
        data_to_save (np.ndarray): (usually) a numpy array

    Returns:

    """
    config = load_config(config_filepath)
    save_dir = Path(config['Paths']['data'])
    save_path = save_dir / f'{save_name}.npy'
    np.save(save_path, data_to_save, allow_pickle=True)
    config['Paths'][save_name] = str(save_path)
    save_config(config, config_filepath)


def load_data(config_filepath, data_key):
    """
    load an npy file with data

    Args:
        config_filepath (Path): path to the config file
        data_key (str): config key for the target data

    Returns:
        data (np.ndarray): a np array with data

    """

    config = load_config(config_filepath)
    return np.load(config['Paths'][data_key], allow_pickle=True)


def save_h5(config_filepath, save_name, data_dict):
    """
    save an h5 file from a data dictionary

    Args:
        config_filepath (Path): path to the config file
        save_name (str): name of the object to be saved
        data_dict (dict): dict of numpy arrays

    Returns:

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

    Args:
        config_filepath (Path): path to the config file
        data_key (str): name of the data to be loaded

    Returns:

    """
    config = load_config(config_filepath)
    return h5_to_dict(config['Paths'][data_key])


def print_time(action, time):
    """
    prints the time adjusted for hours/minutes/seconds based on length

    Args:
        action (str): description of the completed action
        time (float): elapsed time

    Returns:

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
    """
    Reads all contents from h5 and returns them in a nested dict object.

    Args:
        h5file (str): path to h5 file
        path (str): path to group within h5 file

    Returns:
    ans (dict): dictionary of all h5 group contents
    """

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
    """
    Quick and dirty dict dumper to h5

    Args:
        data_dict (dict): dictionary (potentially nested) of data!
        h5 (h5py.File): h5 File (or Group) to populate

    Returns:
    """

    for key, item in data_dict.items():
        if isinstance(item, dict):
            group = h5.create_group(key)
            dict_to_h5(item, group)
        else:
            h5.create_dataset(key, data=item)


def dump_nwb(nwb_path):
    """
    Print out nwb contents

    Args:
        nwb_path (str): path to the nwb file

    Returns:
    """
    with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        for interface in nwbfile.processing['Face Rhythm'].data_interfaces:
            print(interface)
            time_series_list = list(nwbfile.processing['Face Rhythm'][interface].time_series.keys())
            for ii, time_series in enumerate(time_series_list):
                data_tmp = nwbfile.processing['Face Rhythm'][interface][time_series].data
                print(f"     {time_series}:    {data_tmp.shape}   ,  {data_tmp.dtype}   ,   {round((data_tmp.size * data_tmp.dtype.itemsize)/1000000000, 6)} GB")


def absolute_index(session, vid_num, iter_frame):
    return int(sum(session['vid_lens'][:vid_num]) + iter_frame)


def get_pts(nwb_path):
    pts_all = {}
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        for ts_name, ts in nwbfile.processing['Face Rhythm']['Original Points'].time_series.items():
            pts_all[ts_name] = ts.data[()]
    return pts_all

def save_pts(nwb_path, pts_all):
    create_nwb_group(nwb_path, 'Original Points')
    for point_name, points in pts_all.items():
        create_nwb_ts(nwb_path, 'Original Points', point_name, points, 1.0)


def update_config(new_project_path, config_name):
    config_filepath = str(Path(new_project_path) / 'configs' / ('config_' + config_name + '.yaml'))
    config = load_config(config_filepath)
    old_project_path = config['Paths']['project']
    for cat, cat_dict in config.items():
        for key, value in cat_dict.items():
            if type(value) is str and old_project_path in value:
                cat_dict[key] = value.replace(old_project_path, new_project_path)
    save_config(config, config_filepath)
    return config_filepath



##### MULTICORE HELPERS #####
def multithreading(func, args, workers):
    with ThreadPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    return list(res)
def multiprocessing(func, args, workers):
    with ProcessPoolExecutor(workers) as ex:
        res = ex.map(func, args)
    return list(res)
#############################

def estimate_size_of_float_array(numel=None, input_shape=None, bitsize=64):
    '''
    Estimates the size of a hypothetical array based on shape or number of 
    elements and the bitsize

    Args:
        numel (int): 
            number of elements in the array. If None, then 'input_shape'
            is used instead
        input_shape (tuple of ints):
            shape of array. Output of array.shape . Used if numel is None
        bitsize (int):
            bit size / width of the hypothetical data. eg:
                'float64'=64
                'float32'=32
                'uint8'=8
    
    Returns:
        size_estimate_in_bytes (int):
            size, in bytes, of hypothetical array. Doesn't include metadata,
            but for numpy arrays, this is usually very small (~128 bytes)

    '''

    if numel is None:
        numel = np.product(input_shape)
    
    bytes_per_element = bitsize/8
    
    size_estimate_in_bytes = numel * bytes_per_element
    return size_estimate_in_bytes
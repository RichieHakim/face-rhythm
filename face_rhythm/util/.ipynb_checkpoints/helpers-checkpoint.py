import ruamel.yaml as yaml
import cv2
import torch
import sys
import numpy as np

def version_check(config_filepath):
    '''
    Checks the versions of various important softwares.
    Prints those versions
    
    Parameters
    ----------
    config_filepath (Path): path to config file
    
    Returns
    -------
    
    '''
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
        print("What computer are you running this on? I haven't tested it on OSX or anything except windows and ubuntu.")
        print('Autodetected operating system: OSX. Using "/" for directory slashes')
    
    config = load_config(config_filepath)
    config['slash_type'] = slash_type
    save_config(config, config_filepath)

    
def load_config(config_filepath):
    '''
    Loads config file into memory
    
    Parameters
    ----------
    config_filepath (Path): path to config file
    
    Returns
    -------
    config (dict) : actual config dict
    
    '''
    with open(config_filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config

        
def save_config(config, config_filepath):
    '''
    Dumps config file to yaml
    
    Parameters
    ----------
    config (dict): config dict
    config_filepath (Path): path to config file
    
    Returns
    -------
    
    '''
    with open(config_filepath, 'w') as f:
        yaml.safe_dump(config, f)  
        
        
def generate_config(config_filepath):
    '''
    Generates bare config file with just basic info
    
    Parameters
    ----------
    config_filepath (Path): path to config file
    
    Returns
    -------
    
    '''
    
    basic_config = {'path_base_dir': str(config_filepath.parent),
                    'path_config' : str(config_filepath)}
    
    with open(config_filepath, 'w') as f:
        yaml.safe_dump(basic_config, f)
        
        
def import_videos(config_filepath):
    '''
    Define the directory of videos
    Import the videos as read objects
    
    Prints those versions
    
    Parameters
    ----------
    config_filepath (Path): path to the config file 
    
    Returns
    -------
    
    '''
    
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
            if os.path.isfile(os.path.join(dir_vid,ii)) and fileName_vid_prefix in ii:
                fileNames_allInPathWithPrefix.append(ii)
        numVids = len(fileNames_allInPathWithPrefix)

        ## make a variable containing all of the file paths
        path_vid_allFiles = list()
        for ii in range(numVids):
            path_vid_allFiles.append(f'{dir_vid}{slash_type}{fileNames_allInPathWithPrefix[ii]}')

    else: ## Single file import
        path_vid = f'{dir_vid}{slash_type}{fileName_vid}'
        path_vid_allFiles = list()
        path_vid_allFiles.append(path_vid)
        numVids = 1

    config['numVids'] = numVids
    path_vid_allFiles = sorted(path_vid_allFiles)
    
    config['path_vid_allFiles'] = path_vid_allFiles
    
    save_config(config, config_filepath)

    
def get_video_data(config_filepath):
    '''
    get info on the imported video(s): num of frames, video height and width, framerate
    
    Parameters
    ----------
    config_filepath (Path): path to the config file 
    
    Returns
    -------
    
    '''
    
    config = load_config(config_filepath)
    multiple_files_pref = config['multiple_files_pref']
    path_vid_allFiles = config['path_vid_allFiles']
    numVids  = config['numVids']
    print_fileNames_pref = config['print_fileNames_pref']
    
    if multiple_files_pref:
        path_vid = path_vid_allFiles[0]
        video = cv2.VideoCapture(path_vid_allFiles[0])
        numFrames_firstVid = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        numFrames_allFiles = np.ones(numVids) * np.nan # preallocation
        for ii in range(numVids):
            video = cv2.VideoCapture(path_vid_allFiles[ii])
            numFrames_allFiles[ii] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        numFrames_total_rough = np.uint64(sum(numFrames_allFiles))

        print(f'number of videos: {numVids}')
        print(f'number of frames in FIRST video (roughly):  {numFrames_firstVid}')
        print(f'number of frames in ALL videos (roughly):   {numFrames_total_rough}')
    else:
        video = cv2.VideoCapture(path_vid_allFiles[0])
        numFrames_onlyVid = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        numFrames_total_rough = numFrames_onlyVid
        numFrames_allFiles = numFrames_total_rough
        print(f'number of frames in ONLY video:   {numFrames_onlyVid}')
    
    config['numFrames_total_rough'] = numFrames_total_rough
    
    Fs = video.get(cv2.CAP_PROP_FPS) ## Sampling rate (FPS). Manually change here if necessary
    print(f'Sampling rate pulled from video file metadata:   {round(Fs,3)} frames per second')
    config['vid_Fs'] = Fs
    
    if print_fileNames_pref:
        print(f'\n {np.array(path_vid_allFiles).transpose()}')

    video.set(1,1)
    ok, frame = video.read()
    vid_height = frame.shape[0]
    vid_width = frame.shape[1]

    config['numFrames_allFiles'] = numFrames_allFiles
    config['vid_height'] = vid_height
    config['vid_width'] = vid_width
    
    save_config(config, config_filepath)


def save_data(config_filepath, save_name, data_to_save):
    config = load_config(config_filepath)
    save_dir = config['save_dir']
    save_path = f'{save_dir}/{save_name}.npy'
    np.save(save_path, data_to_save, allow_pickle=True)
    config[f'path_{save_name}'] = save_path
    save_config(config, config_filepath)


def load_data(config_filepath, data_key):
    config = load_config(config_filepath)
    return np.load(config[data_key], allow_pickle=True)

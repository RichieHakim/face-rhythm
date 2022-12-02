from face_rhythm.util import helpers, setup

from pathlib import Path
import shutil


# def run_basic(run_name):
#     project_path = Path('test_runs').resolve() / run_name
#     video_path = Path('test_data').resolve() / run_name / 'session1'
#     overwrite_config = True
#     remote = True
#     trials = False
#     multisession = False
#
#     config_filepath = setup.setup_project(project_path, video_path, run_name, overwrite_config, remote, trials,
#                                           multisession)
#     return config_filepath
#
#
# def test_config_update():
#     run_name = 'single_session_single_video'
#     config_filepath = run_basic(run_name)
#     config = helpers.load_config(config_filepath)
#     old_project_path = config['Paths']['project']
#     new_project_path = str(Path(old_project_path).parent / 'test')
#     shutil.copytree(old_project_path, new_project_path)
#     new_config_filepath = helpers.update_config(new_project_path, run_name)


    # Cleanup
    #new_config = helpers.load_config(new_config_filepath)
    #shutil.rmtree(new_config['Paths']['project'])


# def test_create_nwb_group():
#     nwb_path = Path('data') / 'nwbs' / 'nwb_silver.nwb'
#     group_name = 'Gold'
#     helpers.create_nwb_group(nwb_path,group_name)
#     with NWBHDF5IO(nwb_path, 'r') as io:
#         nwbfile = io.read()
# #         assert group_name in nwbfile.processing['Face Rhythm'].data_interfaces.keys()
# #
# def test_create_nwb_ts(nwb_path, group_name, ts_name, data, Fs):
#     nwb_path = Path('data') / 'nwbs' / 'nwb_silver.nwb'
#     group_name = 'Gold'
#     ts_name = 'test'
#     data = []
#     Fs = 1
#     helpers.create_nwb_ts(nwb_path, group_name, ts_name, data, Fs)
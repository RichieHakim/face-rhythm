from pathlib import Path

from pynwb import NWBHDF5IO

from face_rhythm.util import  helpers


# def test_create_nwb_group():
#     nwb_path = Path('data') / 'nwbs' / 'nwb_silver.nwb'
#     group_name = 'Gold'
#     helpers.create_nwb_group(nwb_path,group_name)
#     with NWBHDF5IO(nwb_path, 'r') as io:
#         nwbfile = io.read()
#         assert group_name in nwbfile.processing['Face Rhythm'].data_interfaces.keys()
#
# def test_create_nwb_ts(nwb_path, group_name, ts_name, data, Fs):
#     nwb_path = Path('data') / 'nwbs' / 'nwb_silver.nwb'
#     group_name = 'Gold'
#     ts_name = 'test'
#     data = []
#     Fs = 1
#     helpers.create_nwb_ts(nwb_path, group_name, ts_name, data, Fs)
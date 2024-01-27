
########################################
## Import parameters from CLI
########################################

import os
print(f"script environment: {os.environ['CONDA_DEFAULT_ENV']}")


## Argparse --path_params, --directory_save
import argparse
parser = argparse.ArgumentParser(
    prog='Face-Rhythm Basic Pipeline',
    description='This script runs the basic pipeline using a json file containing the parameters.',
)
parser.add_argument(
    '--path_params',
    '-p',
    required=True,
    metavar='',
    type=str,
    default=None,
    help='Path to json file containing parameters.',
)
parser.add_argument(
    '--directory_save',
    '-d',
    required=False,
    metavar='',
    type=str,
    default=None,
    help="Directory to use as 'directory_project' and save results to. Overrides 'directory_project' field in parameters file.",
)
args = parser.parse_args()
path_params = args.path_params
directory_save = args.directory_save



## Checks for path_params and directory_save
from pathlib import Path

## Check path_params
### Check if path_params is valid
assert Path(path_params).exists(), f"Path to parameters file does not exist: {path_params}"
### Check if path is absolute. If not, convert to absolute path.
if not Path(path_params).is_absolute():
    path_params = Path(path_params).resolve()
    print(f"Warning: Input path_params is not absolute. Converted to absolute path: {path_params}")
### Warn if suffix is not json
print(f"Warning: suffix of path_params is not .json: {path_params}") if Path(path_params).suffix != '.json' else None
print(f"path_params: {path_params}")

## Check directory_save
### Check if directory_save is valid
if args.directory_save is not None:
    assert Path(args.directory_save).exists(), f"Path to directory_save does not exist: {args.directory_save}"
    ### Check if directory_save is absolute. If not, convert to absolute path.
    if not Path(args.directory_save).is_absolute():
        args.directory_save = Path(args.directory_save).resolve()
        print(f"Warning: Input directory_save is not absolute. Converted to absolute path: {args.directory_save}")
    ### Check that directory_save is a directory
    assert Path(args.directory_save).is_dir(), f"Input directory_save is not a directory: {args.directory_save}"
    ### Set directory_save
    print(f"directory_save: {directory_save}")

## Load parameters
import json
with open(path_params, 'r') as f:
    params = json.load(f)


if directory_save is not None:
    params['project']['directory_project'] = directory_save


## START THE JOB
import face_rhythm as fr
import time

tic_start = time.time()

## Run the pipeline
fr.pipelines.pipeline_basic(params)

print(f"Project directory: {params['project']['directory_project']}")
print(f'Time elapsed: {time.time() - tic_start:.2f} seconds')

## End the job and kill the kernel
os._exit(0)
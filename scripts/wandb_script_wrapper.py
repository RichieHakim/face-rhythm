#!/usr/bin/env python
"""
WandB Script Wrapper

This script is intended to serve as a wrapper to execute an existing target Python script
without any modifications, while capturing its standard output, standard error, and system 
metrics (CPU, memory, and GPU) with Weights & Biases (WandB). It leverages WandB’s 
built-in GPU monitoring (via monitor_gpus=True) to automatically capture GPU metrics, and 
streams subprocess output (stdout and stderr) to the WandB dashboard in real time.

How It Works:
  1. Parses command-line arguments, expecting two positional arguments:
       - path_params: Path to a JSON file containing parameters for the target script. This
         JSON must include at minimum:
           * "path_script": String path to the target script to be executed.
           * "kwargs_wandb_init": (Optional) A dictionary of keyword arguments for 
             wandb.init().
           * "params_script" can be provided, which will be saved as a JSON to the output
             directory and --path_params will be passed to the target script.
       - directory_save: Directory path where output files (such as logs and saved parameters)
         will be stored. Will be passed to the target script as --directory_save.

  2. Loads the parameters from the given JSON file and ensures that the target script exists.
  3. Initializes a new WandB run using wandb.init(), passing in any provided keyword arguments,
     and enables GPU monitoring automatically.
  4. Constructs a command to execute the target script with the necessary arguments (including 
     the saved parameters file) and starts it as a subprocess.
  5. Concurrently streams the subprocess's stdout and stderr to WandB (and also prints to the 
     console), logging each output line with its timestamp.
  6. Periodically (default every 30 seconds) collects and logs CPU and memory usage using 
     psutil.
  7. Once the target script completes, it finalizes the WandB run and exits with the same exit 
     code as the target process.

Example Usage:
  Suppose you have a parameter file at "/path/to/params.json" and you want to save outputs to 
  "/path/to/save_dir". The parameter JSON file should include entries like:

    {
        "path_script": "/path/to/your_target_script.py",
        "kwargs_wandb_init": {
            "project": "face_rhythm",
            "entity": "your_wandb_username",
            "name": "example_run"
        },
        "params_script": {
            "example_param": "value"
        }
    }

  Then call the wrapper as follows:

      python wandb_script_wrapper.py --path_params /path/to/params.json --directory_save /path/to/save_dir

No changes are required in your core script. The wrapper automatically handles logging and monitoring using WandB.
The core script should accept command-line arguments for --path_params and --directory_save.
"""

import sys
import subprocess
import threading
import time
import wandb
import psutil

def stream_reader(pipe, log_label):
    """
    Reads lines from a subprocess pipe, prints them to the console,
    and logs each line to WandB under the specified label.

    Args:
        pipe (IO): File-like stream (stdout or stderr) of the subprocess.
        log_label (str): Label under which to log the output (e.g., "stdout" or "stderr").
    """
    for line in iter(pipe.readline, ''):
        if line:
            # Echo output to the console.
            print(line, end='')
            # Log the output line to WandB with the current time as the step.
            wandb.log({log_label: line.strip()}, step=int(time.time()))
    pipe.close()

def monitor_system_metrics(interval: int = 30):
    """
    Periodically logs system metrics (CPU and memory usage) to WandB.

    Args:
        interval (int, optional): Interval in seconds between metric logs. Defaults to 30.
    """
    while target_process.poll() is None:
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
        }
        wandb.log(metrics, step=int(time.time()))
        time.sleep(interval)

if __name__ == "__main__":
    import argparse
    import os
    
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="WandB Script Wrapper")
    ## 'path_params' is sys.argv[1]
    parser.add_argument("path_params", type=str, help="Path to the JSON file containing parameters for the target script. Include a field 'kwargs_wandb_init' for WandB initialization.")
    ## 'directory_save' is sys.argv[2]
    parser.add_argument("directory_save", type=str, help="Directory to save the output of the target script.")
    args = parser.parse_args()
    path_params = args.path_params
    directory_save = args.directory_save
    
    # Get params from the JSON file.
    import json
    with open(path_params, 'r') as f:
        params = json.load(f)

    # Gather kwargs_wandb_init from the JSON file.
    kwargs_wandb_init = params.get('kwargs_wandb_init', None)
    # Gather path_script from the JSON file. Error if missing
    path_script = params.get('path_script', None)
    if path_script is None:
        print("Error: 'path_script' is missing in the parameters file.")
        sys.exit(1)
    # Ensure the target script exists.
    if not os.path.isfile(path_script):
        print(f"Error: The target script '{path_script}' does not exist.")
        sys.exit(1)
    
    # Save params['params_script'] as a json file in the directory_save.
    params_script = params.get('params_script', None)
    if params_script is not None:
        path_params_script = os.path.join(directory_save, 'params_script.json')
        with open(path_params_script, 'w') as f:
            json.dump(params_script, f)
    else:
        print("Warning: 'params_script' is not provided in the parameters file. Skipping saving parameters.")
    
    # Ensure WandB is installed.
    try:
        import wandb
    except ImportError:
        print("Error: WandB is not installed. Please install it using 'pip install wandb'.")
        sys.exit(1)
            
    # Initialize WandB with the provided kwargs.
    if kwargs_wandb_init:
        wandb.init(**kwargs_wandb_init)
    else:
        wandb.init()

    # Make command to run the target script.
    command = ["python", path_script, "--path_params", path_params_script, "--directory_save", directory_save]
    # Add any additional arguments from the command line.
    # command.extend(sys.argv[1:])  ## This is dangerous but could be useful.

    # Start the target script as a subprocess.
    target_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Create threads to stream stdout and stderr concurrently.
    stdout_thread = threading.Thread(target=stream_reader, args=(target_process.stdout, "stdout"), daemon=True)
    stderr_thread = threading.Thread(target=stream_reader, args=(target_process.stderr, "stderr"), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    # Create a thread to monitor and log system metrics.
    metrics_thread = threading.Thread(target=monitor_system_metrics, daemon=True)
    metrics_thread.start()

    # Wait for the target process to complete and for threads to finish.
    target_process.wait()
    stdout_thread.join()
    stderr_thread.join()

    # Finalize the WandB run.
    wandb.finish()
    sys.exit(target_process.returncode)
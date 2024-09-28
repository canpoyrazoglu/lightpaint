import os
import subprocess
import multiprocessing
import itertools
import argparse
from tqdm import tqdm
import sys

def run_process_video(python_interpreter, input_file, output_folder, blend_mode, power, decay_param, script_path, process_idx, progress):
    """
    Function to invoke the original script with the specified parameters, using the same Python interpreter.
    """
    command = [
        python_interpreter, script_path,  # Use the same Python interpreter that was used to run this script
        '-i', input_file,                 # Input video file
        '-o', output_folder,              # Output folder to save files
        '-bm', blend_mode,                # Blend mode
        '-pw', str(power),                # Power
        '-dp', str(decay_param),          # Decay parameter
    ]
    
    # Run the command
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Mark the process as done in the shared progress variable
    progress[process_idx] = True

def track_progress(n_processes, total_combinations, progress):
    """
    This function will manage the master progress bar based on when processes complete.
    """
    master_bar = tqdm(total=total_combinations, position=n_processes, leave=True, desc="Master Progress", unit="task")

    completed_tasks = 0

    while completed_tasks < total_combinations:
        # Check progress for each process
        for i in range(n_processes):
            if progress[i]:  # If the process is done
                completed_tasks += 1
                master_bar.update(1)  # Update the master bar
                progress[i] = False  # Reset the flag for the next task

    master_bar.close()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Batch orchestrator for running process_video.py multiple times in parallel.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file.")
    parser.add_argument("-o", "--output_folder", default="video_output", help="Directory to save the output videos.")
    parser.add_argument("-bm", "--blend_modes", required=True, help="Comma-separated list of blend modes (e.g., lighten,add,screen).")
    parser.add_argument("-pw", "--powers", required=True, help="Comma-separated list of power values (e.g., 1,1.2,1.5,2).")
    parser.add_argument("-dp", "--decay_params", required=True, help="Comma-separated list of decay parameters (e.g., 0.6,0.7,0.8,0.9,0.95,0.98).")
    parser.add_argument("-sc", "--script_path", default="process_video.py", help="Path to the process_video.py script (default: process_video.py).")

    args = parser.parse_args()

    # Manually split the comma-separated inputs
    blend_modes = args.blend_modes.split(',')
    powers = [float(pw) for pw in args.powers.split(',')]
    decay_params = [float(dp) for dp in args.decay_params.split(',')]

    # Get the Python interpreter that was used to run this script
    python_interpreter = sys.executable

    # Get the number of available CPU cores
    cpu_count = multiprocessing.cpu_count()

    # Create all combinations of blend modes, powers, and decay parameters
    combinations = list(itertools.product(blend_modes, powers, decay_params))
    total_combinations = len(combinations)

    print(f"Starting {total_combinations} runs with {cpu_count} parallel processes...")

    # Use a multiprocessing Manager to create a shared progress variable
    with multiprocessing.Manager() as manager:
        progress = manager.list([False] * cpu_count)  # Initialize progress flags for each process

        # Create a separate process for tracking progress
        progress_tracker = multiprocessing.Process(target=track_progress, args=(cpu_count, total_combinations, progress))
        progress_tracker.start()

        # Set up multiprocessing pool to run commands in parallel
        with multiprocessing.Pool(cpu_count) as pool:
            # Submit tasks for each combination, using the correct Python interpreter
            pool.starmap(run_process_video, [
                (python_interpreter, args.input, args.output_folder, blend_mode, power, decay_param, args.script_path, i % cpu_count, progress)
                for i, (blend_mode, power, decay_param) in enumerate(combinations)
            ])

        # Ensure the progress tracker finishes
        progress_tracker.join()

    print("All tasks completed.")

if __name__ == "__main__":
    main()

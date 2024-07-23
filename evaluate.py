import os
import subprocess
import logging
import time

# Define the datasets and GNN layers to test
# datasets = ['MOLT-4', 'SW-620', 'NCI-H23', 'OVCAR-8', 'P388', 'SF-295', 'SN12C', 'UACC257', 'NCI1']
datasets = ['MOLT-4']


def get_repo_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isdir(os.path.join(current_dir, '.git')):
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not find the root directory of the repository")
        current_dir = parent_dir
    return current_dir

def milliseconds_to_seconds(milliseconds):
    seconds = milliseconds / 1000
    return f"{seconds:.2f} seconds"

def log_data(text):
    print(text)
    logging.info(text)

def log_time(start_time, action_start_time, message):
    current_time = time.time()
    elapsed_time_since_start = (current_time - start_time) * 1000  # Convert to milliseconds
    elapsed_time_since_last = (current_time - action_start_time) * 1000  # Convert to milliseconds
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_data(f"[TIMER: {current_time_str} | {message} | Time_taken_since_start: {elapsed_time_since_start:.2f} ms ({milliseconds_to_seconds(elapsed_time_since_start)}) | Time_taken_since_last: {elapsed_time_since_last:.2f} ms ({milliseconds_to_seconds(elapsed_time_since_last)})]")



# Run the experiments
def run_experiments():
    experiment_counter = 0
    for dataset in datasets:
            if experiment_counter > 0:
                log_data("Waiting for 15 seconds before the next experiment...")
                time.sleep(15)

            experiment_name = f"[Experiment {experiment_counter + 1} - Dataset: {dataset}]"
            log_data(experiment_name)
            
            # Construct the command to run main.py
            # python main.py --data PC-3 --lr 5e-3 --batchsize 512 --nepoch 10 --hdim 64 --width 4 
            # --depth 6 --dropout 0.4 --normalize 1 --beta 0.999 --gamma 1.5 --decay 0 --seed 10 --patience 50

            command = [
                'python', 'main.py',
                '--data', dataset,
                '--lr', "5e-3",
                '--batchsize', "512",
                '--nepoch', "10",
                '--hdim', "64",
                '--width', "4",
                '--depth', "6",
                '--dropout', "0.4",
                '--normalize', "1",
                '--beta', "0.999",
                '--gamma', "1.5",
                '--decay', "0",
                '--seed', "10",
                '--patience', "50",
            ]
            
            log_data(f"Running command: for dataset {dataset}")
            
            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)
            
            # log_data the output and errors
            log_data(result.stdout)
            if result.stderr:
                log_data(f"Error: {result.stderr}")

            log_data(f"[End of {experiment_name}]")
            experiment_counter += 1

if __name__ == "__main__":
    run_experiments()
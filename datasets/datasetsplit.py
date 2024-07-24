import os
import subprocess
import logging
import time
from torch_geometric.datasets import TUDataset
import shutil

# Define the datasets and GNN layers to test
# datasets = ['MCF-7', 'MOLT-4', 'SW-620', 'NCI-H23', 'OVCAR-8', 'P388', 'SF-295', 'SN12C', 'UACC257', 'NCI1']
datasets = ['MCF-7', 'NC1-H23', "P388"]
trainsize = "0.7"
testsize = "0.15"


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


# def run_tud():
#     datasets_path = f'{get_repo_root()}/datasets'
#     dataset_path = f'{datasets_path}/{'AIDS'}'
#     data = TUDataset(root=datasets_path, name=f'{'AIDS'}')
#     raw_path = os.path.join(dataset_path, 'raw')
#     processed_path = os.path.join(dataset_path, 'processed')
#     for filename in os.listdir(raw_path):
#         shutil.move(os.path.join(raw_path, filename), dataset_path)
#     shutil.rmtree(raw_path)
#     shutil.rmtree(processed_path)

# Run the experiments
def run_experiments():
    experiment_counter = 0
    for dataset in datasets:
            if experiment_counter > 0:
                log_data("Waiting for 15 seconds before the next experiment...")
                time.sleep(15)

            experiment_name = f"[Experiment {experiment_counter + 1} - Dataset: {dataset}]"
            log_data(experiment_name)

            datasets_path = f'{get_repo_root()}/datasets'
            dataset_path = f'{datasets_path}/{dataset}'
            if not os.path.isdir(dataset_path):
                data = TUDataset(root=datasets_path, name=f'{dataset}')
                raw_path = os.path.join(dataset_path, 'raw')
                processed_path = os.path.join(dataset_path, 'processed')
                for filename in os.listdir(raw_path):
                    shutil.move(os.path.join(raw_path, filename), dataset_path)
                shutil.rmtree(raw_path)
                shutil.rmtree(processed_path)

            
            # Construct the command to run main.py
            command = [
                'python', 'dataset.py',
                '--data', dataset,
                '--trainsz', trainsize,
                '--testsz', testsize,
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

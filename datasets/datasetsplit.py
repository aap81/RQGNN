import os
import subprocess
import logging
import time
from torch_geometric.datasets import TUDataset
import urllib.request
import shutil
import zipfile

# Define the datasets and GNN layers to test
DATASETS = [
    "MOLT-4" #, "SW-620", "NCI-H23", "OVCAR-8", "P388", "SF-295", "SN12C", "UACC257", "PC-3", "MCF-7", "PROTEINS", "AIDS", "Mutagenicity", "NCI109", "NCI1", "DD"
]
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


def download_and_extract_tu_dataset(datasets_path, dataset_name):
    dataset_url = f'https://www.chrsmrrs.com/graphkerneldatasets/{dataset_name}.zip'
    dataset_zip = os.path.join(datasets_path, f'{dataset_name}.zip')
    dataset_dir = os.path.join(datasets_path, dataset_name)
    raw_dir = os.path.join(dataset_dir, 'raw')

    if not os.path.exists(raw_dir):
        # Create directory if it doesn't exist
        os.makedirs(datasets_path, exist_ok=True)

        # Download the dataset
        print(f'Downloading {dataset_name} dataset...')
        urllib.request.urlretrieve(dataset_url, dataset_zip)

        # Extract the dataset
        print(f'Extracting {dataset_name} dataset...')
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(datasets_path)

        # Create the raw directory if it doesn't exist
        os.makedirs(raw_dir, exist_ok=True)

        # Move the extracted files into the raw directory
        for filename in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, filename)
            if os.path.isfile(file_path):
                shutil.move(file_path, raw_dir)

        # Clean up the zip file
        os.remove(dataset_zip)


def run_experiments():
    experiment_counter = 0
    for dataset in DATASETS:
            experiment_name = f"[Experiment {experiment_counter + 1} - Dataset: {dataset}]"
            log_data(experiment_name)

            datasets_path = f'{get_repo_root()}/datasets'
            dataset_path = f'{datasets_path}/{dataset}'
            if not os.path.isdir(dataset_path):
                print(f"Adding {dataset} folder to {dataset_path}")
                download_and_extract_tu_dataset(datasets_path, dataset)
            else:
                print(f"Loading existing {dataset} data")
            
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

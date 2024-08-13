import os
import logging
import time
import argparse
import json

import datautils
from name import *
from datautils import download_and_process_tu_dataset, get_repo_root, split_graph_dataset

def split_all_datasets(train_ratio, test_ratio):
    print(f"Splitting all listed datasets: {DATASETS}")
    datasets_processed = 0
    dataset_count = len(DATASETS) 
    for dataset in DATASETS:
        start = time.time()
        print(f"\n{datasets_processed + 1}/{dataset_count} - Dataset: {dataset}")
        datasets_path = f"{get_repo_root()}/datasets"
        graphs = download_and_process_tu_dataset(datasets_path, dataset)
        split_graph_dataset(graphs, datasets_path, dataset, train_ratio, test_ratio)
        end_time = time.time()
        print(f"Processed: {dataset}, time_cost: {(end_time - start)}")
        datasets_processed += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2023, help='seed')
    parser.add_argument('--trainsz', type=float, default=0.7, help='train size')
    parser.add_argument('--testsz', type=float, default=0.15, help='test size')
    args = parser.parse_args()

    seed = args.seed
    trainsz = args.trainsz
    testsz = args.testsz

    assert trainsz + testsz < 1

    datautils.set_seed(seed)
    print("Generator info:")
    print(json.dumps(args.__dict__, indent='\t'))

    start = time.time()
    split_all_datasets(trainsz, testsz)
    end_time = time.time()

    print("\nGenerate successfully, time cost: {}".format(end_time - start))

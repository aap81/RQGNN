import argparse
import time
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from evaluate import get_repo_root
import pdb
import os
import subprocess
import logging
import time
from torch_geometric.datasets import TUDataset
import urllib.request
import shutil
import zipfile

import utils
import model
from name import *
import lossfunc
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='PROTEINS', help='Dataset used')
args = parser.parse_args()


data = args.data

DATASETS = [
    "MOLT-4" , "SW-620", "NCI-H23", "OVCAR-8", "P388", "SF-295", "SN12C", "UACC257", "PC-3", "MCF-7", "PROTEINS", "AIDS", "Mutagenicity", "NCI109", "NCI1", "DD"
]
for data in DATASETS:
	print(f"python main.py --data {data} --lr 5e-3 --batchsize 512 --nepoch 20 --hdim 64 --width 4 --depth 6 --dropout 0.4 --normalize 1 --beta 0.999 --gamma 1.5 --decay 0 --seed 10 --patience 50")
	print("\n")
exit()
featuredim = graphs.num_features

train_graphs = [graphs[i] for i in train_index]
val_graphs = [graphs[i] for i in val_index]
test_graphs = [graphs[i] for i in test_index]

batch_size = 32  # Adjust batch size based on your needs
dataloader = DataLoader(graphs, batch_size=32, shuffle=False)

# Initialize the Graph2Vec model
model = model.Graph2Vec(graphs.num_features, 64, 128, "mean")

all_embeddings = []

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for batch in dataloader:
        # Generate graph embeddings
        embeddings = model(batch)
        
        # Collect the embeddings
        all_embeddings.append(embeddings)

# Concatenate all embeddings into a single tensor
all_embeddings = torch.cat(all_embeddings, dim=0)
pdb.set_trace()
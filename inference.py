import os
import schnetpack as spk
import schnetpack.transform as trn
from ase import Atoms


import torch
import torchmetrics
import pytorch_lightning as pl
from ase.db import connect
import random

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np

# random_number = random.randint(1, 1500)
# random_number = 1

db_file = "new_dataset.db"
split_file = "hl_gap_dataset/split.npz"

db = db = connect(db_file)

best_model = torch.load(os.path.join("hl_gap_dataset", 'best_inference_model'), map_location="cpu")
best_model.to('cuda:0')

split = np.load(split_file)
test_idx = split['test_idx']

converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=100.), dtype=torch.float32)

# Create DataFrame with four columns
df = pd.DataFrame(columns=['mass', 'error', 'predicted', 'actual'])

for i in  tqdm(test_idx.tolist()):

    r = db.get(i)

    numbers = r.numbers
    positions = r.positions

    atoms = Atoms(numbers=numbers, positions=positions)
    inputs = converter(atoms)

    for key, value in inputs.items():
        inputs[key] = value.to('cuda')

    with torch.no_grad():
        pred = best_model(inputs)

    # print('Pred:', pred["homo"].item())
    # print("Real:", r.data.homo[0], r.data.lumo[0])
    # print('Pred:', pred["hl_gap"].item())
    # print("Real:", r.data.hl_gap[0])
    # mass = r.mass
    # error = pred["hl_gap"].item() - r.data.hl_gap[0]
    mass = r.mass
    error = pred["hl_gap"].item() - r.data.hl_gap[0]
    predicted = pred["hl_gap"].item()
    actual = r.data.hl_gap[0]
    df = pd.concat([df, pd.DataFrame({'mass': [mass], 'error': [error], 'predicted': [predicted], 'actual': [actual]})])

# Write DataFrame to CSV
df.to_csv('data.csv', index=False)

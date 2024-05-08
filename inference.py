import os
import schnetpack as spk
import schnetpack.transform as trn
from ase import Atoms


import torch
import torchmetrics
import pytorch_lightning as pl
from ase.db import connect
import random

random_number = random.randint(1, 1500)
# random_number = 1

db_file = "new_dataset.db"
db = db = connect(db_file)
best_model = torch.load(os.path.join("hl_gap_dataset", 'best_inference_model'), map_location="cpu")

converter = spk.interfaces.AtomsConverter(neighbor_list=trn.ASENeighborList(cutoff=100.), dtype=torch.float32)

r = db.get(random_number)


numbers = r.numbers
positions = r.positions

atoms = Atoms(numbers=numbers, positions=positions)
inputs = converter(atoms)

with torch.no_grad():
    pred = best_model(inputs)

# print('Pred:', pred["homo"].item())
# print("Real:", r.data.homo[0], r.data.lumo[0])
print('Pred:', pred["hl_gap"].item())
print("Real:", r.data.hl_gap[0])

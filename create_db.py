from ase import Atoms
from schnetpack.data import ASEAtomsData
import pickle
import numpy as np

# load atoms from npz file. Here, we only parse the first 10 molecules
# data = np.load('./data.npz', allow_pickle=True)
with open('./data.pkl', 'rb') as f:
    data = pickle.load(f)

atoms_list = []
property_list = []
for atom in data:
    ats = Atoms(positions=atom["coords"], numbers=atom["numbers"])
    # properties = {'homo': np.array([atom["homo"]]), 'lumo': np.array([atom["lumo"]])}
    properties = {'hl_gap': np.array([atom["homo"] - atom["lumo"]])}
    property_list.append(properties)
    atoms_list.append(ats)

new_dataset = ASEAtomsData.create(
    './new_dataset.db',
    distance_unit='Ang',
    # property_unit_dict={'homo':'eV', 'lumo':'eV'}
    property_unit_dict={'hl_gap': 'eV'}
)

new_dataset.add_systems(property_list, atoms_list)
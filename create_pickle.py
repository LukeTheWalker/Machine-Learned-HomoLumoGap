import numpy as np
import re
import os
import json
import pickle

atomic_numbers_map = {
    'H': 1,
    'C': 6,
    'B': 5,
    'N': 7
}

hl_gap_file = "hl_cache"

def get_hl_dictionary(hl_gap_file):
    hl_dictionary = {}
    with open(hl_gap_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            file, homo, lumo, gap = line.split(',')
            hl_dictionary[file] = {
                'homo': float(homo),
                'lumo': float(lumo),
                'hl_gap':float(gap)
            }
        return hl_dictionary

def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = lines[2:]
        coords = []
        atomic_numbers = []
        for line in lines:
            splitted = line.split()
            # print(splitted)
            atom = splitted[0]
            x = splitted[1]
            y = splitted[2]
            z = splitted[3]
            coords.append([float(x), float(y), float(z)])
            atomic_numbers.append(atomic_numbers_map[atom])
    return atomic_numbers, coords


def main():
    data = []
    molecule_folder = "molecole"
    hl_cache_file = "hl_cache.txt"
    hl_dict = get_hl_dictionary(hl_cache_file)
    for isomero_folder in os.listdir(molecule_folder):
        for file_name in os.listdir(os.path.join(molecule_folder, isomero_folder)):
            file_complete = os.path.join(molecule_folder, isomero_folder, file_name)
            print(f"Analyzing {file_complete}")
            atomic_numbers, coords = parse_file(file_complete)
            hl_id = f"{isomero_folder}_{file_name}"
            data.append({
                "numbers": atomic_numbers,
                "coords": coords,
                "homo": hl_dict[hl_id]["homo"],
                "lumo": hl_dict[hl_id]["lumo"]
            }) 

    # np.savez('data.npz', *data)
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    # print(json.dumps(get_hl_dictionary("hl_cache.txt"), sort_keys=True, indent=4))
    main()
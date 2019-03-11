import os
import pickle
import sys

import numpy as np


def create_atoms(atoms):
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_distances(coordinates):
    distances = [[np.linalg.norm(r_i - r_j) if i != j else 1e6
                 for j, r_j in enumerate(coordinates)]
                 for i, r_i in enumerate(coordinates)]
    return np.array(distances)


if __name__ == "__main__":

    DATASET = sys.argv[1:][0]

    with open('../dataset/' + DATASET + '/original/data.txt', 'r') as f:
        property_list = f.readline().strip().split()
        data_list = f.read().split('\n\n')
    N = len(data_list)

    with open('../dataset/atoms.txt', 'r') as f:
        atoms = f.read().strip().split()
    atom_dict = dict(zip(atoms, range(len(atoms))))

    Smiles, molecules, Distances, Properties = '', [], [], []

    for no, data in enumerate(data_list):

        print('/'.join(map(str, [no+1, N])))

        data = data.strip().split('\n')

        smiles = data[0]
        Smiles += smiles + '\n'

        atoms, coordinates = [], []
        for atom_xyz in data[1:-1]:
            atom, x, y, z = atom_xyz.split()
            atoms.append(atom)
            coordinates.append([float(v) for v in [x, y, z]])
        coordinates = np.array(coordinates)

        atoms = create_atoms(atoms)
        molecules.append(atoms)

        distances = create_distances(coordinates)
        Distances.append(distances)

        properties = [float(d) for d in data[-1].split()]
        Properties.append(properties)

    """Normalize properties (i.e., mean 0 and std 1)."""
    Properties = np.array(Properties)
    means, stds = np.mean(Properties, 0), np.std(Properties, 0)
    Properties = np.array((Properties - means) / stds)

    dir_input = '../dataset/' + DATASET + '/input/'
    os.makedirs(dir_input, exist_ok=True)

    with open(dir_input + 'Smiles.txt', 'w') as f:
        f.write(Smiles)
    np.save(dir_input + 'molecules', molecules)
    np.save(dir_input + 'Distances', Distances)
    with open(dir_input + 'atom_dict.pickle', 'wb') as f:
        pickle.dump(atom_dict, f)

    for p, P, m, s in zip(property_list, Properties.T, means, stds):
        np.save(dir_input + p, np.expand_dims(np.expand_dims(P, 1), 1))
        np.save(dir_input + p + '_mean', m)
        np.save(dir_input + p + '_std', s)

    print('The preprocess has finished!')

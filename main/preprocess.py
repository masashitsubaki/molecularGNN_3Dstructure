from collections import defaultdict

import numpy as np

from scipy import spatial

import torch


def create_atoms(atoms, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    """
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_distances(coords):
    """Create the distance matrix from a set of 3D coordinates.
    Note that we transform the element 0.0 in the matrix into a large value
    for processing by Gaussian exp(-d^2), where d is the distance.
    """
    distance_matrix = spatial.distance_matrix(coords, coords)
    return np.where(distance_matrix == 0.0, 1e6, distance_matrix)


def split_dataset(dataset, ratio):
    """Shuffle and split a dataset."""
    np.random.seed(1234)  # fix the seed for shuffle.
    np.random.shuffle(dataset)
    n = int(ratio * len(dataset))
    return dataset[:n], dataset[n:]


def create_datasets(dataset, physical_property, device):

    dir_dataset = '../dataset/' + dataset + '/'

    """Initialize atom_dict, in which
    each key is an atom type and each value is its index.
    """
    atom_dict = defaultdict(lambda: len(atom_dict))

    def create_dataset(filename):

        print(filename)

        """Load a dataset."""
        with open(dir_dataset + filename, 'r') as f:
            property_types = f.readline().strip().split()
            data_original = f.read().strip().split('\n\n')

        """The physical_property is an energy, HOMO, or LUMO."""
        property_index = property_types.index(physical_property)

        dataset = []

        for data in data_original:

            data = data.strip().split('\n')
            idx = data[0]
            property = float(data[-1].split()[property_index])

            """Load the atoms and their coordinates of a molecular data."""
            atoms, atom_coords = [], []
            for atom_xyz in data[1:-1]:
                atom, x, y, z = atom_xyz.split()
                atoms.append(atom)
                xyz = [float(v) for v in [x, y, z]]
                atom_coords.append(xyz)

            """Create each data with the above defined functions."""
            atoms = create_atoms(atoms, atom_dict)
            distance_matrix = create_distances(atom_coords)
            molecular_size = len(atoms)

            """Transform the above each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            """
            atoms = torch.LongTensor(atoms).to(device)
            distance_matrix = torch.FloatTensor(distance_matrix).to(device)
            property = torch.FloatTensor([[property]]).to(device)

            dataset.append((atoms, distance_matrix, molecular_size, property))

        return dataset

    dataset_train = create_dataset('data_train.txt')
    dataset_train, dataset_dev = split_dataset(dataset_train, 0.9)
    dataset_test = create_dataset('data_test.txt')

    N_atoms = len(atom_dict)

    return dataset_train, dataset_dev, dataset_test, N_atoms

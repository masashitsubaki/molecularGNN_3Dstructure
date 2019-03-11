import pickle
import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QuantumGNN(nn.Module):
    def __init__(self):
        super(QuantumGNN, self).__init__()
        self.embed_atom = nn.Embedding(n_atoms, dim)
        self.gamma = nn.ParameterList([nn.Parameter(
                     torch.FloatTensor([1.0]).to(device))
                     for _ in range(layer_hidden)])
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(2*dim, 1)

    def pad(self, matrices, pad_value):
        """Pad distance matrices with pad_value for batch processing."""
        shapes = [m.shape[0] for m in matrices]
        M = sum(shapes)
        pad_matrices = pad_value + np.zeros((M, M))
        i = 0
        for j, m in enumerate(matrices):
            j = shapes[j]
            pad_matrices[i:i+j, i:i+j] = m
            i += j
        return torch.FloatTensor(pad_matrices).to(device)

    def sum_axis(self, xs, axis):
        y = [torch.sum(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)

    def mean_axis(self, xs, axis):
        y = [torch.mean(x, 0) for x in torch.split(xs, axis)]
        return torch.stack(y)

    def update(self, xs, V, i, M):
        """Update each atom vector considering (i.e., sum or mean)
        (1) all other atom vectors non-linear transformed by neural network
        and (2) the distances (potentials V) between two atoms in a molecule.
        """
        hs = torch.relu(self.W_atom[i](xs))
        if update == 'sum':
            return xs + torch.matmul(V, hs)
        if update == 'mean':
            return xs + torch.matmul(V, hs) / (M-1)

    def forward(self, inputs):

        atoms, distances = inputs

        axis = [len(a) for a in atoms]

        M = np.concatenate([np.repeat(len(a), len(a)) for a in atoms])
        M = torch.unsqueeze(torch.FloatTensor(M).to(device), 1)

        atoms = torch.cat(atoms)
        atom_vectors = self.embed_atom(atoms)

        distances = self.pad(distances, 1e6)

        atom_vectors_ = atom_vectors.clone()  # For concat in the last layer.
        for i in range(layer_hidden):
            potentials = torch.exp(-self.gamma[i]*distances**2)
            atom_vectors = self.update(atom_vectors, potentials, i, M)
        atom_vectors = torch.cat((atom_vectors, atom_vectors_), 1)

        if output == 'sum':
            molecular_vectors = self.sum_axis(atom_vectors, axis)
        if output == 'mean':
            molecular_vectors = self.mean_axis(atom_vectors, axis)

        for j in range(layer_output):
            molecular_vectors = torch.relu(self.W_output[j](molecular_vectors))

        molecular_properties = self.W_property(molecular_vectors)

        return molecular_properties

    def __call__(self, data_batch, train=True):

        Smiles, inputs = data_batch[0], data_batch[1:-1]
        correct_properties = torch.cat(data_batch[-1])
        predicted_properties = self.forward(inputs)

        if train:
            loss = F.mse_loss(predicted_properties, correct_properties)
            return loss
        else:
            """Transform the normalized properties (i.e., mean 0 and std 1)
            to the unit-based properties (e.g., eV and kcal/mol).
            """
            ts = correct_properties.to('cpu').data.numpy()
            ys = predicted_properties.to('cpu').data.numpy()
            ts = std * np.concatenate(ts) + mean
            ys = std * np.concatenate(ys) + mean
            return Smiles, ts, ys


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch):
            data_batch = list(zip(*dataset[i:i+batch]))
            loss = self.model(data_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        AE_sum, SMILES, Ts, Ys = 0, '', [], []
        for i in range(0, N, batch):
            data_batch = list(zip(*dataset[i:i+batch]))
            (Smiles, correct_properties,
             predicted_properties) = self.model(data_batch, train=False)
            AE_sum += sum(abs(correct_properties-predicted_properties))
            SMILES += ' '.join(Smiles) + ' '
            Ts.append(correct_properties)
            Ys.append(predicted_properties)
        MAE = AE_sum / N
        SMILES = SMILES.strip().split()
        T = map(str, np.concatenate(Ts))
        Y = map(str, np.concatenate(Ys))
        predictions = '\n'.join(['\t'.join(x) for x in zip(SMILES, T, Y)])
        return MAE, predictions

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write(MAEs + '\n')

    def save_predictions(self, predictions, filename):
        with open(filename, 'w') as f:
            f.write('Smiles\tCorrect\tPredict\n')
            f.write(predictions + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]


def load_numpy(file_name):
    return np.load(file_name + '.npy')


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, property, update, output, dim, layer_hidden, layer_output, batch,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_hidden, layer_output, batch, decay_interval,
     iteration) = map(int, [dim, layer_hidden, layer_output, batch,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])
    unit = property.split('(')[1][:-1]

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load each preprocessed data."""
    dir_input = '../dataset/' + DATASET + '/input/'
    with open(dir_input + 'Smiles.txt') as f:
        Smiles = f.read().strip().split()
    molecules = load_tensor(dir_input + 'molecules', torch.LongTensor)
    Distances = load_numpy(dir_input + 'Distances')
    properties = load_tensor(dir_input + property, torch.FloatTensor)
    mean = load_numpy(dir_input + property + '_mean')
    std = load_numpy(dir_input + property + '_std')
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        atom_dict = pickle.load(f)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(Smiles, molecules, Distances, properties))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_ = split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    """Set a model."""
    n_atoms = len(atom_dict)
    torch.manual_seed(1234)
    model = QuantumGNN().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MAEs = '../output/result/MAEs--' + setting + '.txt'
    file_predictions = '../output/result/predictions--' + setting + '.txt'
    file_model = '../output/model/' + setting
    MAEs = ('Epoch\tTime(sec)\tLoss_train(MSE,normalized)\t'
            'Error_dev(MAE,' + unit + ')\tError_test(MAE,' + unit + ')')
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')

    """Start training."""
    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        MAE_dev = tester.test(dataset_dev)[0]
        MAE_test, predictions_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        MAEs = '\t'.join(map(str, [epoch, time, loss_train,
                                   MAE_dev, MAE_test]))
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_predictions(predictions_test, file_predictions)
        tester.save_model(model, file_model)

        print(MAEs)

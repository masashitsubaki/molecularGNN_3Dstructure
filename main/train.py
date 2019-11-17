import sys
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import preprocess as pp


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_atom = nn.Embedding(N_atoms, dim)
        self.gamma = nn.ModuleList([nn.Embedding(N_atoms, 1)
                                   for _ in range(layer_hidden)])
        self.W_atom = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_hidden)])
        self.W_output = nn.ModuleList([nn.Linear(dim, dim)
                                       for _ in range(layer_output)])
        self.W_property = nn.Linear(dim, 1)

    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_atom[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    # def update(self, matrix, vectors, layer):
    #     hidden_vectors = self.W_atom[layer](vectors)
    #     return torch.relu(hidden_vectors + torch.matmul(matrix, hidden_vectors))

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def forward(self, inputs):

        atoms, distance_matrices, molecular_sizes = inputs

        """Cat or pad each input data for batch processing."""
        atoms = torch.cat(atoms)
        distance_matrix = self.pad(distance_matrices, 1e6)

        """GNN layer."""
        atom_vectors = self.embed_atom(atoms)
        for l in range(layer_hidden):
            gammas = torch.squeeze(self.gamma[l](atoms))
            M = torch.exp(-gammas*distance_matrix**2)
            atom_vectors = self.update(M, atom_vectors, l)

        """Output layer."""
        for l in range(layer_output):
            atom_vectors = torch.relu(self.W_output[l](atom_vectors))

        """Molecular vector by sum of the atom vectors."""
        molecular_vectors = self.sum(atom_vectors, molecular_sizes)

        """Molecular property."""
        properties = self.W_property(molecular_vectors)

        return properties

    def __call__(self, data_batch, train):

        inputs = data_batch[:-1]
        correct_properties = torch.cat(data_batch[-1])

        if train:
            predicted_properties = self.forward(inputs)
            loss = F.mse_loss(predicted_properties, correct_properties)
            return loss
        else:
            with torch.no_grad():
                predicted_properties = self.forward(inputs)
            ts = correct_properties.to('cpu').data.numpy()
            zs = predicted_properties.to('cpu').data.numpy()
            ts, zs = np.concatenate(ts), np.concatenate(zs)
            sum_absolute_error = sum(np.abs(ts-zs))
            return sum_absolute_error


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        for i in range(0, N, batch_train):
            data_batch = list(zip(*dataset[i:i+batch_train]))
            loss = self.model(data_batch, train=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()
        return loss_total


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0
        for i in range(0, N, batch_test):
            data_batch = list(zip(*dataset[i:i+batch_test]))
            sum_absolute_error = self.model(data_batch, train=False)
            SAE += sum_absolute_error
        MAE = SAE / N
        return MAE

    def save_errors(self, errors, filename):
        with open(filename, 'a') as f:
            f.write(errors + '\n')


if __name__ == "__main__":

    (DATASET, property, dim, layer_hidden, layer_output,
     batch_train, batch_test, lr, lr_decay, decay_interval, iteration,
     setting) = sys.argv[1:]
    (dim, layer_hidden, layer_output, batch_train, batch_test, decay_interval,
     iteration) = map(int, [dim, layer_hidden, layer_output, batch_train,
                            batch_test, decay_interval, iteration])
    lr, lr_decay = map(float, [lr, lr_decay])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses a GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses a CPU...')
    print('-'*100)

    print('Preprocessing the', DATASET, 'dataset.')
    print('Just a moment......')
    (dataset_train, dataset_dev, dataset_test,
     N_atoms) = pp.create_datasets(DATASET, property, device)
    print('-'*100)

    print('The preprocess has finished!')
    print('# of training data samples:', len(dataset_train))
    print('# of development data samples:', len(dataset_dev))
    print('# of test data samples:', len(dataset_test))
    print('-'*100)

    print('Creating a model.')
    torch.manual_seed(1234)
    model = MolecularGraphNeuralNetwork().to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*100)

    for i in range(layer_hidden):
        ones = nn.Parameter(torch.ones((N_atoms, 1))).to(device)
        model.gamma[i].weight.data = ones

    file_errors = '../output/errors--' + setting + '.txt'
    file_model = '../output/model--' + setting

    errors = ('Epoch\tTime(sec)\tLoss_train(MSE)\t'
              'Error_dev(MAE)\tError_test(MAE)')
    with open(file_errors, 'w') as f:
        f.write(errors + '\n')

    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)

    start = timeit.default_timer()

    for epoch in range(iteration):

        epoch += 1
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train = trainer.train(dataset_train)
        error_dev = tester.test(dataset_dev)
        error_test = tester.test(dataset_test)

        time = timeit.default_timer() - start

        if epoch == 1:
            minutes = time * iteration / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*100)
            print(errors)

        error = '\t'.join(map(str, [epoch, time, loss_train,
                                    error_dev, error_test]))
        tester.save_errors(error, file_errors)

        print(error)

    print('The training has finished!')
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
import utils.dists as dists

inpsiz = 784 
hidensiz = 500 
numclases = 10
numepchs = 4
bachsiz = 100
l_r = 0.01 
momentum = 0.9
log_interval = 10
rou = 1
loss_thres = 0.001


class Generator():
    """Generator for MNIST dataset."""

    # Extract MNIST data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.MNIST(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.testset = datasets.MNIST(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.labels = list(self.trainset.classes)

    # Group the data by label
    def group(self):
        # Create empty dict of labels
        grouped_data = {label: []
                        for label in self.labels}  # pylint: disable=no-member

        # Populate grouped data dict
        for datapoint in self.trainset:  # pylint: disable=all
            _, label = datapoint  # Extract label
            label = self.labels[label]

            grouped_data[label].append(  # pylint: disable=no-member
                datapoint)

        self.trainset = grouped_data  # Overwrite trainset with grouped data

    # Run data generation
    def generate(self, path):
        self.read(path)
        self.trainset_size = len(self.trainset)  # Extract trainset size
        self.group()

        return self.trainset


class Loader(object):
    """Load and pass IID data partitions."""

    def __init__(self, config, generator):
        # Get data from generator
        self.config = config
        self.trainset = generator.trainset
        self.testset = generator.testset
        self.labels = generator.labels
        self.trainset_size = generator.trainset_size

        # Store used data seperately
        self.used = {label: [] for label in self.labels}
        self.used['testset'] = []

    def extract(self, label, n):
        if len(self.trainset[label]) > n:
            extracted = self.trainset[label][:n]  # Extract data
            self.used[label].extend(extracted)  # Move data to used
            del self.trainset[label][:n]  # Remove from trainset
            return extracted
        else:
            logging.warning('Insufficient data in label: {}'.format(label))
            logging.warning('Dumping used data for reuse')

            # Unmark data as used
            for label in self.labels:
                self.trainset[label].extend(self.used[label])
                self.used[label] = []

            # Extract replenished data
            return self.extract(label, n)

    def get_partition(self, partition_size):
        # Get an partition uniform across all labels

        # Use uniform distribution
        dist = dists.uniform(partition_size, len(self.labels))
        print(dist)

        partition = []  # Extract data according to distribution
        for i, label in enumerate(self.labels):
            partition.extend(self.extract(label, dist[i]))

        # Shuffle data partition
        random.shuffle(partition)

        return partition

    def get_testset(self):
        # Return the entire testset
        return self.testset

class Net(nn.Module):
    def __init__(self, inpsiz, hidensiz, numclases):
         super(Net, self).__init__()
         self.inputsiz = inpsiz
         self.l1 = nn.Linear(inpsiz, hidensiz) 
         self.relu = nn.ReLU()
         self.l2 = nn.Linear(hidensiz, numclases) 
    def forward(self, y):
         outp = self.l1(y)
         outp = self.relu(outp)
         outp = self.l2(outp)
         return outp



def get_optimizer(model):
    # return optim.SGD(model.parameters(), lr=l_r, momentum=momentum)
    return optim.Adam(model.parameters(), lr=l_r)

def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def extract_weights(model):
    weights = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            weights.append((name, weight.data))

    return weights


def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)

def flatten_weights(weights):
    # Flatten weights into vectors
    weight_vecs = []
    for _, weight in weights:
        weight_vecs.extend(weight.flatten().tolist())

    return np.array(weight_vecs)

def train(model, trainloader, optimizer, epochs):
   
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        for batch_id, (image, label) in enumerate(trainloader):
            # image, label = image.to(device), label.to(device)
            image = image.reshape(-1,28*28)
    
            output = model(image)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))

            # Stop training if model is already in good shape
            if loss.item() < loss_thres:
                return loss.item()
    
    logging.info('loss: {}'.format(loss.item()))
    return loss.item()


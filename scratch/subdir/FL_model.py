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


# numepchs = 4
# bachsiz = 100
# l_r = 0.01 
# momentum = 0.9
# log_interval = 10
# rou = 1
# loss_thres = 0.001

device = torch.device('cpu')

class Arguments():
    def __init__(self, config=False):
        if config:
            self.batch_size = config.fl.batch_size
            self.epochs = config.fl.epochs
            self.rounds = config.fl.rounds
            self.num_nodes = config.nodes.total
            self.num_participants = config.nodes.participants_per_round
            self.num_aggregators = config.nodes.aggregators_per_round
        self.lr = 0.001
        self.momentum = 0.5
        self.no_cuda = True
        self.seed = 1
        self.log_interval = 10
        self.loss_thres = 0.001
        self.inpsiz = 784 
        self.hidensiz = 500 
        self.numclases = 10


class Generator():
    """Generator for MNIST dataset."""

    # Extract MNIST data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.MNIST(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,), (0.5,))
            ]))
        self.testset = datasets.MNIST(
            path, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5,), (0.5,))
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
        dist = dists.normal(partition_size, len(self.labels))
        # print(dist)

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
    def __init__(self):
        arg = Arguments()
        super(Net, self).__init__()
        #  self.inputsiz = arg.inpsiz
        #  self.l1 = nn.Linear(arg.inpsiz, arg.hidensiz) 
        #  self.relu = nn.ReLU()
        #  self.l2 = nn.Linear(arg.hidensiz, arg.numclases)
        self.fc1 = nn.Linear(28 * 28, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        #  outp = self.l1(y)
        #  outp = self.relu(outp)
        #  outp = self.l2(outp)
        #  return outp
        x = torch.relu(self.fc1(x.view(-1, 28 * 28)))
        x = self.fc2(x)
        return x



def get_optimizer(model, config):
    args = Arguments(config)
    # return optim.SGD(model.parameters(), lr=l_r, momentum=momentum)
    return optim.Adam(model.parameters(), lr=args.lr)

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

# def train(model, trainloader, optimizer, epochs, config):
    
#     args = Arguments(config)
#     # model.to(device)
#     model.train()
#     criterion = nn.CrossEntropyLoss().to(device)
    
#     for epoch in range(1, epochs + 1):
#         for batch_id, (image, label) in enumerate(trainloader):
#             image, label = image.to(device), label.to(device)
#             image = image.reshape(-1,28*28)
#             optimizer.zero_grad()
#             output = model(image)
#             loss = criterion(output, label)
            
#             loss.backward()
#             optimizer.step()
#             if batch_id % args.log_interval == 0:
#                 logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
#                     epoch, epochs, loss.item()))

#             # Stop training if model is already in good shape
#             if loss.item() < args.loss_thres:
#                 return loss.item()
    
#     logging.info('loss: {}'.format(loss.item()))
#     return loss.item()

def train(model, train_loader, optimizer, epochs):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    print("Training completed!")

# def test(model, testloader):
#     model.to(torch.device('cpu'))
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = len(testloader.dataset)
#     with torch.no_grad():
#         for image, label in testloader:
#             image, label = image.to(device), label.to(device)
#             output = model(image)
#             # sum up batch loss
#             test_loss += F.nll_loss(output, label, reduction='sum').item()
#             # get the index of the max log-probability
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(label.view_as(pred)).sum().item()

#     accuracy = correct / total
#     logging.debug('Accuracy: {:.2f}%'.format(100 * accuracy))

#     return accuracy

def test(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy =  100 * correct / total
    print(f"accuracy on testset {accuracy:.2f}%") 
    return accuracy
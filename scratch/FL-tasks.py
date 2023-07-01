import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import syft as sy  

class Arguments():
    def __init__(self):
        self.batch_size = 100
        self.epochs = 10
        self.lr = 0.001
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.num_nodes = 100
        self.num_participants = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
# Group the data by label
def group(trainset, labels):
    # Create empty dict of labels
    grouped_data = {label: [] for label in labels}  # pylint: disable=no-member

    # Populate grouped data dict
    for datapoint in trainset:  # pylint: disable=all
        _, label = datapoint  # Extract label
        label = labels[label]

        grouped_data[label].append(datapoint)   # pylint: disable=no-member

    return grouped_data  # Overwrite trainset with grouped data

def save_model(model):
        path = './global'
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))
        
def setUp():
    args = Arguments()
    hook = sy.TorchHook(torch)  # <-- NEW: hook PyTorch ie add extra functionalities to support Federated Learning
    nodes = [sy.VirtualWorker(hook, id= str(i)) for i in range(args.num_nodes)]

    federated_train_loader = sy.FederatedDataLoader( # <-- this is now a FederatedDataLoader 
        datasets.MNIST('./data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
        .federate(nodes), # <-- NEW: we distribute the dataset across all the workers, it's now a FederatedDataset
        batch_size=args.batch_size, shuffle=True)   
    
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True)

    trainset = datasets.MNIST(
            "./data", train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
    testset = datasets.MNIST(
            "./data", train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
    labels = list(trainset.classes)
    trainset_size = len(trainset)
    trainset = group(trainset , labels) # grouped data by labels
    logging.info('Dataset size: {}'.format(trainset_size))
    logging.debug('Labels ({}): {}'.format(
            len(labels), labels))

    model = Net()



# def train(args, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(federated_train_loader): # <-- now it is a distributed dataset
#         model.send(data.location) # <-- NEW: send the model to the right location
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#         model.get() # <-- NEW: get the model back
#         if batch_idx % args.log_interval == 0:
#             loss = loss.get() # <-- NEW: get the loss back
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * args.batch_size, len(train_loader) * args.batch_size, #batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
            

# def test(args, model, device, test_loader):
    # model.eval()
    # test_loss = 0
    # correct = 0
    # with torch.no_grad():
    #     for data, target in test_loader:
    #         data, target = data.to(device), target.to(device)
    #         output = model(data)
    #         test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
    #         pred = output.argmax(1, keepdim=True) # get the index of the max log-probability 
    #         correct += pred.eq(target.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
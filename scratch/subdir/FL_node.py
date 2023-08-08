import FL_model
import torch
import copy
import logging
from utils.CSVManager import CSVFileManager
from run import MLModel


class Node(object):

    def __init__(self, nodeinfo) -> None:
        self.node = copy.copy(nodeinfo)
        self.loss = 10.0 # set intial loss big 
    
    def download(self, argv):
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv):
        try:
            return argv.copy()
        except:
            return argv


    def set_data(self, data, config):
        # Extract from config
        test_partition = self.test_partition = config.nodes.test_partition

        # Download data
        self.data = self.download(data)

        # Extract trainset, testset (if applicable)
        self.trainset = data[:int(len(self.data) * (1 - test_partition))]
        self.testset = data[int(len(self.data) * (1 - test_partition)):]
       
    def configure(self, config):
        # configure the node before a training, getting the last global model
        model_path = config.paths.model
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size
        # Download most recent global model
        path = model_path + '/global'
        self.model = FL_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # Create optimizer
        self.optimizer = FL_model.get_optimizer(self.model, config)
    
    def train(self,round, fileManager:CSVFileManager, config):
        logging.info('training on node #{}'.format(self.node.nodeId))

        # Perform model training
        trainloader = FL_model.get_trainloader(self.trainset, self.batch_size)
        self.loss = FL_model.train(self.model, trainloader,
                       self.optimizer, self.epochs) #not reg
        
        # Extract model weights and biases
        weights = FL_model.extract_weights(self.model)
        # testing the model
        testloader = FL_model.get_testloader(self.testset, 1000)
        # Generate report for server
        accuracy = FL_model.test(self.model, testloader)
        #creating the MLModel       
        report_id = fileManager.get_instance_id('modelId') + 1
        mlmodel = MLModel(
            modelId=report_id,
            nodeId=self.node.nodeId,
            taskId= fileManager.get_instance_id("taskId"),
            round=round,
            type=1, # 1 for local model
            positiveVote=0,
            negativeVote=0,
            evaluator1=-1,
            evaluator2=-1,
            evaluator3=-1,
            aggregated=False,
            aggModelId=0,
            accuracy=0
        )
        fileManager.write_instance(mlmodel)
        self.report = Report(report_id, self.node, self.data, weights,self.loss,accuracy, mlmodel)
        return mlmodel
    def evaluate(self):
        pass
class Report(object):
    """Federated learning client report."""

    def __init__(self, id, node,data, weights, loss,accuracy, model):
        self.id = id
        self.node_id = node.nodeId
        self.num_samples = len(data)
        self.loss = loss
        self.weights = weights
        self.accuracy = accuracy
        self.model = model
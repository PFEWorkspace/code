import logging
import random
import torch
import numpy as np
from utils.CSVManager import CSVFileManager
from run import MLModel, MLModelRefrence
from FL_model import Generator, Loader, Net, extract_weights
from FL_node import Node

import utils.dists as dists


class FLManager(object):

    def __init__(self, config):
        self.config = config

    def setUp(self, nodesList):
        logging.info('setting up nodes, datasets and initial model for a new FL task')    
        self.round = 0 
        models_path = self.config.paths.FLmodels
        self.modelsFileManager = CSVFileManager(models_path, MLModel._fields_)
        initial_model = MLModel(
            modelId=self.modelsFileManager.get_instance_id("modelId") + 1,
            nodeId=-1,
            taskId=self.modelsFileManager.get_instance_id("taskId") + 1,
            round=0,
            type=3, # 3 for global model
            positiveVote=0,
            negativeVote=0,
            evaluator1=-1,
            evaluator2=-1,
            evaluator3=-1,
            aggregated=True,
            aggModelId=0,
            accuracy=0
        )
        self.modelsFileManager.write_instance(initial_model)

        self.load_data()
        self.load_model()        
        self.make_nodes(nodesList)

        return initial_model

    def load_data(self):
        # Extract config for loaders
        config = self.config

        # Set up data generator
        generator = Generator()

        # Generate data
        data_path = self.config.paths.data
        data = generator.generate(data_path)
        labels = generator.labels

        logging.info('Dataset size: {}'.format(
            sum([len(x) for x in [data[label] for label in labels]])))
        logging.debug('Labels ({}): {}'.format(
            len(labels), labels))

        # Set up data loader
        self.loader = Loader(config, generator)

        logging.info('Loader: {}, IID: {}'.format(
            self.config.loader, self.config.data.IID))

    def load_model(self):
        model_path = self.config.paths.model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = Net()
        self.save_model(self.model, model_path)

         # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    def make_nodes(self, nodes_info):
        nodes = []
        [nodes.append(Node(node)) for node in nodes_info]
        [self.set_node_data(node) for node in nodes]
        self.nodes = nodes

    def set_node_data(self, node):
        # if self.config.data.partition.get('size'):
        #     partition_size = self.config.data.partition.get('size')
        # elif self.config.data.partition.get('range'):
        #     start, stop = self.config.data.partition.get('range')
        #     partition_size = random.randint(start, stop)   
        
        data = self.loader.get_partition(node.node.datasetSize)
        node.set_data(data, self.config)

    def start_round(self, selectedTrainers):
        rounds = self.config.fl.rounds
        logging.info('**** Round {}/{} ****'.format(self.round, rounds))

        #configuration
        loading = self.config.data.loading
        localmodels = []
        for i in selectedTrainers:
            if not self.nodes[i].node.dropout :    
                if loading == 'dynamic' and self.round > 0 :
                    self.set_node_data(self.nodes[i])
                self.nodes[i].configure(self.config)
                localmodels.append((self.nodes[i].train(self.round, self.modelsFileManager)).get_refrence())
        return localmodels        
 
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)
    
    def save_model(self, model, path):
        path += '/global'
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def save_reports(self, round, reports):
        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
                report.weights)) for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(extract_weights(self.model))
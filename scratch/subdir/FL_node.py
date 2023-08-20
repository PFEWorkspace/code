import FL_model
import torch
import copy
import logging
import numpy as np
from typing import List, Dict
from utils.CSVManager import CSVFileManager
from run import MLModel
from config import Config


class Node(object):

    def __init__(self, nodeinfo, config) -> None:
        self.node = copy.copy(nodeinfo)
        self.loss = 10.0 # set intial loss big 
        self.reports = []
        self.numEvaluations = 0 
        self.numAggregations = 0
        self.trueEvaluation = 0
        self.falseEvaluation = 0
        self.trueAggregation = 0
        self.falseAggregation = 0
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

    def add_new_evaluation(self):
        self.numEvaluations +=1

    def add_true_evaluation(self):
        self.trueEvaluation += 1

    def add_false_evaluation(self):
        self.falseEvaluation += 1
    
    def add_new_aggregation(self):
        self.numAggregations +=1

    def add_true_aggregation(self):
        self.trueAggregation += 1

    def add_false_aggregation(self):
        self.falseAggregation += 1
        self.trueAggregation =- 1

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
        
        # Download most recent global model
        path = model_path + '/global'
        self.model = FL_model.Net()
        # print(self.model.parameters())
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

        # Create optimizer
        self.optimizer = FL_model.get_optimizer(self.model, config)
    
    
      
    def train(self, numRound, fileManager:CSVFileManager):
        logging.info('training on node #{}'.format(self.node.nodeId))

        # Perform model training
        trainloader = FL_model.get_trainloader(self.trainset, self.batch_size)
        FL_model.train(self.model, trainloader, self.optimizer, self.epochs) #not reg
        
        # Extract model weights and biases
        weights = FL_model.extract_weights(self.model)
        # testing the model
        testloader = FL_model.get_testloader(self.testset, self.batch_size)
        # Generate report for server
        self.loss, accuracy = FL_model.test(self.model, testloader)
        #creating the MLModel       
        report_id = fileManager.get_instance_id('modelId') + 1
        mlmodel = MLModel(
            modelId=report_id,
            nodeId=self.node.nodeId,
            taskId= fileManager.get_instance_id("taskId"),
            round=numRound,
            type=0, # 0 for local model
            positiveVote=0,
            negativeVote=0,
            evaluator1=-1,
            evaluator2=-1,
            evaluator3=-1,
            aggregated=False,
            aggModelId=0,
            accuracy=round(accuracy,2),
            acc1=0.0,
            acc2=0.0,
            acc3=0.0
        )
        fileManager.write_instance(mlmodel)
        self.reports.append(Report(report_id, self.node.nodeId, len(self.data),self.loss, weights,accuracy, mlmodel, copy.deepcopy(self.model)))
        return self.reports[-1]
    
    def evaluate(self, model):
        testloader = FL_model.get_testloader(self.testset, self.batch_size)
        loss, accuracy = FL_model.test(model, testloader)
        return loss, accuracy

    
    def aggregate(self, reports, aggType, numRound, fileManager:CSVFileManager, evaluation=False):

        updated_weights = self.federated_averaging(reports)
        
        if evaluation :
            aggModel = copy.deepcopy(self.model)
            FL_model.load_weights(aggModel,updated_weights)
            return  aggModel
        else:
            
            FL_model.load_weights(self.model, updated_weights) #putting the new weights in the model for this node
            # Test global model accuracy
            
            testloader = FL_model.get_testloader(self.testset, self.batch_size)
            self.loss, accuracy = FL_model.test(self.model, testloader)
        
            #creating the MLModel       
            report_id = fileManager.get_instance_id('modelId') + 1
            mlmodel = MLModel(
            modelId=report_id,
            nodeId=self.node.nodeId,
            taskId= fileManager.get_instance_id("taskId"),
            round=numRound,
            type=aggType, # 1:intermediaire or 2:global
            positiveVote=0,
            negativeVote=0,
            evaluator1=-1,
            evaluator2=-1,
            evaluator3=-1,
            aggregated=False,
            aggModelId=0,
            accuracy=round(accuracy,2),
            acc1=0.0,
            acc2=0.0,
            acc3=0.0
            )
            # fileManager.write_instance(mlmodel)
            self.reports.append(Report(report_id, self.node.nodeId, len(self.data),self.loss, updated_weights,accuracy, mlmodel, copy.deepcopy(self.model)))

            return mlmodel, self.model 
        

    
    def federated_averaging(self, reports):
        models = []
        for r in reports :
            models.append(r.net.state_dict())
        updated_weights = self.average_weights(models)
        return updated_weights
            
    
    def average_weights(self,weights: List[Dict[str,torch.Tensor]])-> Dict[str,torch.Tensor]:
        weights_avg = copy.deepcopy(weights[0])

        for key in weights_avg.keys():
            for i in range(1, len(weights)):
                weights_avg[key] += weights[i][key]
            weights_avg[key] = torch.div(weights_avg[key], len(weights))

        return weights_avg
    
    def updateHonestyTrainer(self,globalModel, globalAcc,config:Config):
        if self.node.dropout: 
            contrib = config.fl.malus
        else:

            mlmodel = self.get_last_model()
            net = self.get_net(mlmodel.modelId)           
            if mlmodel.positiveVote > mlmodel.negativeVote: #the model was valid 
                contrib = self.contribution(net, globalModel)
            else:
                contrib = - config.fl.honesty_beta * abs(globalAcc - mlmodel.accuracy)
        self.node.honesty = self.node.honesty + config.fl.honesty_alpha * contrib
        return self.node.honesty            

    def contribution(self, local, globalModel):
        local.eval()
        globalModel.eval()
        
        # Convert model parameters to tensors if they are not already
        local_model_params = torch.cat([param.view(-1) for param in local.parameters()])#torch.tensor(local.parameters(), requires_grad=True)
        global_model_params = torch.cat([param.view(-1) for param in globalModel.parameters()])#torch.tensor(globalModel.parameters(), requires_grad=False)
        # print("local: ", local_model_params)
        # print("global: ", global_model_params)
        # Calculate the cosine similarity
        cosine_similarity = torch.nn.functional.cosine_similarity(local_model_params, global_model_params, dim=0)
        
        # Calculate the contribution as the square of the cosine similarity
        contribution = cosine_similarity ** 2
        print("node {} contribution {}".format(self.node.nodeId, contribution))
        return contribution.item()

    def updateHonestyAggregator(self, numEvals, numAgg, config:Config):
        contrib = (config.fl.honesty_phi* self.trueEvaluation + self.trueAggregation - config.fl.honesty_gamma*(self.falseEvaluation+self.falseAggregation))/(numEvals+numAgg)
        self.node.honesty = self.node.honesty + config.fl.honesty_alpha * contrib
        return self.node.honesty 
    

    def save_model(self, model, path):
        path += '/global'
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))
        
    def save_reports(self, round, reports):
        if reports:
            self.saved_reports['round{}'.format(round)] = [(report.client_id, self.flatten_weights(
                report.weights)) for report in reports]

        # Extract global weights
        self.saved_reports['w{}'.format(round)] = self.flatten_weights(FL_model.extract_weights(self.model))  

    def get_report(self,id):
        for r in self.reports:
            if r.id == id :
                return r

    def get_last_model(self):
        return self.reports[-1].model  
    
    def get_net(self, id):    
        return self.get_report(id).net

    def resetModel(self,id, model):
        report = self.get_report(id)
        report.model = model
        
class Report(object):
    """Federated learning client report."""

    def __init__(self, id, nodeid ,dataLength,loss, weights,accuracy, model, net):
        self.id = id
        self.node_id = nodeid
        self.num_samples = dataLength
        self.loss = loss
        self.weights = weights
        self.accuracy = accuracy
        self.model = model
        self.net = net
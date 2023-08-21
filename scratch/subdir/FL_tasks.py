import logging
import random
import torch
import copy
import numpy as np
from utils.CSVManager import CSVFileManager
from run import MLModel, FLNodeStruct
from FL_model import Generator, Loader, Net, extract_weights
from FL_node import Node, Report

import utils.dists as dists


class FLManager(object):

    def __init__(self, config):
        self.config = config
        self.nodesFileManager = CSVFileManager(self.config.nodes.source, FLNodeStruct._fields_)
        models_path = self.config.paths.FLmodels
        self.modelsFileManager = CSVFileManager(models_path, MLModel._fields_)

    def setUp(self, nodesList):
        logging.info('setting up nodes, datasets and initial model for a new FL task')    
        self.round = 0        
        initial_model = MLModel(
            modelId=self.modelsFileManager.get_instance_id("modelId") + 1,
            nodeId=-1,
            taskId=self.modelsFileManager.get_instance_id("taskId") + 1,
            round=0,
            type=2, # 2 for global model
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
        for node in self.nodes: #passing the global model
            node.model = node.download(self.model)
        self.globalModel = initial_model
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
        # if self.config.paths.reports:
        #     self.saved_reports = {}
        #     self.save_reports(0, [])  # Save initial model

    def make_nodes(self, nodes_info):
        nodes = []
        [nodes.append(Node(node, self.config)) for node in nodes_info]
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

    def start_round(self, selectedTrainers, numSelectedTrainers):
        rounds = self.config.fl.rounds
        print('**** Round {}/{} ****'.format(self.round, rounds))

        #configuration
        loading = self.config.data.loading
        localmodels = []
        
        for i in range(0,numSelectedTrainers):
            print("training on node "+str(selectedTrainers[i]))
            if not self.nodes[selectedTrainers[i]].node.dropout :    
                if loading == 'dynamic' and self.round > 0 :
                    self.set_node_data(self.nodes[selectedTrainers[i]])

                self.nodes[selectedTrainers[i]].configure(self.config)
                report = self.nodes[selectedTrainers[i]].train(self.round, self.modelsFileManager)
                localmodels.append(report.model)
                
                # # Extract flattened weights (if applicable)
                # if self.config.paths.reports:
                #     self.save_reports(self.round, report)
        return localmodels        
    
        
    def evaluateLocal(self, nodeId, model:MLModel):
        #get the model from the source node
        m = self.nodes[model.nodeId].get_net(model.modelId)
        if m:
            loss, acc = self.nodes[nodeId].evaluate(m)
            acc = round(acc,2)
            print("accuracy {} evaluator accuracy {}".format(model.accuracy,acc))
            #get the evaluation
            evaluation = False
            if (abs(model.accuracy - acc) < self.config.fl.local_validation_threshold):
                evaluation = True

            #update the model and nodes
            if model.evaluator1==-1: #first evaluation
                model.evaluator1 = nodeId
                model.acc1 = acc
                self.modelsFileManager.modify_instance_field(model.modelId,"evaluator1",model.evaluator1)
                self.modelsFileManager.modify_instance_field(model.modelId,"acc1",round(model.acc1,2))
                self.nodes[nodeId].add_new_evaluation()
                if evaluation:
                    model.positiveVote = 1
                    self.modelsFileManager.modify_instance_field(model.modelId,"positiveVote",model.positiveVote)
                else :
                    model.negativeVote = 1
                    self.modelsFileManager.modify_instance_field(model.modelId,"negativeVote",model.negativeVote) 
            elif model.evaluator2 == -1 :
                model.evaluator2 = nodeId
                model.acc2 = acc 
                self.modelsFileManager.modify_instance_field(model.modelId,"evaluator2",model.evaluator2)
                self.modelsFileManager.modify_instance_field(model.modelId,"acc2",round(model.acc2,2))
                self.nodes[nodeId].add_new_evaluation()
                #check if its true or false                      
                if evaluation:
                    model.positiveVote += 1
                    self.modelsFileManager.modify_instance_field(model.modelId,"positiveVote",model.positiveVote)
                    if model.positiveVote==2 :
                        self.nodes[nodeId].add_true_evaluation()
                        self.nodes[model.evaluator1].add_true_evaluation()
                else :
                    model.negativeVote += 1 
                    self.modelsFileManager.modify_instance_field(model.modelId,"negativeVote",model.negativeVote)
                    if model.negativeVote==2 :
                        self.nodes[nodeId].add_true_evaluation()
                        self.nodes[model.evaluator1].add_true_evaluation()
            else: # third evaluation
                model.evaluator3 = nodeId
                model.acc3 = acc
                self.modelsFileManager.modify_instance_field(model.modelId,"evaluator3",model.evaluator3)
                self.modelsFileManager.modify_instance_field(model.modelId,"acc3",round(model.acc3,2))
                self.nodes[nodeId].add_new_evaluation()
                if evaluation:
                    model.positiveVote += 1
                    self.modelsFileManager.modify_instance_field(model.modelId,"positiveVote",model.positiveVote)
                    if abs(model.acc1 - model.accuracy)< self.config.fl.local_validation_threshold : #the first eval gave a positiveVote
                        self.nodes[nodeId].add_true_evaluation()
                        self.nodes[model.evaluator1].add_true_evaluation()
                        self.nodes[model.evaluator2].add_false_evaluation()
                    elif abs(model.acc2 - model.accuracy) < self.config.fl.local_validation_threshold : #second one who gate the positivevote
                        self.nodes[nodeId].add_true_evaluation()
                        self.nodes[model.evaluator2].add_true_evaluation()
                        self.nodes[model.evaluator1].add_false_evaluation()
                else :
                    model.negativeVote += 1
                    self.modelsFileManager.modify_instance_field(model.modelId,"negativeVote",model.negativeVote)
                    if abs(model.acc1 - model.accuracy)>= self.config.fl.local_validation_threshold : #the first eval gave a negativeVote
                        self.nodes[nodeId].add_true_evaluation()
                        self.nodes[model.evaluator1].add_true_evaluation()
                        self.nodes[model.evaluator2].add_false_evaluation()
                    elif abs(model.acc2 - model.accuracy)>= self.config.fl.local_validation_threshold : #second one who gate the positivevote
                        self.nodes[nodeId].add_true_evaluation()
                        self.nodes[model.evaluator2].add_true_evaluation()
                        self.nodes[model.evaluator1].add_false_evaluation()  
            #reset mlmodel
            self.nodes[model.nodeId].resetModel(model.modelId, model)
        return model    
    
    def evaluateIntermediaire(self, nodeId, model:MLModel):
       
        allModels = self.modelsFileManager.retrieve_instances()
        modelToEval = self.nodes[model.nodeId].get_net(model.modelId)
        modelsToagg=[]
        for m in allModels:
            if m.aggregated and m.aggModelId==model.modelId :
                modelsToagg.append(self.nodes[m.nodeId].get_report(m.modelId))
                
        
        #aggregate them and compare it to self.nodes[model.nodeId].model and update the aggregations/evaluations accordingly        
        evaluationModel = self.nodes[nodeId].aggregate(modelsToagg, -1, self.round, self.modelsFileManager, True)
        
        evaluation = self.compare_model(modelToEval, evaluationModel)
                
        if model.evaluator1==-1:
            model.evaluator1 = nodeId
            self.modelsFileManager.modify_instance_field(model.modelId,"evaluator1",model.evaluator1)
            if evaluation:
                model.positiveVote += 1
                self.modelsFileManager.modify_instance_field(model.modelId,"positiveVote",model.positiveVote)
                self.nodes[nodeId].add_new_aggregation()
                self.nodes[nodeId].add_true_aggregation()
            else :
                model.negativeVote += 1
                self.modelsFileManager.modify_instance_field(model.modelId,"negativeVote",model.negativeVote)
                self.nodes[nodeId].add_new_aggregation()
                self.nodes[nodeId].add_true_aggregation() # the second evaluator will decide wich one will be to false
        elif model.evaluator2==-1:  # to have 2nd eval means first one was false  
            model.evaluator2 = nodeId
            self.modelsFileManager.modify_instance_field(model.modelId,"evaluator2",model.evaluator2)
            if evaluation:
                model.positiveVote += 1
                self.modelsFileManager.modify_instance_field(model.modelId,"positiveVote",model.positiveVote)
                self.nodes[nodeId].add_new_aggregation()
                self.nodes[nodeId].add_true_aggregation()
                self.nodes[model.evaluator1].add_false_aggregation()
            else:
                model.negativeVote += 1
                self.modelsFileManager.modify_instance_field(model.modelId,"negativeVote",model.negativeVote)
                self.nodes[nodeId].add_new_aggregation()
                self.nodes[nodeId].add_true_aggregation()
                self.nodes[model.nodeId].add_false_aggregation()
        #reset mlmodel
        self.nodes[model.nodeId].resetModel(model.modelId, model)
        return model

    def aggregate(self, nodeId, models:MLModel, aggType):
        modelsReports = []
        for m in models:
            modelsReports.append(self.nodes[m.nodeId].get_report(m.modelId))
        
        mlmodel, self.model = self.nodes[nodeId].aggregate(modelsReports, aggType, self.round, self.modelsFileManager, False)
        self.modelsFileManager.write_instance(mlmodel)

        for m in models:
            self.modelsFileManager.modify_instance_field(m.modelId,"aggregated",True)
            self.modelsFileManager.modify_instance_field(m.modelId,"aggModelId",mlmodel.modelId)
            m.aggregated = True
            m.aggModelId = mlmodel.modelId
            self.nodes[m.nodeId].resetModel(m.modelId, m)
        if aggType==2 : #global model
            self.save_model(self.model, self.config.paths.model)
            self.globalModel = mlmodel
            # for node in self.nodes:
            #     node.model = node.download(self.model) #set the global model
            
        # # Extract flattened weights (if applicable)
        # if self.config.paths.reports:
        #     self.save_reports(self.round, modelsReports)
        
        self.nodes[nodeId].add_new_aggregation()
        #assume it's a true aggregration until proven wrong
        self.nodes[nodeId].add_true_aggregation()

        return mlmodel
    
    def resetRound(self, trainers:list, aggregators:list):
        self.round += 1
        print("round number: ",self.round)
        #calculate honesty
        #update the nodes on csv   
        for index in trainers:
            # print("calculating honesty of node ",index)
            honesty = self.nodes[index].updateHonestyTrainer(self.model, self.globalModel.accuracy, self.config)
            self.nodesFileManager.modify_instance_field(index,"honesty",round(honesty,3))
            self.nodesFileManager.modify_instance_field(index,"task",0)
            print("node {} honesty {}".format(index,round(honesty,3)))

            # dropout = random.choices(["true", "false"], weights=[0.1, 0.9])[0]
            # malicious = random.choices(["true", "false"],weights=[0.05, 0.95])[0]
            # self.nodesFileManager.modify_instance_field(index,"dropout",dropout)
            # self.nodesFileManager.modify_instance_field(index,"malicious",malicious)

        numEvals = sum(self.nodes[index].numEvaluations for index in aggregators)
        numAggs= sum(self.nodes[index].numAggregations for index in aggregators) 

        for index in aggregators:
            honesty = self.nodes[index].updateHonestyAggregator(numEvals, numAggs, self.config)
            self.nodesFileManager.modify_instance_field(index,"honesty",round(honesty,3))
            self.nodesFileManager.modify_instance_field(index,"task",1)
            print("node {} honesty {}".format(index,round(honesty,3)))

            self.nodes[index].numEvaluations = 0
            self.nodes[index].numAggregations = 0
            self.nodes[index].trueEvaluation = 0
            self.nodes[index].falseEvaluation = 0
            self.nodes[index].trueAggregation = 0
            self.nodes[index].falseAggregation = 0
            # dropout = random.choices(["true", "false"], weights=[0.1, 0.9])[0]
            # malicious = random.choices(["true", "false"],weights=[0.05, 0.95])[0]
            # self.nodesFileManager.modify_instance_field(index,"dropout",dropout)
            # self.nodesFileManager.modify_instance_field(index,"malicious",malicious)

        instances = self.nodesFileManager.retrieve_instances()  
        for i in range(0,self.config.nodes.total):
            self.nodes[i].node = instances[i]    
        return instances[0:self.config.nodes.total]

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

    def compare_model(self, model1, model2):
        model1.eval()
        model2.eval()
        # Iterate through corresponding parameters of both models
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
           
            # if not torch.allclose(param1.data, param2.data,atol=1e-8, rtol=1e-5):
            diff = torch.abs(param1 - param2)
            if torch.max(diff) > self.config.fl.intermediaire_validation_threshold :
               
                return False
        return True


    # def save_reports(self, round, reports):
    #     if reports:
    #         self.saved_reports['round{}'.format(round)] = [(report.nodeid, self.flatten_weights(
    #             report.weights)) for report in reports]

    #     # Extract global weights
    #     self.saved_reports['w{}'.format(round)] = self.flatten_weights(extract_weights(self.model))

    
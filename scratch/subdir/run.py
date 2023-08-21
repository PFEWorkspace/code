<<<<<<< HEAD
import ctypes
import argparse
import logging
from py_interface import *
import FL_tasks as fl
import numpy as np
import config
import copy
from typing import List

numMaxNodes = 300
numMaxTrainers = 150  
numMaxAggregators = 150
numMaxBCNodes = 100
numMaxModelsToAgg= 20


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='../../config.json',
                    help='Federated learning configuration file.')


args = parser.parse_args()


class MLModel(ctypes.Structure):
    _pack_ = 1  # Pack the structure to match the C++ layout
    _fields_ = [
        ("modelId", ctypes.c_int),
        ("nodeId", ctypes.c_int),
        ("taskId", ctypes.c_int),
        ("round", ctypes.c_int),
        ("type", ctypes.c_int),
        ("positiveVote", ctypes.c_int),
        ("negativeVote", ctypes.c_int),
        ("evaluator1", ctypes.c_int),
        ("evaluator2", ctypes.c_int),
        ("evaluator3", ctypes.c_int),
        ("aggregated", ctypes.c_bool),
        ("aggModelId", ctypes.c_int),
        ("accuracy", ctypes.c_double),
        ("acc1", ctypes.c_double),
        ("acc2", ctypes.c_double),
        ("acc3", ctypes.c_double)
    ]

    def get_refrence(self):
        return MLModelRefrence(modelId= self.modelId, nodeId=self.nodeId, taskId=self.taskId, round=self.round)

class MLModelRefrence(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("modelId", ctypes.c_int),
        ("nodeId", ctypes.c_int),
        ("taskId", ctypes.c_int),
        ("round", ctypes.c_int)
    ]

class FLNodeStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("nodeId", ctypes.c_int),
        ("availability", ctypes.c_bool),
        ("honesty", ctypes.c_double),
        ("datasetSize", ctypes.c_int),
        ("freq", ctypes.c_int),
        ("transRate", ctypes.c_int),
        ("task", ctypes.c_int),
        ("dropout", ctypes.c_bool),
        ("malicious", ctypes.c_bool)
    ]

    def learning_cost(self,datasetSize,freq):
        return datasetSize / freq
        

    def communication_cost(self, m, transRate):
        return m / transRate  
        

    def get_cost(self,datasetSize,freq,transRate, w):
        
        return self.learning_cost(datasetSize,freq) + self.communication_cost(w,transRate)

class BCNodeStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("nodeId", ctypes.c_int),
        ("task", ctypes.c_int),
    ]


class AiHelperEnv(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type", ctypes.c_int),
        ("nodeId", ctypes.c_int),
        ("models",  MLModel * numMaxModelsToAgg),
        ("numNodes", ctypes.c_int),
        ("numTrainers", ctypes.c_int),
        ("numAggregators", ctypes.c_int),
        ("numRounds", ctypes.c_int),
        ("nodes", FLNodeStruct * numMaxNodes)
    ]

class AiHelperAct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("model", MLModel),
        ("numLocalModels", ctypes.c_int),
        ("localModels", MLModel * numMaxTrainers),
        ("selectedTrainers", ctypes.c_int * numMaxTrainers),
        ("selectedAggregators", ctypes.c_int * numMaxAggregators),
        ("numTrainers", ctypes.c_int),
        ("numAggregators", ctypes.c_int),
        ("numFLNodes", ctypes.c_int),
        ("FLNodesInfo", FLNodeStruct * numMaxNodes)
    ]

class AiHelperContainer:
    use_ns3ai = True
    nodes: List[FLNodeStruct]= []
    # FL_manager: fl.FLManager
    def __init__(self, config:config.Config, uid: int = 2333) -> None:
        self.rl = Ns3AIRL(uid, AiHelperEnv, AiHelperAct)
        self.FL_manager = fl.FLManager(config)
        pass

    def exactSelection(self, act,config):
        alpha = config.fl.alpha 
        nodes_scores = []
        
        for node in self.nodes:

            if node.availability: 
                score = alpha * node.honesty - (1 - alpha) * node.get_cost(node.datasetSize, node.freq, node.transRate, config.model.size)
                nodes_scores.append({"nodeId":node.nodeId , "score": score})
        sorted_indexes =sorted(nodes_scores, key=lambda x: x["score"], reverse=True) # np.argsort([score for _, score in node_scores])[::-1]
        # print(str(sorted_indexes))
        # act.selectedAggregators= sorted_indexes[0:config.nodes.aggregators_per_round]
        # act.selectedTrainers= sorted_indexes[config.nodes.aggregators_per_round : config.nodes.aggregators_per_round+config.nodes.participants_per_round]
        for i in range(0,config.nodes.aggregators_per_round):
            act.selectedAggregators[i]= sorted_indexes[i]["nodeId"]
        for i in range(0, config.nodes.participants_per_round):
            act.selectedTrainers[i] = sorted_indexes[i+config.nodes.aggregators_per_round]["nodeId"]   
        act.numAggregators = config.nodes.aggregators_per_round
        act.numTrainers = config.nodes.participants_per_round
        

    def DRLSelection(self, act):
        pass

    def hybridSelection(self, act):
        pass


    def do(self, env:AiHelperEnv, act:AiHelperAct, config:config.Config) -> AiHelperAct:
      
        if env.type == 0x01: # initialization of clients and initial model
            print("init FL task")
            [self.nodes.append(copy.copy(env.nodes[i])) for i in range(env.numNodes)]
            m = self.FL_manager.setUp(self.nodes)
            self.numNodes = env.numNodes
            act.model = MLModel(modelId=m.modelId,nodeId=m.nodeId,taskId=m.taskId,round=m.round)                      
        if env.type == 0x02 : # selection
            #get the candidated nodes
            self.numNodes = env.numNodes
            print("numNodes", self.numNodes)
            self.nodes = []
            [self.nodes.append(copy.copy(env.nodes[i])) for i in range(0,env.numNodes)]   
            
            print('select trainers and aggregators')
            if config.nodes.selection == "score" :
                self.exactSelection(act,config)
            elif config.nodes.selection == "DRL":
                self.DRLSelection(act)
            else : # "hybrid" 
                self.hybridSelection(act)

            # self.selectedAggregators = copy.deepcopy(act.selectedAggregators)
            # self.selectedTrainers = copy.deepcopy(act.selectedTrainers)
            self.selectedAggregators = []
            self.selectedTrainers = []
            [self.selectedAggregators.append(act.selectedAggregators[index]) for index in range(0,act.numAggregators) ]
            [self.selectedTrainers.append(act.selectedTrainers[index]) for index in range(0, act.numTrainers)]
        if env.type == 0x03 : #training
            print('local training')
            lm = self.FL_manager.start_round(self.selectedTrainers, config.nodes.participants_per_round)
            act.numLocalModels = len(lm)
            for i in range(0,len(lm)) :
                act.localModels[i] =  MLModel(modelId=lm[i].modelId,nodeId=lm[i].nodeId,taskId=lm[i].taskId,round=lm[i].round,type=lm[i].type, positiveVote=lm[i].positiveVote, negativeVote=lm[i].negativeVote,evaluator1=lm[i].evaluator1, evaluator2=lm[i].evaluator2,evaluator3=lm[i].evaluator3,aggregated=lm[i].aggregated, aggModelId=lm[i].aggModelId, accuracy=lm[i].accuracy, acc1=lm[i].acc1, acc2=lm[i].acc2, acc3=lm[i].acc3)
            
        if env.type == 0x04 : #evaluation
            print("model evaluation")
            nodeId = env.nodeId
            m = copy.copy(env.models[0])
            if m.type == 0 : #local
                model = self.FL_manager.evaluateLocal(nodeId, m)
            elif m.type == 1 : #intermediaire
                model = self.FL_manager.evaluateIntermediaire(nodeId, m)
            act.model = MLModel(modelId=model.modelId,nodeId=model.nodeId,taskId=model.taskId,round=model.round,type=model.type, positiveVote=model.positiveVote, negativeVote=model.negativeVote,evaluator1=model.evaluator1, evaluator2=model.evaluator2,evaluator3=model.evaluator3,aggregated=model.aggregated, aggModelId=model.aggModelId, accuracy=model.accuracy, acc1=model.acc1, acc2=model.acc2, acc3=model.acc3)
        if env.type == 0x05 : #aggregation  
            print("**************** model aggregation ********************")
            nodeId = env.nodeId
            numModels = env.numNodes
            aggType = env.numRounds
            models = []
            for i in range(0,numModels):
                models.append(copy.copy(env.models[i]))
            model = self.FL_manager.aggregate(nodeId, models,aggType)
            act.model = MLModel(modelId=model.modelId,nodeId=model.nodeId,taskId=model.taskId,round=model.round,type=model.type, positiveVote=model.positiveVote, negativeVote=model.negativeVote,evaluator1=model.evaluator1, evaluator2=model.evaluator2,evaluator3=model.evaluator3,aggregated=model.aggregated, aggModelId=model.aggModelId, accuracy=model.accuracy, acc1=model.acc1, acc2=model.acc2, acc3=model.acc3)
        if env.type == 0x06: #restround
            print("***********  NEW ROUND ***************")   
            n = self.FL_manager.resetRound(self.selectedTrainers,self.selectedAggregators)
            act.numFLNodes = config.nodes.total
            # act.FLNodesInfo = []
            # print("updating all nodes")
            for i in range(0,act.numFLNodes):
                act.FLNodesInfo[i] = FLNodeStruct(
                    nodeId = n[i].nodeId,
                    availability = n[i].availability,
                    honesty = n[i].honesty,
                    datasetSize = n[i].datasetSize,
                    freq = n[i].freq,
                    transRate = n[i].transRate,
                    task = n[i].task,
                    dropout = n[i].dropout,
                    malicious = n[i].malicious
                ) 
       
        return act

if __name__ == '__main__':

    # Read configuration file
    fl_config = config.Config(args.config)

    ns3Settings = {
        'numNodes' : fl_config.nodes.total,
        'participantsPerRound' : fl_config.nodes.participants_per_round,
        'aggregatorsPerRound' : fl_config.nodes.aggregators_per_round,
        'source' : fl_config.nodes.source,
        'flRounds' : fl_config.fl.rounds,
        'numBCNodes' : fl_config.nodes.bc,
        'targetAccuracy' : fl_config.fl.target_accuracy,
        'testPartition' : fl_config.nodes.test_partition,
        'x' : fl_config.fl.x
    };

    mempool_key = 1001
    mem_size = 1024 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    exp = Experiment(mempool_key, mem_size, 'main', '../../', using_waf=False)
    exp.reset()
    try:
        memblock_key = 2333
        container = AiHelperContainer(fl_config,memblock_key)

        pro = exp.run(setting=ns3Settings, show_output= True)
        while not container.rl.isFinish():
            with container.rl as data:
                if data == None:
                    break
                data.act = container.do(data.env , data.act, fl_config)
                pass            
        pro.wait()
    except Exception as e :
        print("something went wrong")
        print(e)
    finally:    
        del exp

=======
import ctypes
import argparse
import logging
import numpy as np
from py_interface import *
import FL_tasks as fl
import numpy as np
import config
import copy
from typing import List

numMaxNodes = 300
numMaxTrainers = 150  
numMaxAggregators = 150
numMaxBCNodes = 100
numMaxModelsToAgg= 20


# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='../../config.json',
                    help='Federated learning configuration file.')


args = parser.parse_args()


class MLModel(ctypes.Structure):
    _pack_ = 1  # Pack the structure to match the C++ layout
    _fields_ = [
        ("modelId", ctypes.c_int),
        ("nodeId", ctypes.c_int),
        ("taskId", ctypes.c_int),
        ("round", ctypes.c_int),
        ("type", ctypes.c_int),
        ("positiveVote", ctypes.c_int),
        ("negativeVote", ctypes.c_int),
        ("evaluator1", ctypes.c_int),
        ("evaluator2", ctypes.c_int),
        ("evaluator3", ctypes.c_int),
        ("aggregated", ctypes.c_bool),
        ("aggModelId", ctypes.c_int),
        ("accuracy", ctypes.c_double),
        ("acc1", ctypes.c_double),
        ("acc2", ctypes.c_double),
        ("acc3", ctypes.c_double)
    ]

    def get_refrence(self):
        return MLModelRefrence(modelId= self.modelId, nodeId=self.nodeId, taskId=self.taskId, round=self.round)

class MLModelRefrence(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("modelId", ctypes.c_int),
        ("nodeId", ctypes.c_int),
        ("taskId", ctypes.c_int),
        ("round", ctypes.c_int)
    ]

class FLNodeStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("nodeId", ctypes.c_int),
        ("availability", ctypes.c_bool),
        ("honesty", ctypes.c_double),
        ("datasetSize", ctypes.c_int),
        ("freq", ctypes.c_int),
        ("transRate", ctypes.c_int),
        ("task", ctypes.c_int),
        ("dropout", ctypes.c_bool),
        ("malicious", ctypes.c_bool)
    ]
    def learning_cost(self):
        c = self.datasetSize / self.freq 
        return c

    def communication_cost(self, m):
        c = m / self.transRate  
        return c

    def get_cost(self, w):
        return self.learning_cost() + self.communication_cost(w)

    def learning_cost(self,datasetSize,freq):
        return datasetSize / freq
        

    def communication_cost(self, m, transRate):
        return m / transRate  
        

    def get_cost(self,datasetSize,freq,transRate, w):
        
        return self.learning_cost(datasetSize,freq) + self.communication_cost(w,transRate)

class BCNodeStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("nodeId", ctypes.c_int),
        ("task", ctypes.c_int),
    ]


class AiHelperEnv(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type", ctypes.c_int),
        ("nodeId", ctypes.c_int),
        ("models",  MLModel * numMaxModelsToAgg),
        ("numNodes", ctypes.c_int),
        ("numTrainers", ctypes.c_int),
        ("numAggregators", ctypes.c_int),
        ("numRounds", ctypes.c_int),
        ("nodes", FLNodeStruct * numMaxNodes)
    ]

class AiHelperAct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("model", MLModel),
        ("numLocalModels", ctypes.c_int),
        ("localModels", MLModel * numMaxTrainers),
        ("selectedTrainers", ctypes.c_int * numMaxTrainers),
        ("selectedAggregators", ctypes.c_int * numMaxAggregators),
        ("numTrainers", ctypes.c_int),
        ("numAggregators", ctypes.c_int),
        ("numFLNodes", ctypes.c_int),
        ("FLNodesInfo", FLNodeStruct * numMaxNodes)
    ]

class AiHelperContainer:
    use_ns3ai = True
    nodes: List[FLNodeStruct]= []
    # FL_manager: fl.FLManager
    def __init__(self, config:config.Config, uid: int = 2333) -> None:
        self.rl = Ns3AIRL(uid, AiHelperEnv, AiHelperAct)
        self.FL_manager = fl.FLManager(config)
        pass

    def exactSelection(self, act,config):
        alpha = config.fl.alpha 
        nodes_scores = []
        
        for node in self.nodes:

            if node.availability: 
                score = alpha * node.honesty - (1 - alpha) * node.get_cost(node.datasetSize, node.freq, node.transRate, config.model.size)
                nodes_scores.append({"nodeId":node.nodeId , "score": score})
        sorted_indexes =sorted(nodes_scores, key=lambda x: x["score"], reverse=True) # np.argsort([score for _, score in node_scores])[::-1]
        # print(str(sorted_indexes))
        # act.selectedAggregators= sorted_indexes[0:config.nodes.aggregators_per_round]
        # act.selectedTrainers= sorted_indexes[config.nodes.aggregators_per_round : config.nodes.aggregators_per_round+config.nodes.participants_per_round]
        for i in range(0,config.nodes.aggregators_per_round):
            act.selectedAggregators[i]= sorted_indexes[i]["nodeId"]
        for i in range(0, config.nodes.participants_per_round):
            act.selectedTrainers[i] = sorted_indexes[i+config.nodes.aggregators_per_round]["nodeId"]   
        act.numAggregators = config.nodes.aggregators_per_round
        act.numTrainers = config.nodes.participants_per_round
        

    def DRLSelection(self, act):
        pass

    def hybridSelection(self, act):
        pass


    def do(self, env:AiHelperEnv, act:AiHelperAct, config:config.Config) -> AiHelperAct:
      
        if env.type == 0x01: # initialization of clients and initial model
            print("init FL task")
            [self.nodes.append(copy.copy(env.nodes[i])) for i in range(env.numNodes)]
            m = self.FL_manager.setUp(self.nodes)
            self.numNodes = env.numNodes
            act.model = MLModel(modelId=m.modelId,nodeId=m.nodeId,taskId=m.taskId,round=m.round)                      
        if env.type == 0x02 : # selection
            #get the candidated nodes
            self.numNodes = env.numNodes
            print("numNodes", self.numNodes)
            self.nodes = []
            [self.nodes.append(copy.copy(env.nodes[i])) for i in range(0,env.numNodes)]   
            
            print('select trainers and aggregators')
            if config.nodes.selection == "score" :
                self.exactSelection(act,config)
            elif config.nodes.selection == "DRL":
                self.DRLSelection(act)
            else : # "hybrid" 
                self.hybridSelection(act)

            # self.selectedAggregators = copy.deepcopy(act.selectedAggregators)
            # self.selectedTrainers = copy.deepcopy(act.selectedTrainers)
            self.selectedAggregators = []
            self.selectedTrainers = []
            [self.selectedAggregators.append(act.selectedAggregators[index]) for index in range(0,act.numAggregators) ]
            [self.selectedTrainers.append(act.selectedTrainers[index]) for index in range(0, act.numTrainers)]
        if env.type == 0x03 : #training
            print('local training')
            lm = self.FL_manager.start_round(self.selectedTrainers, config.nodes.participants_per_round)
            act.numLocalModels = len(lm)
            for i in range(0,len(lm)) :
                act.localModels[i] =  MLModel(modelId=lm[i].modelId,nodeId=lm[i].nodeId,taskId=lm[i].taskId,round=lm[i].round,type=lm[i].type, positiveVote=lm[i].positiveVote, negativeVote=lm[i].negativeVote,evaluator1=lm[i].evaluator1, evaluator2=lm[i].evaluator2,evaluator3=lm[i].evaluator3,aggregated=lm[i].aggregated, aggModelId=lm[i].aggModelId, accuracy=lm[i].accuracy, acc1=lm[i].acc1, acc2=lm[i].acc2, acc3=lm[i].acc3)
            
        if env.type == 0x04 : #evaluation
            print("model evaluation")
            nodeId = env.nodeId
            m = copy.copy(env.models[0])
            if m.type == 0 : #local
                model = self.FL_manager.evaluateLocal(nodeId, m)
            elif m.type == 1 : #intermediaire
                model = self.FL_manager.evaluateIntermediaire(nodeId, m)
            act.model = MLModel(modelId=model.modelId,nodeId=model.nodeId,taskId=model.taskId,round=model.round,type=model.type, positiveVote=model.positiveVote, negativeVote=model.negativeVote,evaluator1=model.evaluator1, evaluator2=model.evaluator2,evaluator3=model.evaluator3,aggregated=model.aggregated, aggModelId=model.aggModelId, accuracy=model.accuracy, acc1=model.acc1, acc2=model.acc2, acc3=model.acc3)
        if env.type == 0x05 : #aggregation  
            print("**************** model aggregation ********************")
            nodeId = env.nodeId
            numModels = env.numNodes
            aggType = env.numRounds
            models = []
            for i in range(0,numModels):
                models.append(copy.copy(env.models[i]))
            model = self.FL_manager.aggregate(nodeId, models,aggType)
            act.model = MLModel(modelId=model.modelId,nodeId=model.nodeId,taskId=model.taskId,round=model.round,type=model.type, positiveVote=model.positiveVote, negativeVote=model.negativeVote,evaluator1=model.evaluator1, evaluator2=model.evaluator2,evaluator3=model.evaluator3,aggregated=model.aggregated, aggModelId=model.aggModelId, accuracy=model.accuracy, acc1=model.acc1, acc2=model.acc2, acc3=model.acc3)
        if env.type == 0x06: #restround
            print("***********  NEW ROUND ***************")   
            n = self.FL_manager.resetRound(self.selectedTrainers,self.selectedAggregators)
            act.numFLNodes = config.nodes.total
            # act.FLNodesInfo = []
            # print("updating all nodes")
            for i in range(0,act.numFLNodes):
                act.FLNodesInfo[i] = FLNodeStruct(
                    nodeId = n[i].nodeId,
                    availability = n[i].availability,
                    honesty = n[i].honesty,
                    datasetSize = n[i].datasetSize,
                    freq = n[i].freq,
                    transRate = n[i].transRate,
                    task = n[i].task,
                    dropout = n[i].dropout,
                    malicious = n[i].malicious
                ) 
       
        return act

if __name__ == '__main__':

    # Read configuration file
    fl_config = config.Config(args.config)

    ns3Settings = {
        'numNodes' : fl_config.nodes.total,
        'participantsPerRound' : fl_config.nodes.participants_per_round,
        'aggregatorsPerRound' : fl_config.nodes.aggregators_per_round,
        'source' : fl_config.nodes.source,
        'flRounds' : fl_config.fl.rounds,
        'numBCNodes' : fl_config.nodes.bc,
        'targetAccuracy' : fl_config.fl.target_accuracy,
        'testPartition' : fl_config.nodes.test_partition,
        'x' : fl_config.fl.x
    };

    mempool_key = 1001
    mem_size = 1024 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    exp = Experiment(mempool_key, mem_size, 'main', '../../', using_waf=False)
    exp.reset()
    try:
        memblock_key = 2333
        container = AiHelperContainer(fl_config,memblock_key)

        pro = exp.run(setting=ns3Settings, show_output= True)
        while not container.rl.isFinish():
            with container.rl as data:
                if data == None:
                    break
                data.act = container.do(data.env , data.act, fl_config)
                pass            
        pro.wait()
    except Exception as e :
        print("something went wrong")
        print(e)
    finally:    
        del exp
>>>>>>> bc_fl

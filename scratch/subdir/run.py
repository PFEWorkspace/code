import ctypes
import argparse
import logging
from py_interface import *
import FL_tasks as fl
import numpy as np
import config


numMaxNodes = 100 
numMaxTrainers = 50  
numMaxAggregators = 20
numMaxBCNodes = 30
modelSize = 120

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
        ("accuracy", ctypes.c_double)
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
        ("dropout", ctypes.c_bool)
    ]

    def learning_cost(self):
        c = self.datasetSize / self.freq 
        return c

    def communication_cost(self, m):
        c = m / self.transRate  
        return c

    def get_cost(self, w):
        return self.learning_cost() + self.communication_cost(w)

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
        # ("modelId", ctypes.c_int),
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
        ("localModels", MLModelRefrence * numMaxTrainers),
        ("selectedTrainers", ctypes.c_int * numMaxTrainers),
        ("selectedAggregators", ctypes.c_int * numMaxAggregators),
        ("numTrainers", ctypes.c_int),
        ("numAggregators", ctypes.c_int)
    ]

class AiHelperContainer:
    use_ns3ai = True

    def __init__(self, uid: int = 2333) -> None:
        self.rl = Ns3AIRL(uid, AiHelperEnv, AiHelperAct)
        pass

    def exactSelection(self, act):
        alpha = config.fl.alpha # to add to config file 
        node_scores = [(i, alpha * node.honesty - (1 - alpha) * node.get_cost(node, config.model.size)) for i, node in enumerate(self.nodes)]
        sorted_indexes = np.argsort([score for _, score in node_scores])[::-1]
       
        act.selectedAggregators= sorted_indexes[0:config.nodes.aggregators_per_round]
        act.selectedTrainers= sorted_indexes[config.nodes.aggregators_per_round : config.nodes.aggregators_per_round+config.nodes.participants_per_round]
        act.numAggregators = config.nodes.aggregators_per_round
        act.numTrainers = config.nodes.participants_per_round

    def DRLSelection(self, act):
        pass

    def hybridSelection(self, act):
        pass


    def do(self, env:AiHelperEnv, act:AiHelperAct, config) -> AiHelperAct:
        FL_manager = fl.FLManager(config)
        if env.type == 0x01: # initialization of clients and initial model
            print("init FL task")
            nodesinfo = []
            [nodesinfo.append(env.nodes[i]) for i in range(env.numNodes)]
            m = FL_manager.setUp(nodesinfo)
            self.nodes = nodesinfo
            self.numNodes = env.numNodes
            act.model = MLModel(modelId=m.modelId,nodeId=m.nodeId,taskId=m.taskId,round=m.round)
                       
        if env.type == 0x02 : # selection
            print('select trainers and aggregators')
            if config.nodes.selection == "score" :
                self.exactSelection(act)
            elif config.nodes.selection == "DRL":
                self.DRLSelection(act)
            else : # "hybrid" 
                self.hybridSelection(act)

            self.selectedAggregators = act.selectedAggregators
            self.selectedTrainers = act.selectedTrainers

        if env.type == 0x03 : #training
            lm = FL_manager.start_round(self.selectedTrainers)
            act.numLocalModels = len(lm)
            act.localModels = lm
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
        'x' : fl_config.fl.x
    };

    mempool_key = 1132
    mem_size = 1024 * 2 * 2 * 2 * 2
    exp = Experiment(mempool_key, mem_size, 'main', '../../', using_waf=False)
    exp.reset()
    try:
        memblock_key = 2333
        container = AiHelperContainer(memblock_key)

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


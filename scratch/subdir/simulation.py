import ctypes
from py_interface import *


numMaxNodes = 100 
numMaxTrainers = 50  
numMaxAggregators = 20

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

class AiHelperEnv(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type", ctypes.c_int),
        ("nodeId", ctypes.c_int),
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
        ("selectedAggregators", ctypes.c_int * numMaxAggregators)
    ]

class AiHelperContainer:
    use_ns3ai = True

    def __init__(self, uid: int = 2333) -> None:
        self.rl = Ns3AIRL(uid, AiHelperEnv, AiHelperAct)
        pass

    def do(self, env:AiHelperEnv, act:AiHelperAct) -> AiHelperAct:
        if env.type == 0x01: # initialization of clients and initial model
            m = MLModel(modelId=1, nodeId=-1, taskId=1, round=0)
            act.model = m
            # create the fl nodes and distribute data 
        return act

if __name__ == '__main__':
    ns3Settings = {};
    mempool_key = 1234
    mem_size = 2048 * 10
    exp = Experiment(mempool_key, mem_size, 'scratch-simulator', '../../', using_waf=False)
    exp.reset()
    try:
        memblock_key = 2333
        container = AiHelperContainer(memblock_key)

        pro = exp.run(setting=ns3Settings, show_output= True)
        print ("run scratch simulator with these settings ", ns3Settings)
        while not container.rl.isFinish():
            with container.rl as data:
                if data == None:
                    break
                data.act = container.do(data.env , data.act)
                pass            
        pro.wait()
    except Exception as e :
        print("something went wrong")
        print(e)
    finally:    
        del exp


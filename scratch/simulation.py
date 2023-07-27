import ctypes
from py_interface import *


numMaxNodes = 100  
numMaxTrainers = 50  
numMaxAggregators = 20

class MLModel(ctypes.Structure):
    _pack_ = 1  # Pack the structure to match the C++ layout
    _fields_ = [
        ("modelId", ctypes.c_int8),
        ("nodeId", ctypes.c_int8),
        ("taskId", ctypes.c_int8),
        ("round", ctypes.c_int8),
        ("type", ctypes.c_int8),
        ("positiveVote", ctypes.c_int8),
        ("negativeVote", ctypes.c_int8),
        ("evaluator1", ctypes.c_int8),
        ("evaluator2", ctypes.c_int8),
        ("evaluator3", ctypes.c_int8),
        ("aggregated", ctypes.c_bool),
        ("aggModelId", ctypes.c_int8),
        ("accuracy", ctypes.c_double)
    ]

class MLModelRefrence(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("modelId", ctypes.c_int8),
        ("nodeId", ctypes.c_int8),
        ("taskId", ctypes.c_int8),
        ("round", ctypes.c_int8)
    ]

class FLNodeStruct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("nodeId", ctypes.c_int8),
        ("availability", ctypes.c_bool),
        ("honesty", ctypes.c_double),
        ("datasetSize", ctypes.c_int8),
        ("freq", ctypes.c_int8),
        ("transRate", ctypes.c_int8),
        ("task", ctypes.c_int8),
        ("dropout", ctypes.c_bool)
    ]

class AiHelperEnv(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type", ctypes.c_int8),
        ("nodeId", ctypes.c_int8),
        ("numNodes", ctypes.c_int8),
        ("numTrainers", ctypes.c_int8),
        ("numAggregators", ctypes.c_int8),
        ("numRounds", ctypes.c_int8),
        ("nodes", FLNodeStruct * numMaxNodes)
    ]

class AiHelperAct(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("model", MLModel),
        ("numLocalModels", ctypes.c_int8),
        ("localModels", MLModelRefrence * numMaxTrainers),
        ("selectedTrainers", ctypes.c_int8 * numMaxTrainers),
        ("selectedAggregators", ctypes.c_int8 * numMaxAggregators)
    ]

class AiHelperContainer:
    use_ns3ai = True

    def __init__(self, uid: int = 2333) -> None:
        self.rl = Ns3AIRL(uid, AiHelperEnv, AiHelperAct)
        pass

    def do(self, env:AiHelperEnv, act:AiHelperAct) -> AiHelperAct:
        if env.type == 0x01: # initialization of clients and initial model
            m = MLModel(modelId=1, nodeId=1, taskId=1, round=0,type=0,positiveVote=0,negativeVote=0,evaluator1=0,evaluator2=0,evaluator3=0,aggrerated=True,aggModelId=0,accuracy=0.0)
            act.model = m
            act.numLocalModels = 20
        return act

if __name__ == '__main__':
    ns3Settings = {};
    mempool_key = 1234
    mem_size = 2048 * 20
    exp = Experiment(mempool_key, mem_size, 'scratch-simulator', '../', using_waf=False)
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


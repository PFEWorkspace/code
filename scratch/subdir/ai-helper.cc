
#include "ai-helper.h"

#include "ns3/log.h"
#include "ns3/core-module.h"

#include "Blockchain.h"

namespace ns3{

NS_LOG_COMPONENT_DEFINE ("AiHelper");
AiHelper* AiHelper::instance = nullptr;
const uint16_t m_ns3ai_id = 2333;
// GlobalValue gNS3AIRLUID (
//   "NS3AIRLUID", 
//   "UID of Ns3AIRL",
//   UintegerValue (2333),
//   MakeUintegerChecker<uint16_t> ()
// );

// TypeId AiHelper::GetTypeId() {
//   static TypeId tid = TypeId("FLNode")
//                           .AddConstructor<AiHelper>();
//     return tid ;                      
// }
                        
AiHelper::AiHelper(): Ns3AIRL<AiHelperEnv, AiHelperAct> (m_ns3ai_id)
{ 
    SetCond (2, 0);
    SetTraining(false);
    numLocalModels = 0 ;
    NS_LOG_FUNCTION_NOARGS();
}

// AiHelper::~AiHelper(){
//     SetFinish();
//     NS_LOG_FUNCTION_NOARGS();
// }


MLModelRefrence
AiHelper::GetModelReference(MLModel model){
    MLModelRefrence mr;
    mr.modelId = model.modelId;
    mr.nodeId = model.nodeId;
    mr.taskId = model.taskId;
    mr.round = model.round;
    return mr;
}

MLModelRefrence
AiHelper::initializeFL(FLNodeStruct *nodes, int& numNodes){
    NS_LOG_FUNCTION_NOARGS();
    
    //set input
    auto env = EnvSetterCond();
    env->type = 0x01; 
    env->numNodes = numNodes ;
    for(int i=0; i<numNodes; i++){env->nodes[i] = nodes[i];}
    SetCompleted();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id)); // to get the momory version

    //get output
    auto act = ActionGetter();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id));
    MLModelRefrence initialModel = GetModelReference(act->model);
    GetCompleted();
      
    return initialModel ;
}

void AiHelper::Selection () 
{
    NS_LOG_FUNCTION_NOARGS();
    Blockchain* bc = Blockchain::getInstance() ; 
    //set input
    auto env = EnvSetterCond();
    env->type = 0x02; 
    env->numNodes = bc->getNumFLNodes();
    for(int i=0; i<bc->getNumFLNodes(); i++){env->nodes[i] = bc->GetNodeInfo(i);}
    SetCompleted();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id)); // to get the momory version

    //get output
    auto act = ActionGetter();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id));
    bc->SetAggregators(act->selectedAggregators, act->numAggregators);
    bc->SetTrainers(act->selectedTrainers, act->numTrainers);
    GetCompleted();
    
}


MLModel
AiHelper::train(int nodeid){
    NS_LOG_FUNCTION_NOARGS();    
    while(GetTraining()){ // the python side is training under anather node's request
        // while training wait till it finish
    }
    // the python side is not training
    if(numLocalModels==0 && !GetTraining()){ //first time calling train
        // launch training in pythonside
        SetTraining(true);
        auto env = EnvSetterCond();
        env->type = 0x03;  
        SetCompleted();
        NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id)); // to get the momory version

        //get output
        auto act = ActionGetter();
        NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id));
        numLocalModels = act->numLocalModels;
        for(int i=0;i<numLocalModels;i++){localModels[i] = act->localModels[i];}
        GetCompleted();
        SetTraining(false);   
    }
    //training is done and models are saved in localmodels

    //return the corresponding mlmodel
    
    return GetLocalModel(nodeid);
}
MLModel
AiHelper::GetLocalModel(int nodeid){
    MLModel model= MLModel();
    int i=0;
    while(i<numLocalModels){
        if(localModels[i].nodeId == nodeid){
            model=localModels[i];
            break;
        }
        i++;
    }
    
    return model;
}

MLModel
AiHelper::evaluate(MLModel model, int aggId){
    //set input
    auto env = EnvSetterCond();
    env->type = 0x04; 
    env->nodeId = aggId;
    env->models[0] = model ;
    SetCompleted();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id)); // to get the momory version

    //get output
    auto act = ActionGetter();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id));
    MLModel model = act->model;
    GetCompleted();
    return model;
}

MLModel
AiHelper::aggregate(std::vector<MLModel> models, int aggId){
    //set input
    auto env = EnvSetterCond();
    env->type = 0x05; 
    env->nodeId = aggId ;
    for(uint i=0; i<models.size(); i++){env->models[i] = models[i];}
    SetCompleted();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id)); // to get the momory version

    //get output
    auto act = ActionGetter();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id));
    MLModel model = act->model;
    GetCompleted();
    return model;
}
}
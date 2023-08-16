
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
     int versionBefore=0;
    int versionAfter=0;
    //set input
    
    auto env = EnvSetterCond();
    env->type = 0x01; 
    env->numNodes = numNodes ;
    for(int i=0; i<numNodes; i++){env->nodes[i] = nodes[i];}
    SetCompleted();
    versionBefore = (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id);
    NS_LOG_INFO("Version before: " << versionBefore); // to get the momory version

    //get output
    
    auto act = ActionGetterCond();
    // restart:
    versionAfter= (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id);
    NS_LOG_INFO("Version after: " << versionAfter);
    // if(versionBefore==versionAfter){
    //     goto restart;
    // }else{
        MLModelRefrence initialModel = GetModelReference(act->model);
        GetCompleted();
      
    return initialModel ;
    // }
}

void AiHelper::Selection () 
{
     int versionBefore=0;
    int versionAfter=0;
    NS_LOG_FUNCTION_NOARGS();
    
    Blockchain* bc = Blockchain::getInstance() ; 
    //set input
    
    auto env = EnvSetterCond();
    env->type = 0x02; 
    env->numNodes = bc->getNumFLNodes();
    for(int i=0; i<bc->getNumFLNodes(); i++){env->nodes[i] = bc->GetNodeInfo(i);}
    SetCompleted();
    versionBefore = (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id);
    NS_LOG_INFO("Version before: " << versionBefore); // to get the momory version

    // restart:
    auto act = ActionGetterCond();
    
    versionAfter= (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id);
    NS_LOG_INFO("Version after: " << versionAfter);
    // if(versionBefore==versionAfter){
    //     goto restart;
    // }else{
        bc->SetAggregators(act->selectedAggregators, act->numAggregators);
        bc->SetTrainers(act->selectedTrainers, act->numTrainers);
        GetCompleted();
    // }
    
}


MLModel
AiHelper::train(int nodeid){
    std::lock_guard<std::mutex> lock(mtx);
    NS_LOG_FUNCTION_NOARGS();   
    int versionBefore=0;
    int versionAfter=0;

    while(GetTraining()){ // the python side is training under anather node's request
        // while training wait till it finish
    }
    // the python side is not training
    if(numLocalModels==0 && !GetTraining()){ //first time calling train
        // launch training in pythonside
        SetTraining(true);
        // restart:
        auto env = EnvSetterCond();
        env->type = 0x03; 
        SetCompleted(); 
        versionBefore = (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id);
        NS_LOG_INFO("Version before training: " << versionBefore); // to get the momory version

        //get output
       
        auto act = ActionGetterCond();        
        versionAfter= (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id);
        NS_LOG_INFO("Version after training: " << versionAfter);
        // if(versionBefore==versionAfter){
        //     goto restart;
        // }else{
            numLocalModels = act->numLocalModels;
            for(int i=0;i<numLocalModels;i++){localModels[i] = act->localModels[i];}
            GetCompleted();
            SetTraining(false); 
        // }  
    }
    
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
    std::lock_guard<std::mutex> lock(mtx);

    NS_LOG_FUNCTION_NOARGS();
    int versionBefore=0;
    int versionAfter=0;
    //set input
//    restart:
    auto env = EnvSetterCond();
    env->type = 0x04; 
    env->nodeId = aggId;
    env->models[0] = model ;
    SetCompleted();
    versionBefore = (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id);
    NS_LOG_INFO("Version: " << versionBefore); // to get the momory version

    //get output
    
    auto act = ActionGetterCond();
    versionAfter= (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id);
    NS_LOG_INFO("Version: " << versionAfter);
    //  if(versionBefore==versionAfter){
    //     goto restart;
    //  }else{
        MLModel evalModel;
        evalModel = act->model;
        NS_LOG_INFO("before get completed");
        GetCompleted();
        NS_LOG_INFO("after get completed");
        return evalModel;
    //  }  
}

MLModel
AiHelper::aggregate(std::vector<MLModel> models, int aggId, int aggType){
    std::lock_guard<std::mutex> lock(mtx);
    NS_LOG_FUNCTION_NOARGS();
    //set input
    
    auto env = EnvSetterCond();
    env->type = 0x05; 
    env->nodeId = aggId ;
    env->numNodes = models.size(); // use the numNodes for the number of models to aggregate, just bcs in this type numNodes is empty
    env->numRounds = aggType; //used the numRounds for the type cus its empty
    for(uint i=0; i<models.size(); i++){env->models[i] = models[i];}
    SetCompleted();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id)); // to get the momory version

    //get output
    auto act = ActionGetterCond();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id));
    MLModel model = act->model;
    GetCompleted();
    return model;
}
}
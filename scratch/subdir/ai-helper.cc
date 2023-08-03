
#include "ai-helper.h"

#include "ns3/log.h"
#include "ns3/core-module.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace ns3{

NS_LOG_COMPONENT_DEFINE ("AiHelper");
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
    NS_LOG_FUNCTION_NOARGS();

    // UintegerValue uv;
    // gNS3AIRLUID.GetValue (uv);
    // m_ns3ai_id = uv.Get ();
    // NS_LOG_UNCOND("m_ns3ai_id " << m_ns3ai_id);

    // m_ns3ai_mod = new Ns3AIRL<AiHelperEnv, AiHelperAct> (m_ns3ai_id);
    SetCond (2, 0);
    
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

void AiHelper::ExactSelection () 
{
 NS_LOG_FUNCTION_NOARGS();
    
    //set input
    auto env = EnvSetterCond();
    env->type = 0x02; 
    // env->numNodes = numNodes ;

    // for(int i=0; i<numNodes; i++){env->nodes[i] = nodes[i];}
    SetCompleted();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id)); // to get the momory version

    //get output
    auto act = ActionGetter();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id));
    Blockchain* blockchain = Blockchain::getInstance() ; 
    int agg = act->numAggregators ;
    int trai = act->numTrainers ;
    blockchain->SetAggregators(act->selectedAggregators, act->numAggregators);
    blockchain->SetTrainers(act->selectedTrainers, act->numTrainers);
    GetCompleted();
   // NS_LOG_INFO("from python "<< initialModel.modelId );
}


MLModel
AiHelper::train(int nodeid, int globalModel){
    NS_LOG_FUNCTION_NOARGS();
    
    //set input
    auto env = EnvSetterCond();
    env->type = 0x02; 
    env->nodeId = nodeid;
    // env->modelId = globalModel;
    SetCompleted();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id)); // to get the momory version

    //get output
    auto act = ActionGetter();
    NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id));
    MLModel model = act->model;
    GetCompleted();
    NS_LOG_INFO("from python "<< model.modelId );
   
    return model ;
}

}
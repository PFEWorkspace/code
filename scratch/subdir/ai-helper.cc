
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

FLNodeStruct*
AiHelper::GetNodesFromFile(const std::string& filename,  int& numNodes){

    FLNodeStruct* nodeList = nullptr;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    // Ignore the header line
    std::string line;
    std::getline(file, line);

    // Read data line by line and create FLNodeStruct nodes
    int count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);

        FLNodeStruct node;
        std::string field;

        std::getline(ss, field, ','); // Read the ID field
        node.nodeId = std::stoi(field);

        std::getline(ss, field, ','); // Read the Availability field
        node.availability = (field == "true");

        std::getline(ss, field, ','); // Read the Honesty field
        node.honesty = std::stod(field);

        std::getline(ss, field, ','); // Read the Dataset Size field
        node.datasetSize = std::stoi(field);

        std::getline(ss, field, ','); // Read the Frequency field
        node.freq = std::stoi(field);

        std::getline(ss, field, ','); // Read the Transmission Rate field
        node.transRate = std::stoi(field);

        std::getline(ss, field, ','); // Read the Task field
        node.task = std::stoi(field);

        std::getline(ss, field); // Read the Dropout field
        node.dropout = (field == "true");

        // Resize the array for each new node
        nodeList = (FLNodeStruct*)realloc(nodeList, (count + 1) * sizeof(FLNodeStruct));
        nodeList[count++] = node;
    }

    file.close();
    numNodes = count;
    return nodeList;
}

MLModelRefrence
AiHelper::GetModelReference(MLModel model){
    NS_LOG_INFO("model info "<< model.modelId);
    MLModelRefrence mr;
    mr.modelId = model.modelId;
    mr.nodeId = model.nodeId;
    mr.taskId = model.taskId;
    mr.round = model.round;
    return mr;
}

MLModelRefrence
AiHelper::initializeFL(const std::string& filename){
    NS_LOG_FUNCTION_NOARGS();
    int numNodes = 0 ;
    FLNodeStruct* nodes = GetNodesFromFile(filename, numNodes);   
    NS_LOG_INFO("the number of initialized nodes is " << numNodes);
    //set input
    auto env = EnvSetterCond();
    env->type = 0x01; 
    env->numNodes = numNodes ;
    for(int i=0; i<numNodes; i++){env->nodes[i] = nodes[i];}
    SetCompleted();
    // NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id)); // to get the momory version

    //get output
    auto act = ActionGetter();
    // NS_LOG_INFO("Version: " << (int)SharedMemoryPool::Get()->GetMemoryVersion(m_ns3ai_id));
    MLModelRefrence initialModel = GetModelReference(act->model);
    GetCompleted();
    // NS_LOG_INFO("modelid"<< initialModel.modelId);
   
    return initialModel ;
}
}
#ifndef AI_HELPER_H
#define AI_HELPER_H

#include "ns3/ns3-ai-module.h"


namespace ns3 {
const int8_t numMaxNodes = 100;
const int8_t numMaxTrainers = 50;
const int8_t numMaxAggregators = 20;

    struct MLModel{
        int8_t modelId; // used to access the model on the file directly
        int8_t nodeId; // the id of the trainer if it's a local model, the aggregatorID if it's an intermediate or global model, -1 if it's the initial model
        int8_t taskId; // the id of the FL task
        int8_t round; // the number of the FL round
        int8_t type; // 1:local, 2:intermediaire or 3:global
        int8_t positiveVote; // counter for the validations
        int8_t negativeVote; //counter for the non validations
        int8_t evaluator1 ; // the id of the first evaluator node
        int8_t evaluator2; // the id of the second evaluator
        int8_t evaluator3; // the id of the third evaluator, if there is one
        bool aggregated; // true if it was assigned to be aggregated, else false
        int8_t aggModelId; // the id of the aggregated model including this one (level+1)
        double accuracy; // the accuracy of the model
        //TODO decide the type weights;
    }Packed;

    struct MLModelRefrence{
        int8_t modelId;
        int8_t nodeId;
        int8_t taskId;
        int8_t round;
    }Packed;

    struct FLNodeStruct
    {
        int8_t nodeId;
        bool availability;
        double honesty;
        int8_t datasetSize;
        int8_t freq;
        int8_t transRate;
        int8_t task; // 0 : train , 1: aggregate , 2: evaluate
        bool dropout; // true if the node will be droping out of its task
    } Packed;

    struct AiHelperEnv
    {
        int8_t type; //1: initialisation, 2:selection, 3:train, 4:evaluation, 5:aggregation
        int8_t nodeId; // used for evaluation or aggregation, to know which node is launching this task
        int8_t numNodes;
        int8_t numTrainers;
        int8_t numAggregators;
        int8_t numRounds;  
        FLNodeStruct nodes[numMaxNodes];
      

    } Packed;

    struct AiHelperAct
    {
        MLModel model; // used in initialisation, evaluation and aggregation
        int8_t numLocalModels ; // could be diffrent from the trainers number in case of dropout
        MLModelRefrence localModels[numMaxTrainers]; // list of the refrences of the local models generated after a training task, just the refrence cus after the training no need to know all the info about the model, they will be filled automaticaly from c++ side and the next task is directly afecting it to evaluation
        int8_t selectedTrainers[numMaxTrainers]; // list of selected nodes for the training task
        int8_t selectedAggregators[numMaxAggregators]; // list of the selected nodes for the evaluation and aggregation task
    } Packed;

    class AiHelper
    {
        public:
        // static TypeId GetTypeId(void);
        AiHelper();
        virtual ~AiHelper();

        MLModelRefrence initializeFL(const std::string& filename);


        private:
        
        FLNodeStruct* GetNodesFromFile(const std::string& filename,  int& numNodes);
        MLModelRefrence GetModelReference(MLModel model);
        uint16_t m_ns3ai_id;
        Ns3AIRL<AiHelperEnv, AiHelperAct> * m_ns3ai_mod;        

    };

}
#endif
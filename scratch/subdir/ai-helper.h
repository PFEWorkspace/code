#ifndef AI_HELPER_H
#define AI_HELPER_H

#include "ns3/ns3-ai-module.h"


namespace ns3 {
const int numMaxNodes = 100;
const int numMaxTrainers = 50;
const int numMaxAggregators = 20;

    struct MLModel{
        int modelId; // used to access the model on the file directly
        int nodeId; // the id of the trainer if it's a local model, the aggregatorID if it's an intermediate or global model, -1 if it's the initial model
        int taskId; // the id of the FL task
        int round; // the number of the FL round
        int type; // 1:local, 2:intermediaire or 3:global
        int positiveVote; // counter for the validations
        int negativeVote; //counter for the non validations
        int evaluator1 ; // the id of the first evaluator node
        int evaluator2; // the id of the second evaluator
        int evaluator3; // the id of the third evaluator, if there is one
        bool aggregated; // true if it was assigned to be aggregated, else false
        int aggModelId; // the id of the aggregated model including this one (level+1)
        double accuracy; // the accuracy of the model
        //TODO decide the type weights;
    }Packed;

    struct MLModelRefrence{
        int modelId;
        int nodeId;
        int taskId;
        int round;
    }Packed;

    struct FLNodeStruct
    {
        int nodeId;
        bool availability;
        double honesty;
        int datasetSize;
        int freq;
        int transRate;
        int task; // 0 : train , 1: aggregate , 2: evaluate
        bool dropout; // true if the node will be droping out of its task
    } Packed;

    struct AiHelperEnv
    {
        int type; //1: initialisation, 2:selection, 3:train, 4:evaluation, 5:aggregation
        int nodeId; // used for evaluation or aggregation, to know which node is launching this task
        int numNodes;
        int numTrainers;
        int numAggregators;
        int numRounds;  
        FLNodeStruct nodes[numMaxNodes];
      

    } Packed;

    struct AiHelperAct
    {
        MLModel model; // used in initialisation, evaluation and aggregation
        int numLocalModels ; // could be diffrent from the trainers number in case of dropout
        MLModelRefrence localModels[numMaxTrainers]; // list of the refrences of the local models generated after a training task, just the refrence cus after the training no need to know all the info about the model, they will be filled automaticaly from c++ side and the next task is directly afecting it to evaluation
        int selectedTrainers[numMaxTrainers]; // list of selected nodes for the training task
        int selectedAggregators[numMaxAggregators]; // list of the selected nodes for the evaluation and aggregation task
    } Packed;

    class AiHelper: public Ns3AIRL<AiHelperEnv, AiHelperAct> 
    {
        public:
        // static TypeId GetTypeId(void);
        AiHelper();
        
        MLModelRefrence initializeFL(FLNodeStruct *nodes, int& numNodes);


        private:
        
        // FLNodeStruct* GetNodesFromFile(const std::string& filename,  int& numNodes);
        MLModelRefrence GetModelReference(MLModel model);
      
        // Ns3AIRL<AiHelperEnv, AiHelperAct> * m_ns3ai_mod;        

    };

}
#endif
#ifndef AI_HELPER_H
#define AI_HELPER_H

#include "ns3/ns3-ai-module.h"


#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <mutex>

#include "../../rapidjson/document.h"
#include "../../rapidjson/error/en.h"
#include "../../rapidjson/writer.h"
#include "../../rapidjson/stringbuffer.h"

namespace ns3 {

const int numMaxNodes = 1000;
const int numMaxTrainers = 200;
const int numMaxAggregators = 100;
const int numMaxBCNodes = 100;
const int numMaxModelsToAgg = 20;

    struct MLModel{
        int modelId; // used to access the model on the file directly
        int nodeId; // the id of the trainer if it's a local model, the aggregatorID if it's an intermediate or global model, -1 if it's the initial model
        int taskId; // the id of the FL task 
        int round; // the number of the FL round
        int type; // 0:local, 1:intermediaire or 2:global, the enum MODELTYPE
        int positiveVote; // counter for the validations
        int negativeVote; //counter for the non validations
        int evaluator1 ; // the id of the first evaluator node
        int evaluator2; // the id of the second evaluator
        int evaluator3; // the id of the third evaluator, if there is one
        bool aggregated; // true if it was assigned to be aggregated, else false
        int aggModelId; // the id of the aggregated model including this one (level+1)
        double accuracy; // the accuracy of the model
        double acc1; // the accuracy obtained from 1st evaluator
        double acc2;// the accuracy obtained from 2nd evaluator
        double acc3; // the accuracy obtained from 3rd evaluator
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
        int task; // 0 : train , 1: aggregate , 2: evaluate, the enum TASK
        bool dropout; // true if the node will be droping out of its task
    } Packed;

     struct BCNodeStruct
    {
        int nodeId;
        int task; 

    } Packed;

    struct AiHelperEnv
    {
        int type; //1: initialisation, 2:selection, 3:train, 4:evaluation, 5:aggregation
        int nodeId; // used for evaluation or aggregation, to know which node is doing this task
        MLModel models[numMaxModelsToAgg]; // the model to evaluate or aggregate
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
        MLModel localModels[numMaxTrainers]; // list of the refrences of the local models generated after a training task, just the refrence cus after the training no need to know all the info about the model, they will be filled automaticaly from c++ side and the next task is directly afecting it to evaluation
        int selectedTrainers[numMaxTrainers]; // list of selected nodes for the training task
        int selectedAggregators[numMaxAggregators]; // list of the selected nodes for the evaluation and aggregation task
        int numTrainers;
        int numAggregators;
    } Packed;

    class AiHelper: public Ns3AIRL<AiHelperEnv, AiHelperAct> 
    {
        public:
        std::mutex mtx;

        static AiHelper* getInstance(){
            if (instance==nullptr){
                instance = new AiHelper();
            }
            return instance;
        }
        
        
        MLModelRefrence initializeFL(FLNodeStruct *nodes, int& numNodes);
        void Selection () ;
        MLModel train(int nodeid);
        MLModel GetLocalModel(int nodeid);
        MLModel evaluate(MLModel model, int aggId);
        MLModel aggregate(std::vector<MLModel> models, int aggId, int aggType);
        //setters and getters
        void SetTraining(bool train){training=train;};
        double GetTraining() const{return training;};

        private:
        
        bool training ;
        int numLocalModels ;
        MLModel localModels[numMaxTrainers];
        static AiHelper* instance;
        AiHelper();
        // FLNodeStruct* GetNodesFromFile(const std::string& filename,  int& numNodes);
        MLModelRefrence GetModelReference(MLModel model);      

    };

}
#endif
#ifndef BLOCKCHAIN_H
#define BLOCKCHAIN_H

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/log.h"
#include "ns3/internet-module.h"
#include "ai-helper.h"


#include <ctime>
#include <string>
#include <mutex>
#include <list>

namespace ns3 {

struct AggregatorsTasks{
    int nodeId;
    int task; // 1 Aggregate, 2:Evaluate , TASK enum
    MLModel models[numMaxModelsToAgg]; // if task is evaluation 1 model in the list
    // std::time_t startTime; // use it to check if it exceeds a certain amount of time to reschedule the task
}Packed;

class Blockchain {

    private:
    std::mutex mtx;
    static Blockchain* instance;
    // Private attributes
    std::string m_filename;
    double targetAcc;
    int maxFLround;
    int actualFLround;
    int numFLNodes;
    int numBCNodes;
    int numAggregators;
    int numTrainers;
    int modelsToAggAtOnce;
    int receivedCandidatures;
    int modelSize;

    // MLModel modelToEval[numMaxNodes];
    std::vector<MLModel> modelToAgreg;
    std::vector<FLNodeStruct> m_nodesInfo;
    std::list<AggregatorsTasks> tasks;
    Ipv4InterfaceContainer nodesFLAdrs;
    Ipv4InterfaceContainer nodesBCAdrs;
    int aggregators[numMaxAggregators];
    int trainers[numMaxTrainers];
    Ptr<UniformRandomVariable> randomBCAdrsStream;
    std::string currentblockId;
    int taskId;
    Time startTrainingTime;
   
    
    // Private constructor and destructor to ensure singleton.
    Blockchain(){
        // Initialize 
        receivedCandidatures = 0 ;
        actualFLround = 0 ;
        firstagg = true;
        lastagg = false;
        
    }


    // Private methods
    void SaveBlockchainToFile();
    void AddTransactionToBlockchain(const rapidjson::Value& transaction);
    std::string GetTimestamp();
    
    // Object's inherited methods
    virtual void DoDispose();

public:

    
    // Singleton pattern: static method to get the instance.
    static Blockchain* getInstance() {
        if(instance==nullptr){
            instance = new Blockchain();
        }
        return instance;
    }
    
    Blockchain(const Blockchain& obj)= delete;
   
    bool firstagg;
    bool lastagg;

    void WriteTransaction(std::string blockId, int nodeId, const rapidjson::Document& message);
    void PrintBlockchain() const;
    Ipv4Address getFLAddress(int nodeId);
    int getFLNodeId(Ipv4Address adrs);
    void AddTask(AggregatorsTasks task);
    bool hasPreviousTask(int nodeid);
    void RemoveTask(int id);
    bool ResetRound(MLModel globalModel);
    
    Ipv4Address getBCAddress();
    int GetAggregatorNotBusy(int eval1, int eval2);
    void SetFLAddressContainer(Ipv4InterfaceContainer container);
    void SetBCAddressContainer(Ipv4InterfaceContainer container);

    void SetTargetAcc(double acc){ targetAcc = acc;};
    double GetTargetAcc(){return targetAcc;};
    void SetRandomBCStream();

    void StartTraining(){
        startTrainingTime = Simulator::Now();
    }
    double GetMaxTrainingDelay();
    double GetTrainingDelay(int nodeId);
    bool MaxDelayPassed();
    double GetStillDelay();
     // Setters
    void SetModelsToAggAtOnce(int x){modelsToAggAtOnce=x;}
    int GetModelsToAggAtOnce(){return modelsToAggAtOnce;}

    void SetAggregators(int aggs[], int num){
        for(int i=0; i<num; i++){aggregators[i]=aggs[i];}
    }

    void SetTrainers(int aggs[], int num){
        for(int i=0; i<num; i++){trainers[i]=aggs[i];}
    }

    void SetCurrentBlockId(){
        currentblockId = std::to_string(taskId)+ "/"+ std::to_string(actualFLround); 
    };

    void SetMaxFLRound(int value) {
        maxFLround = value;
    }

    void SetActualFLRound(int value) {
        actualFLround = value;
    }

    void SetModelSize(int size){
        modelSize = size;
    }

    
    // void SetModelToEval(int index,  const MLModel& value) {
    //     if (index >= 0 && index < numMaxNodes) {
    //         modelToEval[index] = value;
    //     }
    // }

    void AddModelToAgg( const MLModel& value); 

    void removeModelsToAgg(int start, int end){
        modelToAgreg.erase(modelToAgreg.begin()+start, modelToAgreg.begin()+end);
    }

    std::vector<MLModel> getxModelsToAgg(int x){
        std::lock_guard<std::mutex> lock(mtx);
        x = std::min(x, static_cast<int>(modelToAgreg.size()));
        std::vector<MLModel> sublist(modelToAgreg.begin(), modelToAgreg.begin() + x);
        removeModelsToAgg(0,x);
        return sublist;
    }
    int getModelsToAggSize(){return modelToAgreg.size();}

    void SetNodeInfo(int index, const FLNodeStruct& value) {
        
        if (index >= 0 && index < numMaxNodes) {
            m_nodesInfo[index] = value;
        }
    }
    void AddNodeInfo(const FLNodeStruct& value){
        m_nodesInfo.push_back(value);
    }

    void IncReceivedCandidatures(){receivedCandidatures++;}
    int GetReceivedCandidatures(){return receivedCandidatures;}

    void setNumFLNodes(int value) {
        numFLNodes = value;
    }

    void setNumBCNodes(int value) {
        numBCNodes = value;
    }

    void setNumAggregators(int value) {
        numAggregators = value;
    }

    void setNumTrainers(int value) {
        numTrainers = value;
    }
    void setTaskId(int taskid){this->taskId = taskid;}
   
    // Getters
    int getTrainer(int index){return trainers[index];}
    int getAggregator(int index){return aggregators[index];}
    
    int getNumAggTasksAwaiting();
    int getTaskId()const{return this->taskId;}
    
    std::string getCurrentBlockId()const {
        return currentblockId;
    }

    int getNumFLNodes() const {
        return numFLNodes;
    }

    int getNumBCNodes() const {
        return numBCNodes;
    }

    int getNumAggregators() const {
        return numAggregators;
    }

    int getNumTrainers() const {
        return numTrainers;
    }
    
    int GetMaxFLRound() const {
        return maxFLround;
    }

    int GetActualFLRound() const {
        return actualFLround;
    }

    // MLModel GetModelToEval(int index) const {
    //     if (index >= 0 && index < numMaxNodes) {
    //         return modelToEval[index];
    //     }
    //     return MLModel(); // or appropriate default value
    // }

    MLModel GetModelToAgreg(int index) const {
        if (index >= 0 && index < numMaxNodes) {
            return modelToAgreg[index];
        }
        return MLModel(); // or appropriate default value
    }

   
    const FLNodeStruct& GetNodeInfo(int index) const {
        if (index >= 0 && index < numMaxNodes) {
            return m_nodesInfo[index];
        }
        // Return a default or placeholder FLNodeStruct
        return FLNodeStruct();
    }

};

// Blockchain* Blockchain::instance = nullptr;



} // namespace ns3
#endif

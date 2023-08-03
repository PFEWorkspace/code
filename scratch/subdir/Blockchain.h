
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "../../rapidjson/document.h"
#include "ns3/internet-module.h"
#include "ai-helper.h"

namespace ns3 {


// Assuming the maximum number of nodes, BCNodes, aggregators, trainers, and other constants are defined.
const int numMaxNodes = 100;
const int numMaxBCNodes = 50;
const int numMaxAggregators = 10;
const int numMaxTrainers = 20;

class Blockchain {

    private:

    static Blockchain* instance;

    // Private constructor and destructor to ensure singleton.
    Blockchain(){
        // Initialize other members here if needed.
        for (int i = 0; i < numMaxNodes; ++i) {
            modelToEval[i] = 0;
            modelToAgreg[i] = 0;
        }
        for (int i = 0; i < numMaxBCNodes; ++i) {
            notBusyNodes[i] = BCNodeStruct(); // Initialize with default constructor.
        }
        for (int i = 0; i < numMaxNodes; ++i) {
            m_nodesInfo[i] = FLNodeStruct(); // Initialize with default constructor.
        }
    }

   

    // Private attributes
    std::string m_filename;
    int maxFLround;
    int actualFLround;
    int modelToEval[numMaxNodes];
    int modelToAgreg[numMaxNodes];
    BCNodeStruct notBusyNodes[numMaxBCNodes];
    FLNodeStruct m_nodesInfo[numMaxNodes];
    Ipv4InterfaceContainer nodesFLAdrs;
    Ipv4InterfaceContainer nodesBCAdrs;
    int aggregators[numMaxAggregators];
    int trainers[numMaxTrainers];

    // Private methods
    void SaveBlockchainToFile();
    void AddTransactionToBlockchain(const rapidjson::Value& transaction);
    std::string Blockchain::GetTimestamp()const;

    // Object's inherited methods
    virtual void DoDispose();

public:

    
    // Singleton pattern: static method to get the instance.
    static Blockchain* getInstance() {
        if(instance==NULL){
            instance = new Blockchain();
        }
        return instance;
    }
    
    Blockchain(const Blockchain& obj)= delete;
    // Setters and Getters
    void WriteTransaction(uint32_t nodeId);
    void PrintBlockchain() const;
    Ipv4Address getFLAddress(int nodeId);
    Ipv4Address getBCAddress();

    void SetFLAddressContainer(Ipv4InterfaceContainer container);
    void SetBCAddressContainer(Ipv4InterfaceContainer container);

     // Setters

    void SetAggregators(int aggs[], int num){
        for(int i=0; i<num; i++){aggregators[i]=aggs[i];}
    }

    void SetTrainers(int aggs[], int num){
        for(int i=0; i<num; i++){trainers[i]=aggs[i];}
    }

    void SetMaxFLRound(int value) {
        maxFLround = value;
    }

    void SetActualFLRound(int value) {
        actualFLround = value;
    }

    void SetModelToEval(int index, int value) {
        if (index >= 0 && index < numMaxNodes) {
            modelToEval[index] = value;
        }
    }

    void SetModelToAgreg(int index, int value) {
        if (index >= 0 && index < numMaxNodes) {
            modelToAgreg[index] = value;
        }
    }

    void SetNotBusyNode(int index, const BCNodeStruct& value) {
        if (index >= 0 && index < numMaxBCNodes) {
            notBusyNodes[index] = value;
        }
    }

    void SetNodeInfo(int index, const FLNodeStruct& value) {
        if (index >= 0 && index < numMaxNodes) {
            m_nodesInfo[index] = value;
        }
    }

    // Similar setters for other attributes

    // Getters
    int GetMaxFLRound() const {
        return maxFLround;
    }

    int GetActualFLRound() const {
        return actualFLround;
    }

    int GetModelToEval(int index) const {
        if (index >= 0 && index < numMaxNodes) {
            return modelToEval[index];
        }
        return 0; // or appropriate default value
    }

    int GetModelToAgreg(int index) const {
        if (index >= 0 && index < numMaxNodes) {
            return modelToAgreg[index];
        }
        return 0; // or appropriate default value
    }

    const BCNodeStruct& GetNotBusyNode(int index) const {
        if (index >= 0 && index < numMaxBCNodes) {
            return notBusyNodes[index];
        }
        // Return a default or placeholder BCNodeStruct
    }

    const FLNodeStruct& GetNodeInfo(int index) const {
        if (index >= 0 && index < numMaxNodes) {
            return m_nodesInfo[index];
        }
        // Return a default or placeholder FLNodeStruct
    }


};

Blockchain* Blockchain::instance = nullptr;



} // namespace ns3

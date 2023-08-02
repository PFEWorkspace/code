
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "../../rapidjson/document.h"
#include "ai-helper.h"

namespace ns3 {

class Blockchain : public Object {
public:
    static TypeId GetTypeId();

    void WriteTransaction(uint32_t nodeId);

    void PrintBlockchain() const;

private:
    //attributes
    std::string m_filename;
    int maxFLround ;
    int actualFLround;
    int modelToEval[numMaxNodes]; //we suppose that we only store modelIds
    int modelToAgreg[numMaxNodes];
   
    BCNodeStruct notBusyNodes[numMaxBCNodes];
    FLNodeStruct m_nodesInfo[numMaxNodes];
    // functions
    void SaveBlockchainToFile();
    void AddTransactionToBlockchain(const rapidjson::Value& transaction);
    uint32_t GetTimestamp() const;
   
    // Object's inherited methods
    virtual void DoDispose();
};

} // namespace ns3


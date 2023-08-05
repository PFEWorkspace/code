<<<<<<< HEAD

#include "ns3/log.h"
#include "Blockchain.h"
#include "ns3/log.h"
#include <chrono>
=======
#include "Blockchain.h"

>>>>>>> FL2

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("Blockchain");

<<<<<<< HEAD
=======
Blockchain* Blockchain::instance = nullptr;

>>>>>>> FL2
//TypeId Blockchain::GetTypeId() {
//    static TypeId tid = TypeId("Blockchain")
//        .SetParent<Object>()
//        .AddConstructor<Blockchain>()
//        .AddAttribute("Filename", "Blockchain filename", StringValue("blockchain.json"),
//            MakeStringAccessor(&Blockchain::m_filename),
//            MakeStringChecker())
//        ;
//    return tid;
//}

<<<<<<< HEAD
Blockchain& Blockchain::getInstance()  {
    // Create the instance if it doesn't exist
        // (This is lazy initialization)
        if (!instance) {
            instance = new Blockchain();
        }
        return *instance;
}

Ipv4Adresse Blockchain::getFLAdress(int nodeId)
{
    return nodeFLAdrs.GetAdress(nodeId);
    }

//Ipv4Adresse Blockchain::getBCAdress()
//{
//    // return a random adress from blockchain nodes
//    }
void Blockchain::WriteTransaction(uint32_t nodeId) {
    rapidjson::Document transaction;
    transaction.SetObject();
    transaction.AddMember("node_id", nodeId, transaction.GetAllocator());
    transaction.AddMember("timestamp", GetTimestamp(), transaction.GetAllocator());
=======
Ipv4Address Blockchain::getFLAddress(int nodeId)
{
    return nodesFLAdrs.GetAddress(nodeId);
}


Ipv4Address Blockchain::getBCAddress(){
    uint32_t randomid = randomBCAdrsStream->GetInteger();
    return nodesBCAdrs.GetAddress(randomid);
}

void Blockchain::SetFLAddressContainer(Ipv4InterfaceContainer container){
    nodesFLAdrs = container ;
}

void Blockchain::SetBCAddressContainer(Ipv4InterfaceContainer container){
    nodesBCAdrs = container;
}

void Blockchain::SetRandomBCStream(){
        randomBCAdrsStream = CreateObject<UniformRandomVariable>();
        randomBCAdrsStream->SetAttribute("Min", DoubleValue(0)); // bcs the addresses and ids of bc nodes start after the FL so min is number of flnodes
        randomBCAdrsStream->SetAttribute("Max", DoubleValue(numBCNodes-1)); // and max is the sum of both group of nodes size
}

void Blockchain::WriteTransaction(uint32_t nodeId) {
    rapidjson::Document transaction;
    rapidjson::Value value;
    transaction.SetObject();
    value = nodeId ;
    transaction.AddMember("node_id",value, transaction.GetAllocator());
    std::string time= GetTimestamp();
    const char* t = time.c_str() ;
    value.SetString(t, time.size());
    transaction.AddMember("timestamp", value, transaction.GetAllocator());
>>>>>>> FL2

    AddTransactionToBlockchain(transaction);
    SaveBlockchainToFile();
}

void Blockchain::PrintBlockchain() const {
<<<<<<< HEAD
    for (const auto& block : m_blocks) {
        NS_LOG_INFO("Block #" << block["block_id"].GetUint() << ":");
        for (const auto& transaction : block["transactions"].GetArray()) {
            NS_LOG_INFO("  Transaction #" << transaction["transaction_id"].GetUint()
                          << ", Node ID: " << transaction["node_id"].GetUint()
                          << ", Timestamp: " << transaction["timestamp"].GetUint());
        }
    }
=======
    // for (const auto& block : m_blocks) {
    //     NS_LOG_INFO("Block #" << block["block_id"].GetUint() << ":");
    //     for (const auto& transaction : block["transactions"].GetArray()) {
    //         NS_LOG_INFO("  Transaction #" << transaction["transaction_id"].GetUint()
    //                       << ", Node ID: " << transaction["node_id"].GetUint()
    //                       << ", Timestamp: " << transaction["timestamp"].GetUint());
    //     }
    // }
>>>>>>> FL2
}



void Blockchain::SaveBlockchainToFile() {
    // Implementation to save the blockchain to the file
}

void Blockchain::AddTransactionToBlockchain(const rapidjson::Value &transaction) {
    // Implementation to add a new transaction to the blockchain
}

<<<<<<< HEAD
string Blockchain::GetTimestamp() const {
     auto r=std::chrono::system_clock::now(); //Contains data about current time
    std::string s = std::format("{:%Y%m%d%H%M}", r);
    return s;
=======
std::string Blockchain::GetTimestamp() const {
    std::time_t currentTime = std::time(nullptr);
    // Convert the time_t object to a string
    std::stringstream ss;
    ss << std::ctime(&currentTime); // Convert time_t to char*
    std::string timeString = ss.str();
    return timeString;
>>>>>>> FL2
}

void Blockchain::DoDispose() {
    NS_LOG_FUNCTION_NOARGS();
    // Clean up resources
<<<<<<< HEAD
    Object::DoDispose();
}

} // namespace ns3
=======
   
}

} // namespace ns3
>>>>>>> FL2

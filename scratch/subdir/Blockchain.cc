#include "Blockchain.h"


namespace ns3 {

NS_LOG_COMPONENT_DEFINE("Blockchain");

Blockchain* Blockchain::instance = nullptr;

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

Ipv4Address Blockchain::getFLAddress(int nodeId)
{
    return nodesFLAdrs.GetAddress(nodeId);
}


Ipv4Address Blockchain::getBCAddress(){
    //TODO return random;
    return nodesBCAdrs.GetAddress(0);
}

void Blockchain::SetFLAddressContainer(Ipv4InterfaceContainer container){
    nodesFLAdrs = container ;
}

void Blockchain::SetBCAddressContainer(Ipv4InterfaceContainer container){
    nodesBCAdrs = container;
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

    AddTransactionToBlockchain(transaction);
    SaveBlockchainToFile();
}

void Blockchain::PrintBlockchain() const {
    // for (const auto& block : m_blocks) {
    //     NS_LOG_INFO("Block #" << block["block_id"].GetUint() << ":");
    //     for (const auto& transaction : block["transactions"].GetArray()) {
    //         NS_LOG_INFO("  Transaction #" << transaction["transaction_id"].GetUint()
    //                       << ", Node ID: " << transaction["node_id"].GetUint()
    //                       << ", Timestamp: " << transaction["timestamp"].GetUint());
    //     }
    // }
}



void Blockchain::SaveBlockchainToFile() {
    // Implementation to save the blockchain to the file
}

void Blockchain::AddTransactionToBlockchain(const rapidjson::Value &transaction) {
    // Implementation to add a new transaction to the blockchain
}

std::string Blockchain::GetTimestamp() const {
    std::time_t currentTime = std::time(nullptr);
    // Convert the time_t object to a string
    std::stringstream ss;
    ss << std::ctime(&currentTime); // Convert time_t to char*
    std::string timeString = ss.str();
    return timeString;
}

void Blockchain::DoDispose() {
    NS_LOG_FUNCTION_NOARGS();
    // Clean up resources
   
}

} // namespace ns3
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

void 
Blockchain::WriteTransaction(int blockId, int nodeId, const rapidjson::Document& message) {
    NS_LOG_INFO("wrinting transaction");
    std::lock_guard<std::mutex> lock(mtx);
    std::ifstream inFile("blockchain.json");
    std::stringstream jsonStream;
    jsonStream << inFile.rdbuf();
    std::string jsonString = jsonStream.str();

    // Parse the existing JSON
    rapidjson::Document blockchain;
    blockchain.Parse(jsonString.c_str());
   
    rapidjson::Value transaction;
    rapidjson::Document::AllocatorType& allocator = blockchain.GetAllocator();
    transaction.SetObject();

    // Add node_id
    transaction.AddMember("NodeId", nodeId, allocator);

    // Add timestamp
    std::string time = GetTimestamp();
    transaction.AddMember("Timestamp", rapidjson::Value(time.c_str(), allocator), allocator);

    // Clone the content from input message
    rapidjson::Value contentCopy;
    contentCopy.CopyFrom(message, allocator);
    transaction.AddMember("Content", contentCopy, allocator);

    // Search for the target block based on blockId
    if (blockchain.IsArray()) {
        for (rapidjson::Value::ValueIterator itr = blockchain.Begin(); itr != blockchain.End(); ++itr) {
            rapidjson::Value& block = *itr;
            if (block.HasMember("BlockId") && block["BlockId"].GetInt() == blockId) {
                if (!block.HasMember("Transactions")) {
                    rapidjson::Value transactions(rapidjson::kArrayType);
                    block.AddMember("Transactions", transactions, blockchain.GetAllocator());
                }

                // Add the transaction to the block's transactions array
                rapidjson::Value& transactionsArray = block["Transactions"];
                transactionsArray.PushBack(transaction, allocator);
                 
                // Print the updated block JSON
                rapidjson::StringBuffer updatedBuffer;
                rapidjson::Writer<rapidjson::StringBuffer> updatedWriter(updatedBuffer);
                blockchain.Accept(updatedWriter);
                // Save the updated JSON to the file
                std::ofstream outFile("blockchain.json");
                outFile << updatedBuffer.GetString();
                outFile.close();
                break; // No need to continue searching
            }else{
               // if block not found
                    rapidjson::Value newBlock(rapidjson::kObjectType);
                    newBlock.AddMember("BlockId", blockId, blockchain.GetAllocator());
                    rapidjson::Value transactions(rapidjson::kArrayType);
                    newBlock.AddMember("Transactions", transactions, blockchain.GetAllocator());
                    blockchain.PushBack(newBlock, blockchain.GetAllocator());                 

                   
                    // Add the transaction to the block's transactions array
                    rapidjson::Value& transactionsArray = newBlock["Transactions"];
                    transactionsArray.PushBack(transaction, allocator);

                    // Print the updated block JSON
                    rapidjson::StringBuffer updatedBuffer;
                    rapidjson::Writer<rapidjson::StringBuffer> updatedWriter(updatedBuffer);
                    blockchain.Accept(updatedWriter);

                    // Save the updated JSON to the file
                    std::ofstream outFile("blockchain.json");
                    outFile << updatedBuffer.GetString();
                    outFile.close();
        
            }

        }
    }

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

// void Blockchain::AddTransactionToBlockchain(const rapidjson::Value &transaction) {
//     // Implementation to add a new transaction to the blockchain
// }


std::string Blockchain::GetTimestamp(){
    // std::time_t currentTime = std::time(nullptr);
    // // Convert the time_t object to a string
    // std::stringstream ss;
    // ss << std::ctime(&currentTime); // Convert time_t to char*
    // std::string timeString = ss.str();
    // return timeString;

    std::time_t currentTime = std::time(nullptr);
    // Convert the time_t object to a string
    std::stringstream ss;
    ss << std::ctime(&currentTime); // Convert time_t to char*
    std::string timeString = ss.str();

    // Remove the trailing newline character
    if (!timeString.empty() && timeString.back() == '\n') {
        timeString.pop_back();
    }

    return timeString;
}

void Blockchain::DoDispose() {
    NS_LOG_FUNCTION_NOARGS();
    // Clean up resources
   
}

} // namespace ns3
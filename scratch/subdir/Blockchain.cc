#include "Blockchain.h"

#include "FL-node.h"

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
bool
Blockchain::ResetRound(MLModel globalModel){
    //update round number
    actualFLround++;
    // NS_LOG_INFO("actual FL round " <<actualFLround<<" maxFLRound "<<maxFLround);
    
    if(globalModel.accuracy >= GetTargetAcc()){
        // NS_LOG_INFO("inside accuracy condition "<< globalModel.accuracy << "  " << GetTargetAcc());
        return true;
    }else if(actualFLround < maxFLround ){
        receivedCandidatures = 0 ;
        modelToAgreg.clear();
        m_nodesInfo.clear();
        tasks.clear();
        firstagg = true;
        lastagg = false;
        interagg = 0 ;
        for (int i=0; i<numMaxAggregators;i++){
            aggregators[i]=-1;
        }
        for(int i=0; i<numMaxTrainers;i++){
            trainers[i]=-1;
        }
        SetCurrentBlockId();
        return false;
    }else return true;
   
}
bool
Blockchain::evaluate(){
    if(interagg < numTrainers/(modelsToAggAtOnce*2)){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0); // Range [0.0, 1.0)

        // Generate a random number between 0.0 and 1.0
        double randomValue = dis(gen);

        if (randomValue < 0.3) { // Adjust this threshold as needed
            interagg++;
            return true;
        } else {
           return false;
        }

    }else return false;
}


Ipv4Address Blockchain::getFLAddress(int nodeId)
{
    return nodesFLAdrs.GetAddress(nodeId);
}

int
Blockchain::getFLNodeId(Ipv4Address adrs){
    int id = -1 ;
    for(uint32_t i=0; i < nodesFLAdrs.GetN(); i++){
        if(nodesFLAdrs.GetAddress(i)==adrs){
            id=i;
            break;
        }
    }
    return id;
}

void 
Blockchain::AddModelToAgg( const MLModel& value) {
        NS_LOG_FUNCTION_NOARGS();
        // NS_LOG_INFO("add model to agg: "<<value.modelId);
        modelToAgreg.push_back(value);
    }


bool 
Blockchain::hasPreviousTask(int id){
    for(AggregatorsTasks t :tasks){
        if(t.nodeId==id)return true;
    }
    return false;
}
void
Blockchain::RemoveTask(int id){
    std::lock_guard<std::mutex> lock(mtx);
    // NS_LOG_INFO("1.tasks.size "<<tasks.size());
   for (std::list<AggregatorsTasks>::iterator it = tasks.begin(); it != tasks.end(); ++it) {
    if( it->nodeId == id){
        tasks.erase(it);
        break;
    }
    }
    // NS_LOG_INFO("2.tasks.size "<<tasks.size());
}

void 
Blockchain::AddTask(AggregatorsTasks task){
    std::lock_guard<std::mutex> lock(mtx);
    tasks.push_back(task);
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

double
Blockchain:: GetTrainingDelay(int nodeId){
    for(FLNodeStruct n : m_nodesInfo){
        if(n.nodeId==nodeId){
            return (n.datasetSize / n.freq) + (modelSize / n.transRate) ;
        }
    }
}

double
Blockchain::GetMaxTrainingDelay(){
    // double maxdelay = 0;
    // for( int i=0; i<numTrainers; i++){
    //     double cost = GetTrainingDelay(trainers[i]);
    //     if(maxdelay<cost) maxdelay=cost;
    // }
    // NS_LOG_INFO("max training delay "<< maxdelay);
    // return maxdelay;

    // the max is the max of the training cost plus 3 times the max of evaluation cost plus 4 times the max of communication cost
    // (1+3*0.2)*1000/50 + 4*1600/150
    return 75 + 10;
}

bool
Blockchain::MaxDelayPassed(){
    if(GetStillDelay()<= 0 ){
        return true;
    }else return false;
}
    
double
Blockchain::GetStillDelay(){
    Time now = Simulator::Now();
    Time passed = now - startTrainingTime;
    // NS_LOG_INFO("passed time "<< passed.GetSeconds());
    return GetMaxTrainingDelay() - passed.GetSeconds();
}

int Blockchain::getNumAggTasksAwaiting(){
    int count = 0;
    for(AggregatorsTasks task : tasks){
        if(task.task == AGGREGATE || (task.task == EVALUATE && task.models[0].type==0)){count++;}
    }
    return count;
}

void Blockchain::WriteTransaction(std::string blockId, int nodeId, const rapidjson::Document& message) {
   
    // NS_LOG_INFO("START WRITE TRANSACTION");
    std::lock_guard<std::mutex> lock(mtx);
    // Load the existing JSON from the file or create if not exists
    rapidjson::Document blockchain;
    std::ifstream inFile("blockchain.json");
    if (inFile.good()) {
        // NS_LOG_INFO("Loading existing JSON from the file");
        std::stringstream jsonStream;
        jsonStream << inFile.rdbuf();
        std::string jsonString = jsonStream.str();
        blockchain.Parse(jsonString.c_str());
    } else {
        // NS_LOG_INFO("File doesn't exist, creating initial structure");
        blockchain.SetArray();
    }
    
    rapidjson::Document::AllocatorType& allocator = blockchain.GetAllocator();

    rapidjson::Value transaction;
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

    // NS_LOG_INFO(time);

    bool blockFound = false;

    // Search for the target block based on blockId
    if (blockchain.IsArray()) {
        for (rapidjson::Value::ValueIterator itr = blockchain.Begin(); itr != blockchain.End(); ++itr) {
            rapidjson::Value& block = *itr;
            // NS_LOG_INFO("Iterating on the blocks to find the one matching with blockid");
            if (block.HasMember("BlockId") && block["BlockId"].GetString() == blockId) {
                // NS_LOG_INFO("Found target block");
                if (!block.HasMember("Transactions")) {
                    // NS_LOG_INFO("Block doesn't have any transactions");
                    rapidjson::Value transactions(rapidjson::kArrayType);
                    block.AddMember("Transactions", transactions, allocator);
                }

                // Add the transaction to the block's transactions array
                rapidjson::Value& transactionsArray = block["Transactions"];
                transactionsArray.PushBack(transaction, allocator);

                blockFound = true;

                break; // No need to continue searching
            }
        }
    }

    if (!blockFound) {
        // NS_LOG_INFO("Block not found, creating a new block");

        // Create a new block
        rapidjson::Value newBlock(rapidjson::kObjectType);
        rapidjson::Value value;
        value.SetString(blockId.c_str(),blockId.size(),allocator);
        newBlock.AddMember("BlockId", value, allocator);
        rapidjson::Value transactions(rapidjson::kArrayType);
        newBlock.AddMember("Transactions", transactions, allocator);

        // Add the transaction to the new block's transactions array
        rapidjson::Value& transactionsArray = newBlock["Transactions"];
        transactionsArray.PushBack(transaction, allocator);

        // Push the new block to the blockchain array
        blockchain.PushBack(newBlock, allocator);
    }

    // Print the updated block JSON
    rapidjson::StringBuffer updatedBuffer;
    rapidjson::Writer<rapidjson::StringBuffer> updatedWriter(updatedBuffer);
    blockchain.Accept(updatedWriter);

    // Save the updated JSON to the file
    std::ofstream outFile("blockchain.json");
    outFile << updatedBuffer.GetString();
    outFile.close();
    // NS_LOG_INFO("END WRITE TRANSACTION, file should be closed");
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
int
Blockchain::GetAggregatorNotBusy(int eval1, int eval2){
    int id=-1;
    bool busy;
    for(int agg : aggregators){
        busy= false;       
        for(AggregatorsTasks task : tasks){
            if(agg == task.nodeId){
                busy = true;
                // NS_LOG_INFO("agg "<< agg << " busy");
                break;
            }
        }
        if(!busy && agg!=eval1 && agg!=eval2){
            id = agg ;
            // NS_LOG_INFO("agg "<< agg << " not busy");
            break;
        } 
    }
    return id; // -1 for no available 

}

void Blockchain::DoDispose() {
    NS_LOG_FUNCTION_NOARGS();
    // Clean up resources
   
}

} // namespace ns3
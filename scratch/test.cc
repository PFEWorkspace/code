#include "../rapidjson/document.h"
#include "../rapidjson/filereadstream.h"
#include "../rapidjson/filewritestream.h"
#include "../rapidjson/ostreamwrapper.h"  // Include OStreamWrapper
#include "../rapidjson/writer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <string>


std::string GetTimestamp() {
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

void WriteTransaction(rapidjson::Document& blockchain, int blockId, int nodeId, const rapidjson::Document& message) {
    rapidjson::Value transaction;
    rapidjson::Document::AllocatorType& allocator = blockchain.GetAllocator();
    transaction.SetObject();
    transaction.AddMember("NodeId", nodeId, allocator);
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

                break; // No need to continue searching
            }
            {
            rapidjson::Value& block = *itr;
rapidjson::Value transactions(rapidjson::kArrayType);
 transaction.SetObject();
    transaction.AddMember("NodeId", nodeId, allocator);
    std::string time = GetTimestamp();
    transaction.AddMember("Timestamp", rapidjson::Value(time.c_str(), allocator), allocator);

    // Clone the content from input message
    rapidjson::Value contentCopy;
    contentCopy.CopyFrom(message, allocator);
    transaction.AddMember("Content", contentCopy, allocator);
                    block.AddMember("Transactions", transactions, blockchain.GetAllocator());
                      rapidjson::Value& transactionsArray = block["Transactions"];
                transactionsArray.PushBack(transaction, allocator);
                
            }

        }
    }
}

// int main() {
//     // Load the existing JSON from the file
//     std::ifstream inFile("blockchain.json");
//     std::stringstream jsonStream;
//     jsonStream << inFile.rdbuf();
//     std::string jsonString = jsonStream.str();

//     // Parse the existing JSON
//     rapidjson::Document blockchain;
//     blockchain.Parse(jsonString.c_str());

//     // Create a sample JSON message
//     rapidjson::Document message;
//     message.SetObject();
//     rapidjson::Value contentValue;
//     contentValue.SetString("TEST", message.GetAllocator());
//     message.AddMember("message", contentValue, message.GetAllocator());

//     // Test WriteTransaction function
//     int nodeId = 123;
//     int targetBlockId = 3; // Specify the target block ID here

//     // Check if the blockchain JSON is an array
//     if (blockchain.IsArray()) {
//         for (rapidjson::Value::ValueIterator itr = blockchain.Begin(); itr != blockchain.End(); ++itr) {
//             rapidjson::Value& block = *itr;
//             if (block.HasMember("BlockId") && block["BlockId"].GetInt() == targetBlockId) {
//                 if (!block.HasMember("Transactions")) {
//                     rapidjson::Value transactions(rapidjson::kArrayType);
//                     block.AddMember("Transactions", transactions, blockchain.GetAllocator());
//                 }

//                 WriteTransaction(blockchain, targetBlockId, nodeId, message);

//                 // Print the updated block JSON
//                 rapidjson::StringBuffer updatedBuffer;
//                 rapidjson::Writer<rapidjson::StringBuffer> updatedWriter(updatedBuffer);
//                 blockchain.Accept(updatedWriter);

//                 // Save the updated JSON to the file
//                 std::ofstream outFile("blockchain.json");
//                 outFile << updatedBuffer.GetString();
//                 outFile.close();

//                 break; // No need to continue searching
//             }
//         }
//     }

//     return 0;
// }

// ... (previous includes and functions)

int main() {
    // Load the existing JSON from the file
    std::ifstream inFile("blockchain.json");
    std::stringstream jsonStream;
    jsonStream << inFile.rdbuf();
    std::string jsonString = jsonStream.str();

    // Parse the existing JSON
    rapidjson::Document blockchain;
    blockchain.Parse(jsonString.c_str());

    // Create a sample JSON message
    rapidjson::Document message;
    message.SetObject();
    rapidjson::Value contentValue;
    contentValue.SetString("TEST", message.GetAllocator());
    message.AddMember("message", contentValue, message.GetAllocator());

    // Test WriteTransaction function
    int nodeId = 123;
    int targetBlockId = 3; // Specify the target block ID here

    // Check if the blockchain JSON is an array
    bool blockFound = false;
    if (blockchain.IsArray()) {
        for (rapidjson::Value::ValueIterator itr = blockchain.Begin(); itr != blockchain.End(); ++itr) {
            rapidjson::Value& block = *itr;
            if (block.HasMember("BlockId") && block["BlockId"].GetInt() == targetBlockId) {
                if (!block.HasMember("Transactions")) {
                    rapidjson::Value transactions(rapidjson::kArrayType);
                    block.AddMember("Transactions", transactions, blockchain.GetAllocator());
                }

                WriteTransaction(blockchain, targetBlockId, nodeId, message);

                // Print the updated block JSON
                rapidjson::StringBuffer updatedBuffer;
                rapidjson::Writer<rapidjson::StringBuffer> updatedWriter(updatedBuffer);
                blockchain.Accept(updatedWriter);

                // Save the updated JSON to the file
                std::ofstream outFile("blockchain.json");
                outFile << updatedBuffer.GetString();
                outFile.close();

                blockFound = true;
                break;
            }
        }

        // If the target block doesn't exist, create it
        if (!blockFound) {
            rapidjson::Value newBlock(rapidjson::kObjectType);
            newBlock.AddMember("BlockId", targetBlockId, blockchain.GetAllocator());
            rapidjson::Value transactions(rapidjson::kArrayType);
            newBlock.AddMember("Transactions", transactions, blockchain.GetAllocator());
            blockchain.PushBack(newBlock, blockchain.GetAllocator());

            WriteTransaction(blockchain, targetBlockId, nodeId, message);

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

    return 0;
}

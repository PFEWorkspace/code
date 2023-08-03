#include "ns3/applications-module.h"
#include"ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/address.h"
#include "ns3/address-utils.h"
#include "ns3/inet-socket-address.h"
#include "ai-helper.h"


#include <iostream>
#include <fstream>
#include <string>

#include "../../rapidjson/document.h"
#include "../../rapidjson/error/en.h"
#include "../../rapidjson/writer.h"
#include "../../rapidjson/stringbuffer.h"

namespace ns3{

enum CommunicationType {
  NEWTASK,
  CONDIDATURE,
  SELECTION,
  MODEL
  //add the rest later
};

class Initiator : public Application
{
    public:
     /**
     * \brief Get the type ID.
     * \return The object TypeId.
     */
    static TypeId GetTypeId();

    Initiator();
    ~Initiator() override;

    // Setters
    void setRounds(int rounds) {
        m_rounds = rounds;
    }

    void setTargetAcc(double targetAcc) {
        m_targetAcc = targetAcc;
    }

    void setNumNodes(int numNodes) {
        m_numNodes = numNodes;
    }

    void setNumParticipants(int numParticipants) {
        m_numParticipants = numParticipants;
    }

    void setNumAggregators(int numAggregators) {
        m_numAggregators = numAggregators;
    }

    void setNodesInfo(FLNodeStruct* nodesInfo, int numNodes);

    // Getters
    int getRounds() const {
        return m_rounds;
    }

    double getTargetAcc() const {
        return m_targetAcc;
    }

    int getNumNodes() const {
        return m_numNodes;
    }

    int getNumParticipants() const {
        return m_numParticipants;
    }

    int getNumAggregators() const {
        return m_numAggregators;
    }

    FLNodeStruct* getNodesInfo() {
        return m_nodesInfo;
    }


  protected:
    void DoDispose() override;

  private:
    void StartApplication() override;
    void StopApplication() override;
    
    /**
     * Send a packet.
     */
    void SendPacket(rapidjson::Document d, Address &outgoingAddr);
    


    Ipv4Address m_destAddr; //!< Destination address
    Ipv4Address m_srcAddr; //!< source address
    uint32_t m_destPort{8833}; //!< Destination port
    Ptr<Socket> m_socket; //!< Sending socket
    EventId m_sendEvent;  //!< Send packet event
    
  
    int m_rounds;
    double m_targetAcc;
    int m_numNodes;
    int m_numParticipants;
    int m_numAggregators;
    FLNodeStruct m_nodesInfo[numMaxNodes];


};
std::string ReadFileToString(const std::string& filePath);
bool ParseJSON(const std::string& jsonString, rapidjson::Document& document);


/**
 * Receiver application.
 */
class Receiver : public Application
{
  public:
    /**
     * \brief Get the type ID.
     * \return The object TypeId.
     */
    static TypeId GetTypeId();

    Receiver();
    ~Receiver() override;


  protected:
    void DoDispose() override;

  private:
    void StartApplication() override;
    void StopApplication() override;

    /**
     * Receive a packet.
     * \param socket The receiving socket.
     */
    void Receive(Ptr<Socket> socket);

    Ptr<Socket> m_socket; //!< Receiving socket
    uint32_t m_port{8833};   //!< Listening port
    

  
};
}
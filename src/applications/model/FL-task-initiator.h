#include "ns3/application.h"
#include"ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/address.h"
#include "ns3/address-utils.h"
#include "ns3/inet-socket-address.h"

#include <iostream>
#include <fstream>
#include <string>

#include "../../../rapidjson/document.h"
#include "../../../rapidjson/error/en.h"
#include "../../../rapidjson/writer.h"
#include "../../../rapidjson/stringbuffer.h"

namespace ns3{

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
    
    // data related to the FL task
    uint32_t m_budget; //!< le budget pour la tache FL
    std::string m_model{"MNIST"}; //!< the model name
    int m_rounds{0};
    double m_targetAccuracy;
    int m_epochs{5};
    int m_batchSize{10};

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
#ifndef BC_NODE_H
#define BC_NODE_H

#include "ns3/applications-module.h"
#include"ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/address.h"
#include "ns3/address-utils.h"
#include "ns3/inet-socket-address.h"

#include "FL-node.h"
#include "FL-task-initiator.h"

#include <vector>

namespace ns3{




class BCNode : public Application
{
  public:
    /**
     * \brief Get the type ID.
     * \return The object TypeId.
     */
    static TypeId GetTypeId();

    BCNode();
    ~BCNode() override;

    
  void SetPort(uint32_t port);
  uint32_t GetPort() const;

   void SetDestAddress(Ipv4Address address);
  Ipv4Address GetDestAddress() const;


  protected:
    void DoDispose() override;

  private:
    void StartApplication() override;
    void StopApplication() override;
    void Receive(Ptr<Socket> socket);
    void Send(rapidjson::Document& d);
    void SendTo( rapidjson::Document &d, std::vector<Ipv4Address> &addresses);
    void TreatCandidature(rapidjson::Document &d);
    
    void WriteTransaction();

    FLNodeStruct docToFLNodeStruct(rapidjson::Document &d);

    Ptr<Socket> m_socket; // Receiving socket
    uint32_t m_port{8833};   // Listening port
    Ipv4Address m_destAddr; // the destination address

};

}

#endif

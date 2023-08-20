#ifndef BC_NODE_H
#define BC_NODE_H

#include "ns3/applications-module.h"
#include"ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/address.h"
#include "ns3/address-utils.h"
#include "ns3/inet-socket-address.h"
#include "ai-helper.h"
#include "../../rapidjson/document.h"
#include "Blockchain.h"
// #include "../../rapidjson/error/en.h"
// #include "../../rapidjson/writer.h"
// #include "../../rapidjson/stringbuffer.h"

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
    
  static MLModel DocToMLModel(rapidjson::Document &d);
    
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
    void Send(rapidjson::Document& d, Ipv4Address adrs);
    void SendTo( rapidjson::Document &d, std::vector<Ipv4Address> &addresses);
    void SendBroadCast(rapidjson::Document& d, Ipv4Address adrs);
    void TreatCandidature(rapidjson::Document &d);
    
    void TreatModel(MLModel model, Ipv4Address source, bool reschedule);
    void Selection();
    void WriteTransaction();
    void Evaluation(MLModel model, int nodeId);
    void Aggregation(std::vector<MLModel> models, int nodeId, int type);
    void DetectDropOut(AggregatorsTasks task);
    void NewRound(MLModel globalModel);
    
    
    FLNodeStruct docToFLNodeStruct(rapidjson::Document &d);

    Ptr<Socket> m_socket; // Receiving socket
    uint32_t m_port{8833};   // Listening port
    Ipv4Address m_destAddr; // the destination address
    

};

}

#endif
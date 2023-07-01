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



class FLNode : public Application
{
  public:
    /**
     * \brief Get the type ID.
     * \return The object TypeId.
     */
    static TypeId GetTypeId();

    FLNode();
    ~FLNode() override;


  protected:
    void DoDispose() override;

  private:
    void StartApplication() override;
    void StopApplication() override;
    void Receive(Ptr<Socket> socket);
    void Send(rapidjson::Document d, Address &outgoingAddr);
    void Condidater(std::string FLTaskId);
    void Train();
    void SendModel();


    Ptr<Socket> m_socket; //!< Receiving socket
    uint32_t m_port{8833};   //!< Listening port

  
};
}
#include "ns3/application.h"
#include"ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/address.h"
#include "ns3/address-utils.h"
#include "ns3/inet-socket-address.h"

// #include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>

#include "../../../rapidjson/document.h"
#include "../../../rapidjson/error/en.h"
#include "../../../rapidjson/writer.h"
#include "../../../rapidjson/stringbuffer.h"

namespace ns3{

enum Task{
  TRAIN,
  AGREGATE,
  EVALUATE
};


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

    
  void SetPort(uint32_t port);
  uint32_t GetPort() const;

   void SetDestAddress(Ipv4Address address);
  Ipv4Address GetDestAddress() const;

  void SetDatasetSize(uint32_t size);
  uint32_t GetDatasetSize() const;

  void SetBeta(double beta);
  double GetBeta() const;

  void SetFrequency(double frequency);
  double GetFrequency() const;

  void SetTransmissionRate(double rate);
  double GetTransmissionRate() const;

  void SetAvailability(bool available);
  bool IsAvailable() const;

  void SetHonesty(double honesty);
  double GetHonesty() const;

  protected:
    void DoDispose() override;

  private:
    void StartApplication() override;
    void StopApplication() override;
    void Receive(Ptr<Socket> socket);
    void Send(rapidjson::Document &d);
    void Condidater();
    void Train();
    void SendModel();


    Ptr<Socket> m_socket; // Receiving socket
    uint32_t m_port{8833};   // Listening port
    Ipv4Address m_destAddr; // the destination address
    enum Task givenTask; // the task that the node will be doing : training, evaluating or agregation
    enum Task wantedTask; // the task the node wants to do, will be sent in the condidature
    // characteristics
    uint32_t dataset_size;
    double beta; // necessary CPU cycle to train one data unit
    double freq ; // the frequency of the CPU of the node
    double trans_rate; // transmission rate
    bool availability ; // true if node available to participate, else false
    double honesty; // the honesty score of the node
 
};

}
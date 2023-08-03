#include "ns3/applications-module.h"
#include"ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/address.h"
#include "ns3/address-utils.h"
#include "ns3/inet-socket-address.h"

#include "ai-helper.h"
#include "Blockchain.h"

#include <iostream>
#include <fstream>
#include <string>

// #include "../../rapidjson/document.h"
// #include "../../rapidjson/error/en.h"
// #include "../../rapidjson/writer.h"
// #include "../../rapidjson/stringbuffer.h"

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

  void SetTask(enum Task t);
  Task GetTask() const;

   void SetDestAddress(Ipv4Address address);
  Ipv4Address GetDestAddress() const;

  void SetDatasetSize(int size);
  int GetDatasetSize() const;


  void SetFrequency(double frequency);
  double GetFrequency() const;

  void SetTransmissionRate(double rate);
  double GetTransmissionRate() const;

  void SetAvailability(bool available);
  bool IsAvailable() const;

  void SetHonesty(double honesty);
  double GetHonesty() const;

  void Init(FLNodeStruct n);
  void ResetRound();
  protected:
    void DoDispose() override;

  private:
    void StartApplication() override;
    void StopApplication() override;
    void Receive(Ptr<Socket> socket);
    void Send(Ipv4Address adrs, rapidjson::Document &d);
    void Candidater();
    void Train(int globalModelId);
    void SendModel();


    Ptr<Socket> m_socket; // Receiving socket
    uint32_t m_port{8833};   // Listening port
    Ipv4Address m_destAddr; // the destination address
    int id; //node id
    enum Task task; // the task that the node will be doing : training, evaluating or agregation 
    int dataset_size;
    double freq ; // the frequency of the CPU of the node, btw 50 and 300 MHz
    double trans_rate; // transmission rate, with wifi the values are btw 150 Mbps and 1 Gbps : https://www.techtarget.com/iotagenda/feature/Everything-you-need-to-know-about-IoT-connectivity-options
    bool availability ; // true if node available to participate, else false
    double honesty; // the honesty score of the node
    double dropout ; // if he's going to dropout or not 
    double malicious ; // if he's going to alter his results or not
 
};

}
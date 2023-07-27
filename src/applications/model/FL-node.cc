#include  "FL-node.h"
#include "FL-task-initiator.h"
namespace ns3{
    
NS_LOG_COMPONENT_DEFINE("FLNodeApp");

TypeId FLNode::GetTypeId() {
  static TypeId tid = TypeId("FLNode")
                          .SetParent<Application>()
                          .SetGroupName("Applications")
                          .AddConstructor<FLNode>()
                          .AddAttribute("Port", "Listening port", UintegerValue(8833),
                                        MakeUintegerAccessor(&FLNode::m_port),
                                        MakeUintegerChecker<uint32_t>())
                          .AddAttribute("Destination",
                                          "Target host address.",
                                          Ipv4AddressValue(),
                                          MakeIpv4AddressAccessor(&FLNode::m_destAddr),
                                          MakeIpv4AddressChecker())
                          .AddAttribute("DatasetSize", "Size of the dataset", UintegerValue(0),
                                        MakeUintegerAccessor(&FLNode::dataset_size),
                                        MakeUintegerChecker<uint32_t>())
                          .AddAttribute("Frequency", "Frequency of the CPU", DoubleValue(0.0),
                                        MakeDoubleAccessor(&FLNode::freq),
                                        MakeDoubleChecker<double>())
                          .AddAttribute("TransmissionRate", "Transmission rate", DoubleValue(0.0),
                                        MakeDoubleAccessor(&FLNode::trans_rate),
                                        MakeDoubleChecker<double>())
                          .AddAttribute("Availability", "Availability of the node", BooleanValue(true),
                                        MakeBooleanAccessor(&FLNode::availability),
                                        MakeBooleanChecker())
                          .AddAttribute("Honesty", "Honesty score of the node", DoubleValue(0.0),
                                        MakeDoubleAccessor(&FLNode::honesty),
                                        MakeDoubleChecker<double>());
  return tid;
}

FLNode::FLNode() {
   NS_LOG_FUNCTION_NOARGS();
}

FLNode::~FLNode() {
  NS_LOG_FUNCTION_NOARGS();
}

void FLNode::SetPort(uint32_t port) {
  m_port = port;
}

uint32_t FLNode::GetPort() const {
  return m_port;
}

void FLNode::SetDestAddress(Ipv4Address address) {
  m_destAddr = address;
}

Ipv4Address FLNode::GetDestAddress() const {
  return m_destAddr;
}

void FLNode::SetDatasetSize(uint32_t size) {
  dataset_size = size;
}

uint32_t FLNode::GetDatasetSize() const {
  return dataset_size;
}

// void FLNode::SetBeta(double beta) {
//   this->beta = beta;
// }

// double FLNode::GetBeta() const {
//   return beta;
// }

void FLNode::SetFrequency(double frequency) {
  freq = frequency;
}

double FLNode::GetFrequency() const {
  return freq;
}

void FLNode::SetTransmissionRate(double rate) {
  trans_rate = rate;
}

double FLNode::GetTransmissionRate() const {
  return trans_rate;
}

void FLNode::SetAvailability(bool available) {
  availability = available;
}

bool FLNode::IsAvailable() const {
  return availability;
}

void FLNode::SetHonesty(double honesty) {
  this->honesty = honesty;
}

double FLNode::GetHonesty() const {
  return honesty;
}

void FLNode::DoDispose() {
   NS_LOG_FUNCTION_NOARGS();
    m_socket = nullptr;
    // chain up
    Application::DoDispose();
}

void FLNode::StartApplication() {
  NS_LOG_FUNCTION_NOARGS();

    if (!m_socket)
    {
     
        Ptr<SocketFactory> socketFactory =
            GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
        m_socket = socketFactory->CreateSocket();
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
        m_socket->Bind(local);
    }

    m_socket->SetRecvCallback(MakeCallback(&FLNode::Receive, this));

}
void FLNode::StopApplication() {
  NS_LOG_FUNCTION_NOARGS();
}

void FLNode::Receive(Ptr<Socket> socket) {
    

    Ptr<Packet> packet;
    Address from;
  
    while ((packet = socket->RecvFrom(from)))
    {
        char *packetInfo = new char[packet->GetSize () + 1];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            packet->CopyData (reinterpret_cast<uint8_t*>(packetInfo), packet->GetSize ());
            // NS_LOG_INFO("I'm "<< GetNode()->GetId() << "received " << packet->GetSize() << " bytes from "
            //                         << InetSocketAddress::ConvertFrom(from).GetIpv4()
            //                         << " content: "<< packetInfo) ;
            std::string data = packetInfo ; 
            rapidjson::Document d;
           
            if(ParseJSON(data,d)){
                if(d.HasMember("message_type") && d["message_type"].IsInt()){
                    switch (d["message_type"].GetInt())
                    {
                    case NEWTASK: 
                        /* 
                            newtask is the message sent by the initializer to declare a new task
                            as a response the FL nodes will send their condidature to the blockchain
                         */
                        Condidater();
                        break;
                    
                    default:
                        break;
                    }
                }
            }
        }

    }
}

void FLNode::Send(rapidjson::Document &d) {
 
 if (!m_socket){
    Ptr<SocketFactory> socketFactory = GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
    m_socket = socketFactory->CreateSocket();
    m_socket->Bind();
 }
    rapidjson::StringBuffer packetInfo;
    rapidjson::Writer<rapidjson::StringBuffer> writer(packetInfo);
    d.Accept(writer);

    Ptr<Packet> packet = Create<Packet>(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize());
    m_socket->SendTo(packet,0,InetSocketAddress(m_destAddr, m_port));
}

void FLNode::Condidater() {
 rapidjson::Document d;
 rapidjson::Value value;
 d.SetObject();
 enum CommunicationType msg = CONDIDATURE;
 value = msg;
 d.AddMember("message_type", value, d.GetAllocator());
//  value = wantedTask ;
//  d.AddMember("task", value, d.GetAllocator());
 value = dataset_size ; 
 d.AddMember("data_size", value, d.GetAllocator());
//  value = beta ;
//  d.AddMember("beta", value, d.GetAllocator());
  value = freq ;
 d.AddMember("frequence", value, d.GetAllocator());
  value = trans_rate ;
 d.AddMember("transmission_rate", value, d.GetAllocator());
  value = availability ;
 d.AddMember("availability", value, d.GetAllocator());
  value = honesty ;
 d.AddMember("honesty", value, d.GetAllocator());

 Send(d);
}

void FLNode::Train() {
  // Train implementation
}

void FLNode::SendModel() {
  // SendModel implementation
}


}
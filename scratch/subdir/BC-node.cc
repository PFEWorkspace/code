#include  "FL-node.h"
#include "FL-task-initiator.h"
#include "BC-node.h"
namespace ns3{
    
NS_LOG_COMPONENT_DEFINE("BCNodeApp");

TypeId BCNode::GetTypeId() {
  static TypeId tid = TypeId("BCNode")
                          .SetParent<Application>()
                          .SetGroupName("Applications")
                          .AddConstructor<BCNode>()
                          .AddAttribute("Port", "Listening port", UintegerValue(8833),
                                        MakeUintegerAccessor(&BCNode::m_port),
                                        MakeUintegerChecker<uint32_t>())
                          .AddAttribute("Destination",
                                          "Target host address.",
                                          Ipv4AddressValue(),
                                          MakeIpv4AddressAccessor(&BCNode::m_destAddr),
                                          MakeIpv4AddressChecker())
                        .AddAttribute("DatasetSize", "Size of the dataset", UintegerValue(0),
                                        MakeUintegerAccessor(&BCNode::dataset_size),
                                        MakeUintegerChecker<uint32_t>())
                          .AddAttribute("Frequency", "Frequency of the CPU", DoubleValue(0.0),
                                        MakeDoubleAccessor(&BCNode::freq),
                                        MakeDoubleChecker<double>())
                          .AddAttribute("TransmissionRate", "Transmission rate", DoubleValue(0.0),
                                        MakeDoubleAccessor(&BCNode::trans_rate),
                                        MakeDoubleChecker<double>())
                          .AddAttribute("Availability", "Availability of the node", BooleanValue(true),
                                        MakeBooleanAccessor(&BCNode::availability),
                                        MakeBooleanChecker())
                          .AddAttribute("Honesty", "Honesty score of the node", DoubleValue(0.0),
                                        MakeDoubleAccessor(&BCNode::honesty),
                                        MakeDoubleChecker<double>())
                            .AddAttribute("Blockchain","Blockchain attached to node", Blockchain());
  return tid;
}

BCNode::BCNode() {
   NS_LOG_FUNCTION_NOARGS();
}

BCNode::~BCNode() {
  NS_LOG_FUNCTION_NOARGS();
}

void BCNode::SetPort(uint32_t port) {
  m_port = port;
}

uint32_t BCNode::GetPort() const {
  return m_port;
}

void BCNode::SetDestAddress(Ipv4Address address) {
  m_destAddr = address;
}

Ipv4Address BCNode::GetDestAddress() const {
  return m_destAddr;
}


// void BCNode::SetBeta(double beta) {
//   this->beta = beta;
// }

// double BCNode::GetBeta() const {
//   return beta;
// }

void BCNode::SetFrequency(double frequency) {
  freq = frequency;
}

double BCNode::GetFrequency() const {
  return freq;
}

void BCNode::SetTransmissionRate(double rate) {
  trans_rate = rate;
}

double BCNode::GetTransmissionRate() const {
  return trans_rate;
}

void BCNode::SetAvailability(bool available) {
  availability = available;
}

bool BCNode::IsAvailable() const {
  return availability;
}

void BCNode::SetHonesty(double honesty) {
  this->honesty = honesty;
}

double BCNode::GetHonesty() const {
  return honesty;
}

void BCNode::DoDispose() {
   NS_LOG_FUNCTION_NOARGS();
    m_socket = nullptr;
    // chain up
    Application::DoDispose();
}

void BCNode::StartApplication() {
  NS_LOG_FUNCTION_NOARGS();

    if (!m_socket)
    {
     
        Ptr<SocketFactory> socketFactory =
            GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
        m_socket = socketFactory->CreateSocket();
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
        m_socket->Bind(local);
    }

    m_socket->SetRecvCallback(MakeCallback(&BCNode::Receive, this));

}
void BCNode::StopApplication() {
  NS_LOG_FUNCTION_NOARGS();
}

void BCNode::Receive(Ptr<Socket> socket) {
    
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
                    case NEWTASK : //NEWTASK 
                        /* 
                            newtask is the message sent by the initializer to declare a new task
                            as a response the FL nodes will send their condidature to the blockchain
                         task_id,model_id,rounds,target_acc,num_participants,num_aggregators}
                         */
                      
                      // TODO : mise a jour de la blokchain with the new task , 
                      break;
                    case CANDIDATURE : 
                     // receive les candidatures and treat them

                     // teste ida got all les candidatures ou time passed ou
                      
                      AiHelper aiH = AiHelper () ; 
                      aiH.ExactSelection()

                      //print blockchainaggregators 
                      // step get the addresses from blockchain list of node 
                      // send the messages MODEL to trainers {msgtype:selection, worktype: aggregatio ou learning, modelId}

                        break;
                    case MODEL : //MODEL
                    break ;
                    case EVALUATE : // EVALUATE
                    break; 
                    
                    default:
                        break;
                    }
                }
            }
        }

    }
}

void BCNode::Send(rapidjson::Document &d) {
 
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


void BCNode::SendTo( rapidjson::Document &d, std::vector<Ipv4Address> &addresses) {
    if (!m_socket) {
        Ptr<SocketFactory> socketFactory = GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
        m_socket = socketFactory->CreateSocket();
        m_socket->Bind();
    }

    rapidjson::StringBuffer packetInfo;
    rapidjson::Writer<rapidjson::StringBuffer> writer(packetInfo);
    d.Accept(writer);

    Ptr<Packet> packet = Create<Packet>(reinterpret_cast<const uint8_t*>(packetInfo.GetString()), packetInfo.GetSize());

    for (const auto& destAddr : addresses) {
        m_socket->SendTo(packet->Copy(), 0, InetSocketAddress(destAddr, m_port));
    }
}


// void BCNode::Condidater() {
//  rapidjson::Document d;
//  rapidjson::Value value;
//  d.SetObject();
//  enum CommunicationType msg = CONDIDATURE;
//  value = msg;
//  d.AddMember("message_type", value, d.GetAllocator());
// // //  value = wantedTask ;
// // //  d.AddMember("task", value, d.GetAllocator());
// //  value = dataset_size ; 
// //  d.AddMember("data_size", value, d.GetAllocator());
// // //  value = beta ;
// // //  d.AddMember("beta", value, d.GetAllocator());
// //   value = freq ;
// //  d.AddMember("frequence", value, d.GetAllocator());
// //   value = trans_rate ;
// //  d.AddMember("transmission_rate", value, d.GetAllocator());
// //   value = availability ;
// //  d.AddMember("availability", value, d.GetAllocator());
// //   value = honesty ;
// //  d.AddMember("honesty", value, d.GetAllocator());

//  Send(d);
// }


// void BCNode::SendModel() {
//   // SendModel implementation
// }

void BCNode :: WriteTransaction (){

}

}
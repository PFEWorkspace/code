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

void FLNode::SetTask(enum Task t){
  task = t ;
}
Task FLNode::GetTask() const{
  return task ;
}

void FLNode::SetDestAddress(Ipv4Address address) {
  m_destAddr = address;
}

Ipv4Address FLNode::GetDestAddress() const {
  return m_destAddr;
}

void FLNode::SetDatasetSize(int size) {
  dataset_size = size;
}

int FLNode::GetDatasetSize() const {
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

void FLNode::Init(FLNodeStruct n){
  id = n.nodeId ;
  availability = n.availability ;
  honesty = n.honesty ;
  dataset_size = n.datasetSize;
  freq = n.freq;
  trans_rate = n.transRate ;
  task = Task(n.task);
  dropout = n.dropout ;
  // malicious = n.malicious;
}

void FLNode::ResetRound(){

}

void FLNode::DoDispose() {
   NS_LOG_FUNCTION_NOARGS();
    m_socket = nullptr;
    // chain up
    Application::DoDispose();
}

void FLNode::StartApplication() {
  NS_LOG_FUNCTION_NOARGS();
  // NS_LOG_DEBUG("starting app");

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
        unsigned char *packetInfo = new unsigned char[packet->GetSize()];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            packet->CopyData(packetInfo, packet->GetSize () );
            NS_LOG_DEBUG("i'am "<< GetNode()->GetId() << " and i received a packet");
            // NS_LOG_DEBUG("I'm "<< GetNode()->GetId() << "received " << packet->GetSize() << " bytes from "
            //                         << InetSocketAddress::ConvertFrom(from).GetIpv4()
            //                         << " content: "<< packetInfo) ;

            std::string data(reinterpret_cast<char*>(packetInfo), packet->GetSize()) ; 
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
                        // Candidater(InetSocketAddress::ConvertFrom(from).GetIpv4());
                       Candidater();
                        break;
                    // case SELECTION :
                    //      if(d.HasMember("task") && d["task"].IsInt()){
                    //         SetTask(Task(d['task'].GetInt()));
                    //         if(GetTask() == TRAIN) {
                    //           Train(d["modelId"].GetInt());
                    //         } // the else being aggregate or evaluate
                    //      }
                    default:
                        break;
                    }
                }
            }
        }

    }
}

void FLNode::Send(Ipv4Address adrs, rapidjson::Document &d) {
 
 if (!m_socket){
    Ptr<SocketFactory> socketFactory = GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
    m_socket = socketFactory->CreateSocket();
    m_socket->Bind();
 }
    rapidjson::StringBuffer packetInfo;
    rapidjson::Writer<rapidjson::StringBuffer> writer(packetInfo);
    d.Accept(writer);

    Ptr<Packet> packet = Create<Packet>(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize());
    int result = m_socket->SendTo(packet,0,InetSocketAddress(adrs, m_port));
    // NS_LOG_DEBUG("sent "<< result << " " << packetInfo.GetString());
}

void FLNode::Candidater() {
  Blockchain* bc = Blockchain::getInstance();
  Ipv4Address adr = bc->getBCAddress();
  NS_LOG_DEBUG("sending candidature to " << adr);
  rapidjson::Document d;
  rapidjson::Value value;
  d.SetObject(); 
  value = CANDIDATURE;
  d.AddMember("message_type", value, d.GetAllocator());
  value = id ;
  d.AddMember("node_id", value, d.GetAllocator());
  value = dataset_size ; 
  d.AddMember("dataset_size", value, d.GetAllocator());
  value = task;
  d.AddMember("task", value, d.GetAllocator());
  value = freq ;
  d.AddMember("freq", value, d.GetAllocator());
  value = trans_rate ;
  d.AddMember("trans_rate", value, d.GetAllocator());
  value = availability ;
  d.AddMember("availability", value, d.GetAllocator());
  value = honesty ;
  d.AddMember("honesty", value, d.GetAllocator());
  value = dropout ; 
  d.AddMember("dropout", value, d.GetAllocator());
  
  //TODO change the destination adr to the blockchain
  Send(adr, d);
}

void FLNode::Train(int globalModelId) {
  // Train implementation
}

void FLNode::SendModel() {
  // SendModel implementation
}


}
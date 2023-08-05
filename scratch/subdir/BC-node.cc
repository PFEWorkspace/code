<<<<<<< HEAD
#include  "FL-node.h"
#include "FL-task-initiator.h"
#include "BC-node.h"
=======


#include "BC-node.h"


>>>>>>> FL2
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
<<<<<<< HEAD
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
=======
                                          MakeIpv4AddressChecker());
  return tid;
}

BCNode::BCNode(){
   NS_LOG_FUNCTION_NOARGS();

}

BCNode::~BCNode(){
  NS_LOG_FUNCTION_NOARGS();
}

>>>>>>> FL2

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
<<<<<<< HEAD
        char *packetInfo = new char[packet->GetSize () + 1];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            packet->CopyData (reinterpret_cast<uint8_t*>(packetInfo), packet->GetSize ());
            // NS_LOG_INFO("I'm "<< GetNode()->GetId() << "received " << packet->GetSize() << " bytes from "
            //                         << InetSocketAddress::ConvertFrom(from).GetIpv4()
            //                         << " content: "<< packetInfo) ;
            std::string data = packetInfo ; 
=======
        unsigned char *packetInfo = new unsigned char[packet->GetSize()];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            packet->CopyData(packetInfo, packet->GetSize ());
            NS_LOG_INFO("I'm a bc node id: "<< GetNode()->GetId() << " received " << packet->GetSize() << " bytes from "
                                    << InetSocketAddress::ConvertFrom(from).GetIpv4()
                                    << " content: "<< packetInfo) ;
            std::string data(reinterpret_cast<char*>(packetInfo), packet->GetSize()) ;
>>>>>>> FL2
            rapidjson::Document d;
           
            if(ParseJSON(data,d)){
                if(d.HasMember("message_type") && d["message_type"].IsInt()){
                    switch (d["message_type"].GetInt())
                    {
                    case NEWTASK : //NEWTASK 
<<<<<<< HEAD
                        /* 
                            newtask is the message sent by the initializer to declare a new task
                            as a response the FL nodes will send their condidature to the blockchain
                         task_id,model_id,rounds,target_acc,num_participants,num_aggregators}
                         */
                      
=======
                       
>>>>>>> FL2
                      // TODO : mise a jour de la blokchain with the new task , 
                      break;
                    case CANDIDATURE : 
                     // receive les candidatures and treat them
<<<<<<< HEAD

                     // teste ida got all les candidatures ou time passed ou
                      
                      AiHelper aiH = AiHelper () ; 
                      aiH.ExactSelection()
=======
                    TreatCandidature(d);
                     // teste ida got all les candidatures ou time passed ou
                      
                    //   AiHelper* ai = AiHelper::getInstance();
                    //   ai.ExactSelection();
>>>>>>> FL2

                      //print blockchainaggregators 
                      // step get the addresses from blockchain list of node 
                      // send the messages MODEL to trainers {msgtype:selection, worktype: aggregatio ou learning, modelId}

                        break;
                    case MODEL : //MODEL
<<<<<<< HEAD
=======
                    NS_LOG_INFO("I'm a bc node id: "<< GetNode()->GetId() << " received " << packet->GetSize() << " bytes from "
                                    << InetSocketAddress::ConvertFrom(from).GetIpv4()
                                    << " content: "<< packetInfo) ;
>>>>>>> FL2
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


<<<<<<< HEAD
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
=======
>>>>>>> FL2

void BCNode :: WriteTransaction (){

}

<<<<<<< HEAD
}
=======
FLNodeStruct 
BCNode::docToFLNodeStruct(rapidjson::Document &d){
    FLNodeStruct node = FLNodeStruct();
    if(d.HasMember("nodeId") && d["nodeId"].IsInt()){
        node.nodeId = d["nodeId"].GetInt();
    }
    if(d.HasMember("availability") && d["availability"].IsBool()){
        node.availability = d["availability"].GetBool();
    }
    if(d.HasMember("honesty") && d["honesty"].IsDouble()){
        node.honesty = d["honesty"].GetDouble();
    }
    if(d.HasMember("datasetSize")&& d["datasetSize"].IsInt()){
        node.datasetSize = d["datasetSize"].GetInt();
    }
    if(d.HasMember("freq") && d["freq"].IsInt()){
        node.freq = d["freq"].GetInt();
    }
    if(d.HasMember("transRate") && d["transRate"].IsInt()){
        node.transRate = d["transRate"].GetInt();
    }
    if(d.HasMember("task") && d["task"].IsInt()){
        node.task = d["task"].GetInt();
    }
    return node;
}

void
BCNode::TreatCandidature(rapidjson::Document &d){
    
    Blockchain* bc = Blockchain::getInstance();
    //save the candidature in the blockchain
    bc->SetNodeInfo(bc->GetReceivedCandidatures(), docToFLNodeStruct(d));
    bc->IncReceivedCandidatures(); 
    NS_LOG_INFO("bc node "<< GetNode()->GetId() << " received a candidature number " << bc->GetReceivedCandidatures());
    if(bc->GetReceivedCandidatures()==bc->getNumFLNodes()) //received all candidatur
    {
       //selecting nodes
       NS_LOG_INFO("bc node "<< GetNode() << " starting the selection");
        AiHelper* ai = AiHelper::getInstance();
        ai->Selection(); //the selected nodes are set in the bc

        // write the selection message 
        rapidjson::Document d;
        rapidjson::Value value;
        d.SetObject(); 
        value = SELECTION;
        d.AddMember("message_type", value, d.GetAllocator());
        value = TRAIN;
        d.AddMember("task", value, d.GetAllocator());

        // get selected nodes addresses
        std::vector<Ipv4Address> adrs;
        int numSelectedTrainers = bc->getNumTrainers();
        for(int i=0; i < numSelectedTrainers; i++){
            adrs[i] = bc->getFLAddress( bc->getTrainer(i));
        } 
        //sending the message to all selected trainers
        SendTo(d,adrs);
    }    
        
}

}

>>>>>>> FL2

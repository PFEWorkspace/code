#include "FL-task-initiator.h"
#include "FL-node.h"

#include "Blockchain.h"

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
                                          MakeIpv4AddressChecker());
  return tid;
}

BCNode::BCNode(){
   NS_LOG_FUNCTION_NOARGS();

}

BCNode::~BCNode(){
  NS_LOG_FUNCTION_NOARGS();
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
        unsigned char *packetInfo = new unsigned char[packet->GetSize()];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            packet->CopyData(packetInfo, packet->GetSize ());
            NS_LOG_INFO("I'm a bc node id: "<< GetNode()->GetId() << " received " << packet->GetSize() << " bytes from "
                                    << InetSocketAddress::ConvertFrom(from).GetIpv4()
                                    << " content: "<< packetInfo) ;
            
            std::string data(reinterpret_cast<char*>(packetInfo), packet->GetSize()) ;
            rapidjson::Document d;
            Blockchain* bc = Blockchain::getInstance();
           
            if(ParseJSON(data,d)){
                   
                if(d.HasMember("message_type") && d["message_type"].IsInt()){
                    switch (d["message_type"].GetInt())
                    {
                        case NEWTASK : //NEWTASK
                        if(d.HasMember("task_id") && d["task_id"].IsInt()){
                            bc->setTaskId( d["task_id"].GetInt());
                            bc->SetActualFLRound(0);
                            bc->SetCurrentBlockId(); 
                        }
                        
                        // the max delay for receiving candidature is: size_message=120 / smallest_trans_rate=150 + 2 seconds for
                        Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(120/150 + 2), [this]() { Selection(); });
                        break;

                        case CANDIDATURE : 
                        // receive les candidatures and treat them
                        
                        TreatCandidature(d);
                        break;

                        case MODEL : //MODEL
                        // NS_LOG_INFO("I'm a bc node id: "<< GetNode()->GetId() << " received " << packet->GetSize() << " bytes from "
                        //                 << InetSocketAddress::ConvertFrom(from).GetIpv4()
                        //                 << " content: "<< packetInfo) ;
                       
                        break ;
                    case EVALUATE : // EVALUATE
                    break; 
                    
                    default:
                        break;
                    }
                
                }
                // saving the transaction in the blockchain
                bc->WriteTransaction(bc->getCurrentBlockId(),GetNode()->GetId(),d);
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
   
    for (Ipv4Address destAddr : addresses) {
       m_socket->SendTo(packet, 0, InetSocketAddress(destAddr, m_port));
        // NS_LOG_INFO("sent traininig task to "<< destAddr);
    }
}


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
    //  NS_LOG_INFO("bc node " << GetNode()->GetId() << " received a candidature number " << bc->GetReceivedCandidatures() << " on " << bc->getNumFLNodes());
    // if(bc->GetReceivedCandidatures()==bc->getNumFLNodes()) //received all candidatur
    // {
       
    // }   
        
}

void
BCNode::Selection(){
    //selecting nodes
    Blockchain* bc = Blockchain::getInstance();
    NS_LOG_INFO("received " << bc->GetReceivedCandidatures() << " on " << bc->getNumFLNodes() << " ,starting the selection");
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
    int id;
    for(int i=0; i < numSelectedTrainers; i++){
        id =bc->getTrainer(i);
        adrs.push_back(bc->getFLAddress(id));
    } 
        
    //sending the message to all selected trainers
    SendTo(d,adrs);
}

}


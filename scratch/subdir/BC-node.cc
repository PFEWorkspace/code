

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
           
            if(ParseJSON(data,d)){
                if(d.HasMember("message_type") && d["message_type"].IsInt()){
                    switch (d["message_type"].GetInt())
                    {
                    case NEWTASK : //NEWTASK 
                       
                      // TODO : mise a jour de la blokchain with the new task , 
                      break;
                    case CANDIDATURE : 
                     // receive les candidatures and treat them
                    TreatCandidature(d);
                     // teste ida got all les candidatures ou time passed ou
                      
                    //   AiHelper* ai = AiHelper::getInstance();
                    //   ai.ExactSelection();

                      //print blockchainaggregators 
                      // step get the addresses from blockchain list of node 
                      // send the messages MODEL to trainers {msgtype:selection, worktype: aggregatio ou learning, modelId}

                        break;
                    case MODEL : //MODEL
                    NS_LOG_INFO("I'm a bc node id: "<< GetNode()->GetId() << " received " << packet->GetSize() << " bytes from "
                                    << InetSocketAddress::ConvertFrom(from).GetIpv4()
                                    << " content: "<< packetInfo) ;
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



void BCNode :: WriteTransaction (){

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


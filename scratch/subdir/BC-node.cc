

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
        char *packetInfo = new char[packet->GetSize () ];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            packet->CopyData(reinterpret_cast<uint8_t*>(packetInfo), packet->GetSize ());
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

                     // teste ida got all les candidatures ou time passed ou
                      
                    //   AiHelper aiH = AiHelper () ; 
                    //   aiH.ExactSelection();

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



void BCNode :: WriteTransaction (){

}

}


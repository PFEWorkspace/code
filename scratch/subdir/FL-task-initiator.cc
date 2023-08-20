#include "FL-task-initiator.h"


namespace ns3{

NS_LOG_COMPONENT_DEFINE("TaskInitiatorApp");

TypeId
Initiator::GetTypeId()
{
    static TypeId tid = TypeId("Initiator")
                            .SetParent<Application>()
                            .AddConstructor<Initiator>()           
                            .AddAttribute("Port",
                                          "Destination app port.",
                                          UintegerValue(8833),
                                          MakeUintegerAccessor(&Initiator::m_destPort),
                                          MakeUintegerChecker<uint32_t>());
                           
    return tid;
}
Initiator::Initiator()
{
    NS_LOG_FUNCTION_NOARGS();
    
}

Initiator::~Initiator()
{
    NS_LOG_FUNCTION_NOARGS();
}

void
Initiator::DoDispose()
{
    NS_LOG_FUNCTION_NOARGS();
    m_socket = nullptr;
    // chain up
    Application::DoDispose();
}
 void 
 Initiator::setNodesInfo(FLNodeStruct* nodesInfo, int numNodes) {
      NS_LOG_FUNCTION_NOARGS();
    
      for(int i=0; i<numNodes;i++){
        m_nodesInfo[i] = nodesInfo[i];
      }

    }

void
Initiator::StartApplication()
{

    /* send a soket containing all the data related to the fl task using rapidjson
     to all the subnetwork using a broadcase adress
    */
//    NS_LOG_INFO("starting app");

   rapidjson::Document Info; 
   rapidjson::Value value;
   Info.SetObject();
   enum CommunicationType msg = NEWTASK;
   value = msg;
   Info.AddMember("message_type", value, Info.GetAllocator());

    //initialize model and task fl in python side
    AiHelper* ai = AiHelper::getInstance();
    MLModelRefrence model = ai->initializeFL(m_nodesInfo, m_numNodes);
    // NS_LOG_INFO("task id " << model.taskId);
    value = model.taskId;
    Info.AddMember("task_id", value, Info.GetAllocator());
    value = model.modelId ;
    Info.AddMember("model_id",value, Info.GetAllocator());
    value = m_targetAcc;
    Info.AddMember("target_acc",value, Info.GetAllocator());
    value = m_rounds;
    Info.AddMember("rounds",value, Info.GetAllocator());
    value = m_numParticipants;
    Info.AddMember("num_participants",value, Info.GetAllocator());
    value =m_numAggregators;
    Info.AddMember("num_aggregators",value, Info.GetAllocator());
    

    // Stringify the DOM
    rapidjson::StringBuffer packetInfo;
    rapidjson::Writer<rapidjson::StringBuffer> writer(packetInfo);
    Info.Accept(writer);
    // NS_LOG_INFO(packetInfo.GetString());

    Ptr<SocketFactory> socketFactory = GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
    m_socket = socketFactory->CreateSocket();
    m_socket->SetAllowBroadcast(true);
    m_socket->Bind();
     Ptr<Packet> packet = Create<Packet>(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize());
    for (Ipv4Address destAddr : m_destAddr) {
        m_socket->SendTo(packet, 0, InetSocketAddress(destAddr, m_destPort));
        
    }
    // m_socket->SendTo(packet,0,InetSocketAddress(m_destAddr, m_destPort));
    // m_socket->Connect(InetSocketAddress(m_destAddr, m_destPort));
    // int result = m_socket->Send(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize(),0);
    // NS_LOG_INFO("number of bytes sent " << result); 

}

void
Initiator::StopApplication(){
    NS_LOG_FUNCTION_NOARGS();
    // Simulator::Cancel(m_sendEvent);
}


std::string ReadFileToString(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filePath << std::endl;
        return "";
    }
    
    std::string fileContents((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    file.close();
    return fileContents;
}

bool ParseJSON(const std::string& jsonString, rapidjson::Document& document) {
    document.Parse(jsonString.c_str());
    if (document.HasParseError()) {
        NS_LOG_ERROR( "Failed to parse JSON: "<< rapidjson::GetParseError_En(document.GetParseError()));
        return false;
    }
    return true;
}






// ==============================================
// RECEIVER
// ==============================================

TypeId
Receiver::GetTypeId()
{
    static TypeId tid = TypeId("Receiver")
                            .SetParent<Application>()
                            .AddConstructor<Receiver>()
                            .AddAttribute("Port",
                                          "Listening port.",
                                          UintegerValue(8833),
                                          MakeUintegerAccessor(&Receiver::m_port),
                                          MakeUintegerChecker<uint32_t>());
    return tid;
}

Receiver::Receiver()
{
    NS_LOG_FUNCTION_NOARGS();
}

Receiver::~Receiver()
{
    NS_LOG_FUNCTION_NOARGS();
}

void
Receiver::DoDispose()
{
    NS_LOG_FUNCTION_NOARGS();

    m_socket = nullptr;
    // chain up
    Application::DoDispose();
}

void
Receiver::StartApplication()
{
    NS_LOG_FUNCTION_NOARGS();

    
    if (!m_socket)
    {
        Ptr<SocketFactory> socketFactory =
            GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
        m_socket = socketFactory->CreateSocket();
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
        m_socket->Bind(local);
    }

    m_socket->SetRecvCallback(MakeCallback(&Receiver::Receive, this));
}

void
Receiver::StopApplication()
{
    NS_LOG_FUNCTION_NOARGS();

    if (m_socket)
    {
        m_socket->SetRecvCallback(MakeNullCallback<void, Ptr<Socket>>());
    }
}


void
Receiver::Receive(Ptr<Socket> socket)
{
    

    Ptr<Packet> packet;
    Address from;
  
    while ((packet = socket->RecvFrom(from)))
    {
        char *packetInfo = new char[packet->GetSize () ];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            // packet->CopyData (reinterpret_cast<uint8_t*>(packetInfo), packet->GetSize ());
            // NS_LOG_INFO("I'm "<< GetNode()->GetId() << "received " << packet->GetSize() << " bytes from "
            //                         << InetSocketAddress::ConvertFrom(from).GetIpv4()
            //                         << " content: "<< packetInfo) ;
        }

    }
}

}
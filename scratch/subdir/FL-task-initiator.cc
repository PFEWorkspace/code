#include "FL-task-initiator.h"


namespace ns3{

NS_LOG_COMPONENT_DEFINE("TaskInitiatorApp");

TypeId
Initiator::GetTypeId()
{
    static TypeId tid = TypeId("Initiator")
                            .SetParent<Application>()
                            .AddConstructor<Initiator>()
                            .AddAttribute("Destination",
                                          "Target host address.",
                                          Ipv4AddressValue("255.255.255.255"),
                                          MakeIpv4AddressAccessor(&Initiator::m_destAddr),
                                          MakeIpv4AddressChecker())
                            .AddAttribute("Source",
                                          "Source host address.",
                                          Ipv4AddressValue(),
                                          MakeIpv4AddressAccessor(&Initiator::m_srcAddr),
                                          MakeIpv4AddressChecker())              
                            .AddAttribute("Port",
                                          "Destination app port.",
                                          UintegerValue(8833),
                                          MakeUintegerAccessor(&Initiator::m_destPort),
                                          MakeUintegerChecker<uint32_t>())
                            // .AddAttribute("budget",
                            //               "Budget for FL task.",
                            //               UintegerValue(0),
                            //               MakeUintegerAccessor(&Initiator::m_budget),
                            //               MakeUintegerChecker<uint32_t>(1))                                 
                            ;
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
Initiator::StartApplication()
{

    /* send a soket containing all the data related to the fl task using rapidjson
     to all the subnetwork using a broadcase adress
    */
   NS_LOG_INFO("starting app");
   std::string config = ReadFileToString("/home/hiba/Desktop/PFE/ns-allinone-3.38/ns-3.38/config.json");

   rapidjson::Document d;
   rapidjson::Document Info; 
   rapidjson::Value value;
   Info.SetObject();
   enum CommunicationType msg = NEWTASK;
   value = msg;
   Info.AddMember("message_type", value, Info.GetAllocator());
   if(ParseJSON(config,d)){
        if (d.HasMember("federated_learning") && d["federated_learning"].IsObject()) {
            const rapidjson::Value& FL_config = d["federated_learning"];
           
            if(FL_config.HasMember("rounds") && FL_config["rounds"].IsInt()){
                m_rounds = FL_config["rounds"].GetInt();
                value = m_rounds;
                Info.AddMember("rounds",value, Info.GetAllocator());
            }
            // if(FL_config.HasMember("target_accuracy") && FL_config["target_accuracy"].IsDouble()){
            //     m_targetAccuracy = FL_config["target_accuracy"].GetDouble();
            //     value.SetDouble( m_targetAccuracy);
            //     Info.AddMember("target_Accuracy",value, Info.GetAllocator());
            // }
            // if(FL_config.HasMember("epochs") && FL_config["epochs"].IsInt()){
            //     m_epochs = FL_config["epochs"].GetInt();
            //     value = m_epochs;
            //     Info.AddMember("epochs",value, Info.GetAllocator());
            // }
            // if(FL_config.HasMember("batch_size") && FL_config["batch_size"].IsInt()){
            //     m_batchSize = FL_config["batch_size"].GetInt();
            //     value = m_batchSize ;
            //     Info.AddMember("batch_size",value, Info.GetAllocator());
            // }
            if(FL_config.HasMember("budget") && FL_config["budget"].IsUint()){
                m_budget = FL_config["budget"].GetUint();
                value = m_budget;
                Info.AddMember("budget",value, Info.GetAllocator());
            }
        }
        //  if (d.HasMember("model") && d["model"].IsObject()) {
        //     const rapidjson::Value& model = d["model"];
           
        //     if(model.HasMember("name") && model["name"].IsString()){
        //         m_model = model["name"].GetString();
        //         value.SetString(m_model.c_str(),m_model.size());
        //         Info.AddMember("model",value,Info.GetAllocator());
        //     }
        // }

         if (d.HasMember("nodes") && d["nodes"].IsObject()) {
            const rapidjson::Value& nodes = d["nodes"];
           
            if(nodes.HasMember("source") && nodes["source"].IsString()){
                std::string filename = nodes["source"].GetString();
                // initializing the FL in the python side and get initial model reference
                ns3::AiHelper aihelper = AiHelper();
                MLModelRefrence model = aihelper.initializeFL(filename);
                NS_LOG_INFO ("initial model" << model.modelId << " " << model.nodeId);
            }
        }
        
        // Stringify the DOM
        rapidjson::StringBuffer packetInfo;
        rapidjson::Writer<rapidjson::StringBuffer> writer(packetInfo);
        Info.Accept(writer);
        NS_LOG_INFO(packetInfo.GetString());

        Ptr<SocketFactory> socketFactory =
            GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
        m_socket = socketFactory->CreateSocket();
        m_socket->SetAllowBroadcast(true);
        m_socket->Bind();
        // Ptr<Packet> packet = Create<Packet>(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize());
        // m_socket->SendTo(packet,0,InetSocketAddress(m_destAddr, m_destPort));
        m_socket->Connect(InetSocketAddress(m_destAddr, m_destPort));
        int result = m_socket->Send(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize(),0);
        NS_LOG_INFO(result);
   }

   

}

void
Initiator::StopApplication(){
    NS_LOG_FUNCTION_NOARGS();
    // Simulator::Cancel(m_sendEvent);
}

void
Initiator::SendPacket(rapidjson::Document d, Address &outgoingAddr)
{

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
        char *packetInfo = new char[packet->GetSize () + 1];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            packet->CopyData (reinterpret_cast<uint8_t*>(packetInfo), packet->GetSize ());
            NS_LOG_INFO("I'm "<< GetNode()->GetId() << "received " << packet->GetSize() << " bytes from "
                                    << InetSocketAddress::ConvertFrom(from).GetIpv4()
                                    << " content: "<< packetInfo) ;
        }

    }
}

}
#include "FL-task-initiator.h"


namespace ns3{

NS_LOG_COMPONENT_DEFINE("TaskInitiatorApp");

TypeId
Initiator::GetTypeId()
{
    static TypeId tid = TypeId("Initiator")
                            .SetParent<Application>()
                            .AddConstructor<Initiator>()
                            // .AddAttribute("Destination",
                            //               "Target host address.",
                            //               Ipv4AddressValue("255.255.255.255"),
                            //               MakeIpv4AddressAccessor(&Initiator::m_destAddr),
                            //               MakeIpv4AddressChecker())
                            // .AddAttribute("Source",
                            //               "Source host address.",
                            //               Ipv4AddressValue("255.255.255.255"),
                            //               MakeIpv4AddressAccessor(&Initiator::m_srcAddr),
                            //               MakeIpv4AddressChecker())              
                            // .AddAttribute("Port",
                            //               "Destination app port.",
                            //               UintegerValue(8833),
                            //               MakeUintegerAccessor(&Initiator::m_destPort),
                            //               MakeUintegerChecker<uint32_t>())
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
   if(ParseJSON(config,d)){
        if (d.HasMember("federated_learning") && d["federated_learning"].IsObject()) {
            const rapidjson::Value& FL_config = d["federated_learning"];
           
            if(FL_config.HasMember("rounds") && FL_config["rounds"].IsInt()){
                m_rounds = FL_config["rounds"].GetInt();  
            }
            if(FL_config.HasMember("target_accuracy") && FL_config["target_accuracy"].IsDouble()){
                m_targetAccuracy = FL_config["target_accuracy"].GetDouble();
            }
            if(FL_config.HasMember("epochs") && FL_config["epochs"].IsInt()){
                m_epochs = FL_config["epochs"].GetInt();
            }
            if(FL_config.HasMember("batch_size") && FL_config["batch_size"].IsInt()){
                m_batchSize = FL_config["batch_size"].GetInt();
            }
            if(FL_config.HasMember("budget") && FL_config["budget"].IsUint()){
                m_budget = FL_config["rounds"].GetUint();
            }
        }
         if (d.HasMember("model") && d["model"].IsObject()) {
            const rapidjson::Value& model = d["model"];

            if(model.HasMember("name") && model["name"].IsString()){
                m_model = model["name"].GetString();
            }
        }
        //    NS_LOG_INFO(m_budget << " " << m_model <<" " << m_batchSize <<" " << m_rounds<< " " << m_epochs<< " " << m_targetAccuracy); 
   }

   

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

}
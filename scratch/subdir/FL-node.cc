#include "FL-task-initiator.h"
#include "BC-node.h"

#include  "FL-node.h"

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

void FLNode::SetLearningCost(double learningCost){
  this->learning_cost = learningCost;
}
  double FLNode::GetLearningCost() const{
    return learning_cost ;
  }

  double
  FLNode:: GetEvaluationCost() const{
      return evaluationCost ;
  }

  void FLNode::SetCommunicationCost(double communicationCost){
    this->communication_cost = communicationCost;
  }
  double FLNode::GetCommunicationCost() const{
    return communication_cost;
  }

void FLNode::Init(FLNodeStruct n, int modelsize, double testSize){
  id = n.nodeId ;
  availability = n.availability ;
  honesty = n.honesty ;
  dataset_size = n.datasetSize;
  freq = n.freq;
  trans_rate = n.transRate ;
  task = Task(n.task);
  dropout = n.dropout ;
  malicious = n.malicious;
  model_size = modelsize;
  learning_cost = n.datasetSize / n.freq ;
  communication_cost = model_size / n.transRate ;
  testPartitionSize = testSize ;
  evaluationCost = testSize * n.datasetSize / n.freq ;
 
}

void FLNode::ResetRound(){
  AiHelper* ai = AiHelper::getInstance();
  FLNodeStruct info = ai->getNodeInfo(GetNode()->GetId());
  Init(info, model_size, testPartitionSize);
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
    // NS_LOG_DEBUG("i'am "<< GetNode()->GetId() << " and i received a packet");

    while ((packet = socket->RecvFrom(from)))
    {
        unsigned char *packetInfo = new unsigned char[packet->GetSize()];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            packet->CopyData(packetInfo, packet->GetSize () );
            NS_LOG_DEBUG("I'm "<< GetNode()->GetId() << "received " << packet->GetSize() << " bytes from "
                                    << InetSocketAddress::ConvertFrom(from).GetIpv4()
                                    << " content: "<< packetInfo) ;

            std::string data(reinterpret_cast<char*>(packetInfo), packet->GetSize()) ; 
            rapidjson::Document d;
            std::vector<MLModel> models;
            MLModel m;
            int aggType ;
            if(ParseJSON(data,d)){
                if(d.HasMember("message_type") && d["message_type"].IsInt()){
                    switch (d["message_type"].GetInt())
                    {
                      case NEWTASK: 
                        // Candidater(InetSocketAddress::ConvertFrom(from).GetIpv4());
                        // the average packet size for a candidature is around 120 bytes and the trans_rate is in Mbps
                       Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(120/trans_rate ), [this]() { Candidater(); });
                        break;
                    case SELECTION :
                         if(d.HasMember("task") && d["task"].IsInt()){
                            SetTask(Task(d["task"].GetInt()));
                            if(GetTask() == TRAIN) {
                              // Train();
                                Train();
                            } // the else being aggregate or evaluate
                         }
                      break;
                    case EVALUATION:
                    NS_LOG_INFO("EVALUATION");
                      m = BCNode::DocToMLModel(d);
                      Simulator::ScheduleWithContext(GetNode()->GetId(), Seconds((GetEvaluationCost()+GetCommunicationCost())),[this,m](){Evaluate(m);});     
                    break;
                    case AGGREGATION:
                    NS_LOG_INFO("AGGREGATION");
                        aggType = d["aggregation_type"].GetInt(); // 1:INTERMEDIAIRE, 2:GLOBAL
                        models = docToModels(d);
                        Simulator::ScheduleWithContext(GetNode()->GetId(), Seconds((GetLearningCost()+GetCommunicationCost())), [this, models,aggType](){Aggregate(models,aggType);});
                    break; 
                    case NEWROUND:
                    ResetRound();
                    Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(120/trans_rate ), [this]() { Candidater(); });
                    break;   
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

    // Ptr<Packet> packet = Create<Packet>(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize());
    // int result = m_socket->SendTo(packet,0,InetSocketAddress(adrs, m_port));
     m_socket->Connect(InetSocketAddress(adrs, m_port));
    m_socket->Send(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize(),0);
    // NS_LOG_DEBUG("sent "<< result << " " << packetInfo.GetString());
}

void FLNode::Candidater() {
  Blockchain* bc = Blockchain::getInstance();
  Ipv4Address adr = bc->getBCAddress();
  // NS_LOG_DEBUG("sending candidature to " << adr);
  rapidjson::Document d;
  rapidjson::Value value;
  d.SetObject(); 
  value = CANDIDATURE;
  d.AddMember("message_type", value, d.GetAllocator());
  value = id ;
  d.AddMember("nodeId", value, d.GetAllocator());
  value = dataset_size ; 
  d.AddMember("datasetSize", value, d.GetAllocator());
  value = task;
  d.AddMember("task", value, d.GetAllocator()); // this task (before selection) is the previous assigned task: needed in the DRL selection 
  value = freq ;
  d.AddMember("freq", value, d.GetAllocator());
  value = trans_rate ;
  d.AddMember("transRate", value, d.GetAllocator());
  value = availability ;
  d.AddMember("availability", value, d.GetAllocator());
  value = honesty ;
  d.AddMember("honesty", value, d.GetAllocator());
  // value = dropout ; 
  // d.AddMember("dropout", value, d.GetAllocator());
  
  Send(adr, d);
}

void
FLNode::SendModel(MLModel model, Ipv4Address adrs){
  rapidjson::Document d;
  rapidjson::Value value;
  d.SetObject(); 
  value = MODEL;
  d.AddMember("message_type", value, d.GetAllocator());
  value = model.modelId ;
  d.AddMember("modelId", value, d.GetAllocator());
  value = model.nodeId;
  d.AddMember("nodeId", value, d.GetAllocator());
  value = model.taskId ;
  d.AddMember("taskId", value, d.GetAllocator());
  value = model.round ;
  d.AddMember("round", value, d.GetAllocator());
  value = model.type; 
  d.AddMember("type", value, d.GetAllocator());
  value = model.positiveVote;
  d.AddMember("positiveVote", value, d.GetAllocator());
  value = model.negativeVote ;
  d.AddMember("negativeVote", value, d.GetAllocator());
  value = model.evaluator1 ;
  d.AddMember("evaluator1", value, d.GetAllocator());
  value = model.evaluator2 ;
  d.AddMember("evaluator2", value, d.GetAllocator());
  value = model.evaluator3 ;
  d.AddMember("evaluator3", value, d.GetAllocator());
  value = model.aggregated ;
  d.AddMember("aggregated", value, d.GetAllocator());
  value = model.aggModelId ;
  d.AddMember("aggModelId", value, d.GetAllocator());
  value = model.accuracy ; 
  d.AddMember("accuracy", value, d.GetAllocator());
  value = model.acc1;
  d.AddMember("acc1", value, d.GetAllocator());
  value = model.acc2;
  d.AddMember("acc2", value, d.GetAllocator());
  value = model.acc3;
  d.AddMember("acc3", value, d.GetAllocator());

  Send(adrs, d);

}

void FLNode::Train() {
  if(!dropout){
     AiHelper* ai = AiHelper::getInstance();
  MLModel model = ai->train(id);

  Blockchain* bc = Blockchain::getInstance();
  Ipv4Address adr = bc->getBCAddress();
 
  // NS_LOG_DEBUG("I'am " << id << " sending model to " << adr);
  Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(GetLearningCost()+GetCommunicationCost()), [this, model,adr]() {SendModel(model, adr);});
  }
}
void
FLNode::Evaluate(MLModel model){
  NS_LOG_INFO("fl node evaluation");
  AiHelper* ai = AiHelper::getInstance();
  MLModel evaluatedmodel = ai->evaluate(model, GetNode()->GetId());

  Blockchain* bc = Blockchain::getInstance();
  Ipv4Address adr = bc->getBCAddress();
 
   NS_LOG_INFO("I'am " << id << " sending evaluated model to " << adr);
  SendModel(evaluatedmodel, adr);
}

std::vector<MLModel>
FLNode::docToModels(rapidjson::Document& d){
  std::vector<MLModel> mlModels;
  if (!d.HasParseError() && d.IsObject()) {
        const rapidjson::Value& modelsArray = d["models"];
        if (modelsArray.IsArray()) {
          for (rapidjson::SizeType i = 0; i < modelsArray.Size(); ++i) {
                const rapidjson::Value& model = modelsArray[i];
                MLModel mlModel;

                mlModel.modelId = model["modelId"].GetInt();
                mlModel.nodeId = model["nodeId"].GetInt();
                mlModel.taskId = model["taskId"].GetInt();
                mlModel.round = model["round"].GetInt();
                mlModel.type = model["type"].GetInt();
                mlModel.positiveVote = model["positiveVote"].GetInt();
                mlModel.negativeVote = model["negativeVote"].GetInt();
                mlModel.evaluator1 = model["evaluator1"].GetInt();
                mlModel.evaluator2 = model["evaluator2"].GetInt();
                mlModel.evaluator3 = model["evaluator3"].GetInt();
                mlModel.aggregated = model["aggregated"].GetBool();
                mlModel.aggModelId = model["aggModelId"].GetInt();
                mlModel.accuracy = model["accuracy"].GetDouble();
                mlModel.acc1 = model["acc1"].GetDouble();
                mlModel.acc2 = model["acc2"].GetDouble();
                mlModel.acc3 = model["acc3"].GetDouble();

                mlModels.push_back(mlModel);
            }
        }
  } 

  return mlModels ;       

}

void
FLNode::Aggregate(std::vector<MLModel> models, int aggType){
  AiHelper* ai = AiHelper::getInstance();
  MLModel model = ai->aggregate(models, GetNode()->GetId(), aggType);

  Blockchain* bc = Blockchain::getInstance();
  Ipv4Address adr = bc->getBCAddress();
 
  NS_LOG_DEBUG("I'am " << id << " sending aggregated model " << model.modelId);
  SendModel(model, adr);
}
}
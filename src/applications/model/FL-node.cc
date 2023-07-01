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
                          .AddAttribute("DatasetSize", "Size of the dataset", UintegerValue(0),
                                        MakeUintegerAccessor(&FLNode::dataset_size),
                                        MakeUintegerChecker<uint32_t>())
                          .AddAttribute("Beta", "CPU cycle to train one data unit", DoubleValue(0.0),
                                        MakeDoubleAccessor(&FLNode::beta),
                                        MakeDoubleChecker<double>())
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

void FLNode::SetDatasetSize(uint32_t size) {
  dataset_size = size;
}

uint32_t FLNode::GetDatasetSize() const {
  return dataset_size;
}

void FLNode::SetBeta(double beta) {
  this->beta = beta;
}

double FLNode::GetBeta() const {
  return beta;
}

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

void FLNode::DoDispose() {
   NS_LOG_FUNCTION_NOARGS();
    m_socket = nullptr;
    // chain up
    Application::DoDispose();
}

void FLNode::StartApplication() {
  NS_LOG_FUNCTION_NOARGS();

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
   
}

void FLNode::Send(rapidjson::Document d, Address &outgoingAddr) {
 
   
}

void FLNode::Condidater(std::string FLTaskId) {
  // Condidater implementation
}

void FLNode::Train() {
  // Train implementation
}

void FLNode::SendModel() {
  // SendModel implementation
}
}
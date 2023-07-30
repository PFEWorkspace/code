#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/netanim-module.h"
#include "ai-helper.h"
#include "FL-node.h"
#include "FL-task-initiator.h"

#include <string>
#include <ctime>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FLExperimentSimulation");

FLNodeStruct* GetNodesFromFile(const std::string& filename,  int& numNodes);


int
main(int argc, char* argv[])
{

  LogComponentEnable("TaskInitiatorApp",LOG_LEVEL_INFO);
  LogComponentEnable("FLExperimentSimulation",LOG_LEVEL_INFO);
  LogComponentEnable("AiHelper",LOG_LEVEL_INFO);
  Time::SetResolution(Time::NS);
  
    int numFlNodes= 100;
    int numParticipants=50;
    int numAggregators=20;
    std::string nodes_source= "";
    int flrounds = 10;
    double targetAccuracy = 0.0;
    const uint16_t Port = 8833;
    double xPosMin = 0;
    double xPosMax = 30;
    double yPosMin = 0;
    double yPosMax = 30;
    double zPosMin = 0;
    double zPosMax = 0;


    std::string animFile = "FL-animation.xml";

    //---------------------------------------------
    //----- Command line and getting parameters
    //----------------------------------------------
    CommandLine cmd;
    cmd.AddValue ("numNodes","the total number of nodes", numFlNodes);
    cmd.AddValue ("participantsPerRound","the number of participants per round",numParticipants);
    cmd.AddValue ("aggregatorsPerRound","the number of aggregators per pround",numAggregators);
    cmd.AddValue ("source","the path and name of the file to get initial data about nodes", nodes_source);
    cmd.AddValue ("flRounds", "the number of rounds per FL task", flrounds);
    cmd.AddValue ("targetAccuracy", "the target accuracy for the FL task",targetAccuracy);
    cmd.Parse (argc, argv);


   
   //--------------------------------------------
    //-- Create nodes and network stacks
    //--------------------------------------------

    //getting nodes data from file 

    FLNodeStruct* nodesInfo = GetNodesFromFile(nodes_source, numFlNodes);   

    NS_LOG_INFO("Creating nodes.");
    NodeContainer nodes;
    nodes.Create(numFlNodes);

    // TODO once the blockchain nodes application is changed to one more suitable to our sinario
    // we'll replace this one node acting as a server with the BC nodes
    NodeContainer initiator;
    initiator.Create(1);
    nodes.Add(initiator);
    NS_LOG_INFO("Installing WiFi and Internet stack.");
    WifiHelper wifi;
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    YansWifiPhyHelper wifiPhy;
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiPhy.SetChannel(wifiChannel.Create());
    NetDeviceContainer nodeDevices = wifi.Install(wifiPhy, wifiMac, nodes);
    // NetDeviceContainer initiatorDevice = wifi.Install(wifiPhy, wifiMac, initiator);
    
    InternetStackHelper internet;
    internet.Install(nodes);
    // internet.Install(initiator);
    Ipv4AddressHelper ipAddrs;
    ipAddrs.SetBase("192.168.0.0", "255.255.0.0");
    Ipv4InterfaceContainer nodesIpIfaces = ipAddrs.Assign(nodeDevices);
    // ipAddrs.NewAddress();
    // Ipv4InterfaceContainer initiatorIpIfaces = ipAddrs.Assign(initiatorDevice);
    // Ipv4Address initiatorAddr = initiatorIpIfaces.GetAddress(0);

  
    //--------------------------------------------
    //-- Setup physical layout
    //--------------------------------------------
     // Create a position allocator
    Ptr<RandomBoxPositionAllocator> positionAlloc = CreateObject<RandomBoxPositionAllocator> ();
    Ptr<UniformRandomVariable> xVal = CreateObject<UniformRandomVariable>();
    xVal->SetAttribute("Min", DoubleValue(xPosMin));
    xVal->SetAttribute("Max", DoubleValue(xPosMax));
    positionAlloc->SetAttribute("X", PointerValue(xVal));
    Ptr<UniformRandomVariable> yVal = CreateObject<UniformRandomVariable>();
    yVal->SetAttribute("Min", DoubleValue(yPosMin));
    yVal->SetAttribute("Max", DoubleValue(yPosMax));
    positionAlloc->SetAttribute("Y", PointerValue(yVal));
    Ptr<UniformRandomVariable> zVal = CreateObject<UniformRandomVariable>();
    zVal->SetAttribute("Min", DoubleValue(zPosMin));
    zVal->SetAttribute("Max", DoubleValue(zPosMax));
    positionAlloc->SetAttribute("Z", PointerValue(zVal));
    // Set the position allocator for the nodes
    MobilityHelper mobility;
    mobility.SetPositionAllocator (positionAlloc);
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (nodes);
    // mobility.Install (initiator);

    
     //--------------------------------------------
    //-- Create a custom traffic source and sink
    //--------------------------------------------
    NS_LOG_INFO("installing apps.");
    
    Ptr<Node> receiveNode;
    Ptr<Receiver> receive ;
    for (uint32_t i = 1; i < nodes.GetN()-1 ; ++i) {
      receiveNode = nodes.Get(i);
      receive = CreateObject<Receiver>();
      receiveNode->AddApplication(receive);
      receive->SetStartTime(Seconds(0));
      receive->SetStopTime(Seconds(20));
    }
      
     
    // Ptr<Node> appSink = NodeList::GetNode(1);
    // Ptr<Receiver> receiver = CreateObject<Receiver>();
    // appSink->AddApplication(receiver);
     

    Ptr<Node> appSource = nodes.Get(0);
    Ptr<Initiator> flInitTask = CreateObject<Initiator>();
    appSource->AddApplication(flInitTask);
    flInitTask->SetStartTime(Seconds(1));
    flInitTask->setNumNodes(numFlNodes);
    flInitTask->setRounds(flrounds);
    flInitTask->setTargetAcc(targetAccuracy);
    flInitTask->setNumParticipants(numParticipants);
    flInitTask->setNumAggregators(numAggregators);
    flInitTask->setNodesInfo(nodesInfo, numFlNodes);
    Config::Set("/NodeList/*/ApplicationList/*/$Initiator/Destination",
                Ipv4AddressValue("192.168.255.255"));
                
     // Create the animation object and configure for specified output
    AnimationInterface anim(animFile);

    //--------------------------------------------
    //-- Run the simulation
    //--------------------------------------------
    NS_LOG_INFO("Run Simulation.");
    Simulator::Run();
    Simulator::Destroy();

  return 0 ;
}



FLNodeStruct* GetNodesFromFile(const std::string& filename,  int& numNodes){

    FLNodeStruct* nodeList = nullptr;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    // Ignore the header line
    std::string line;
    std::getline(file, line);

    // Read data line by line and create FLNodeStruct nodes
    int count = 0;
      // std::cerr << "file opened "<<numNodes << std::endl;
    while (std::getline(file, line) && count < numNodes) {
        std::stringstream ss(line);
      
        FLNodeStruct node;
        std::string field;

        std::getline(ss, field, ','); // Read the ID field
        node.nodeId = std::stoi(field);

        std::getline(ss, field, ','); // Read the Availability field
        node.availability = (field == "true");

        std::getline(ss, field, ','); // Read the Honesty field
        node.honesty = std::stod(field);

        std::getline(ss, field, ','); // Read the Dataset Size field
        node.datasetSize = std::stoi(field);

        std::getline(ss, field, ','); // Read the Frequency field
        node.freq = std::stoi(field);

        std::getline(ss, field, ','); // Read the Transmission Rate field
        node.transRate = std::stoi(field);

        std::getline(ss, field, ','); // Read the Task field
        node.task = std::stoi(field);

        std::getline(ss, field); // Read the Dropout field
        node.dropout = (field == "true");

        // Resize the array for each new node
        nodeList = (FLNodeStruct*)realloc(nodeList, (count + 1) * sizeof(FLNodeStruct));
        nodeList[count++] = node;
    }

    file.close();
    // std::cerr << "file closed "<<count << std::endl;
    numNodes = count;
    return nodeList;
}
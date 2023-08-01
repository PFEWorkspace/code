#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/netanim-module.h"
#include "ns3/ascii-file.h"
#include "ai-helper.h"
#include "FL-node.h"
#include "FL-task-initiator.h"

#include <string>
#include <ctime>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FLExperimentSimulation");

FLNodeStruct* GetNodesFromFile(const std::string& filename,  int& numNodes);

void TracePacketTx (Ptr<const Packet> packet, const Address &source, const Address &destination)
{
    // Ici, vous pouvez enregistrer les informations de paquet dans un fichier de trace ASCII ou effectuer d'autres actions souhaitées
    // Par exemple, pour enregistrer les informations de paquet dans un fichier de trace ASCII nommé "packet-trace.txt"
    std::ofstream traceFile;
    traceFile.open ("packet-trace.txt", std::ios::app);
    traceFile << Simulator::Now ().GetSeconds () << " " << packet->GetSize () << std::endl;
    traceFile.close ();
}

int
main(int argc, char* argv[])
{

  LogComponentEnable("TaskInitiatorApp",LOG_LEVEL_INFO);
  LogComponentEnable("FLExperimentSimulation",LOG_LEVEL_INFO);
  LogComponentEnable("FLNodeApp", LOG_LEVEL_INFO);
  LogComponentEnable("AiHelper",LOG_LEVEL_INFO);
  Time::SetResolution(Time::NS);
  
    int numFlNodes= 100;
    int numParticipants=50;
    int numAggregators=20;
    std::string nodes_source= "";
    int flrounds = 10;
    int x =0 ;
    double targetAccuracy = 0.0;
    const uint16_t Port = 8833;
    double xPosMin = 0;
    double xPosMax = 30;
    double yPosMin = 0;
    double yPosMax = 30;
    double zPosMin = 0;
    double zPosMax = 0;

    bool tracing= false;




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
    cmd.AddValue("x"," the number of models to aggregate at once",x);
    cmd.Parse (argc, argv);


   
   //--------------------------------------------
    //-- Create nodes and network stacks
    //--------------------------------------------

    //getting nodes data from file 

    FLNodeStruct* nodesInfo = GetNodesFromFile(nodes_source, numFlNodes);   

    NS_LOG_INFO("Creating nodes.");
    NodeContainer nodes;
    nodes.Create(numFlNodes + 1 ); // federated learning nodes + 1 for the initiator ( + the blockchain nodes to add later)

    
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

    
     //--------------------------------------------
    //-- Create a custom traffic source and sink
    //--------------------------------------------
    NS_LOG_INFO("installing apps.");
    
    Ptr<Node> flNode;
    Ptr<FLNode> FL ;
    for (uint32_t i = 1; i < nodes.GetN()-1; ++i) {
      flNode = nodes.Get(i);
      FL = CreateObject<FLNode>();
      flNode->AddApplication(FL);
      // FL->SetStartTime(Seconds(0));
      //setting the caracteristics of the nodes
      FL->Init(nodesInfo[i]);
    }
    
    Ptr<Node> initiator = nodes.Get(0); 
    Ptr<Initiator> flInitTask = CreateObject<Initiator>();
    initiator->AddApplication(flInitTask);
    // flInitTask->SetStartTime(Seconds(1));
    flInitTask->setNumNodes(numFlNodes);
    flInitTask->setRounds(flrounds);
    flInitTask->setTargetAcc(targetAccuracy);
    flInitTask->setNumParticipants(numParticipants);
    flInitTask->setNumAggregators(numAggregators);
    flInitTask->setNodesInfo(nodesInfo, numFlNodes);
    Config::Set("/NodeList/*/ApplicationList/*/$Initiator/Destination",
                Ipv4AddressValue("192.168.255.255"));
    
    //--------------------------------------------------------------
    //---- tracing 
    //-------------------------------------------------------------
    if (tracing == true)
    {
       NS_LOG_INFO("tracing.");
        // AsciiTraceHelper ascii;
        // wifiPhy.EnableAsciiAll(ascii.CreateFileStream("main.tr"));
      
        // wifiPhy.EnablePcap("wifi-simple-adhoc-grid",nodes);
        // Trace routing tables
        // Ptr<OutputStreamWrapper> routingStream =
        //     Create<OutputStreamWrapper>("wifi-simple-adhoc-grid.routes", std::ios::out);
        // Ipv4RoutingHelper::PrintRoutingTableAllEvery(Seconds(2), routingStream);
        // Ptr<OutputStreamWrapper> neighborStream =
        //     Create<OutputStreamWrapper>("wifi-simple-adhoc-grid.neighbors", std::ios::out);
        // Ipv4RoutingHelper::PrintNeighborCacheAllEvery(Seconds(2), neighborStream);

        // To do-- enable an IP-level trace that shows forwarding events only
    }


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
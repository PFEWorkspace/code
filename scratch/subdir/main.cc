#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"

#include "ns3/netanim-module.h"
#include "ns3/ascii-file.h"

#include "BC-node.h"
#include "FL-node.h"
#include "FL-task-initiator.h"
#include "Blockchain.h"


#include <string>
#include <ctime>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FLExperimentSimulation");

FLNodeStruct* GetNodesFromFile(const std::string& filename,  int& numNodes);
// void ReceivePacketTrace ( Ptr<const Packet> packet, double v, ns3::WifiMode mode, ns3::WifiPreamble preamble);
// void TransmittedPacketTrace(Ptr<const Packet> packet,  ns3::WifiMode mode, ns3::WifiPreamble preamble, unsigned char c);

int
main(int argc, char* argv[])
{

  LogComponentEnable("TaskInitiatorApp",LOG_LEVEL_INFO);
  LogComponentEnable("FLExperimentSimulation",LOG_LEVEL_INFO);
  LogComponentEnable("FLNodeApp", LOG_LEVEL_INFO);
  LogComponentEnable("AiHelper",LOG_LEVEL_INFO);
  LogComponentEnable("Blockchain", LOG_LEVEL_INFO);
  LogComponentEnable("BCNodeApp", LOG_LEVEL_INFO);
  Time::SetResolution(Time::NS);

  Time starting_sim = Simulator::Now();
  
  
    int numFlNodes= 100;
    int numBCNodes = 30;
    int numParticipants=50;
    int numAggregators=20;
    std::string nodes_source= "";
    int flrounds = 10;
    int x =0 ;
    double targetAccuracy = 0.0;
    int modelSize=1600;
    double testPartitionSize = 0.2 ;
    // const uint16_t Port = 8833;
    double xPosMin = 0;
    double xPosMax = 30;
    double yPosMin = 0;
    double yPosMax = 30;
    double zPosMin = 0;
    double zPosMax = 0;

    bool tracing= true;




    std::string animFile = "FL-animation.xml";

    //---------------------------------------------
    //----- Command line and getting parameters
    //----------------------------------------------
    CommandLine cmd;
    cmd.AddValue ("numNodes","the total number of nodes", numFlNodes);
    cmd.AddValue ("numBCNodes", "the number of Blockchain nodes in the community", numBCNodes);
    cmd.AddValue ("participantsPerRound","the number of participants per round",numParticipants);
    cmd.AddValue ("aggregatorsPerRound","the number of aggregators per pround",numAggregators);
    cmd.AddValue ("source","the path and name of the file to get initial data about nodes", nodes_source);
    cmd.AddValue ("flRounds", "the number of rounds per FL task", flrounds);
    cmd.AddValue ("targetAccuracy", "the target accuracy for the FL task",targetAccuracy);
    cmd.AddValue ("modelSize","the size of the model to train",modelSize);
    cmd.AddValue ("testPartition", "percentage for the test partition from datasets", testPartitionSize);
    cmd.AddValue("x"," the number of models to aggregate at once",x);
    cmd.Parse (argc, argv);


   
   //--------------------------------------------
    //-- Create nodes and network stacks
    //--------------------------------------------

    //getting nodes data from file 

    FLNodeStruct* nodesInfo = GetNodesFromFile("scratch/subdir/"+nodes_source, numFlNodes);   

    NS_LOG_INFO("Creating nodes.");
    NodeContainer nodes;
    nodes.Create(numFlNodes); // federated learning nodes

    NodeContainer BCnodes;
    BCnodes.Create(numBCNodes);

    NodeContainer initiator;
    initiator.Create(1);

    NS_LOG_INFO("Installing WiFi and Internet stack.");
    
    // WifiHelper wifi;
    // wifi.SetStandard(WIFI_STANDARD_80211n);
    // wifi.SetRemoteStationManager("ns3::IdealWifiManager");

    // WifiMacHelper wifiMac;
    // wifiMac.SetType("ns3::AdhocWifiMac");
    
    // YansWifiPhyHelper wifiPhy;

    // YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    
    // // wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    // // wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
    // //                            "Exponent", DoubleValue(3.0));
    
    // wifiPhy.SetChannel(wifiChannel.Create());
    

    // wifiPhy.Set("TxPowerStart", DoubleValue(20));
    // wifiPhy.Set("TxPowerEnd", DoubleValue(20));
    // wifiPhy.Set("ChannelSettings",StringValue("{155, 80, BAND_5GHZ, 0}"));
    
    // wifiPhy.Set("Antennas", UintegerValue(2));
    // wifiPhy.Set("MaxSupportedTxSpatialStreams", UintegerValue(2));
    // wifiPhy.Set("MaxSupportedRxSpatialStreams", UintegerValue(2));
    
    // StringValue DataRate = StringValue("HtMcs7");
    
    // wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
    //                                  "DataMode",
    //                                  DataRate,
    //                                  "ControlMode",
    //                                  DataRate);
    
    // Config::Set("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/HtConfiguration/"
    //                     "ShortGuardIntervalSupported",
    //                     BooleanValue(false));

     WifiHelper wifi;
  // wifi.SetRemoteStationManager ("ns3::AarfWifiManager");
  wifi.SetStandard(WIFI_STANDARD_80211a);
  wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode",
                               StringValue("OfdmRate24Mbps"));

  YansWifiChannelHelper wifiChannel;
  wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
  wifiChannel.AddPropagationLoss("ns3::ThreeLogDistancePropagationLossModel");
  wifiChannel.AddPropagationLoss("ns3::NakagamiPropagationLossModel");

  YansWifiPhyHelper wifiPhyHelper;
  wifiPhyHelper.SetChannel(wifiChannel.Create());
  wifiPhyHelper.Set("TxPowerStart", DoubleValue(5));
  wifiPhyHelper.Set("TxPowerEnd", DoubleValue(5));

  WifiMacHelper wifiMacHelper;
  wifiMacHelper.SetType("ns3::AdhocWifiMac");
   
    NetDeviceContainer nodeDevices = wifi.Install(wifiPhyHelper, wifiMacHelper, nodes);
    NetDeviceContainer BCnodeDevices= wifi.Install(wifiPhyHelper, wifiMacHelper,  BCnodes);
    
    // wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
    NetDeviceContainer initiatorDevice = wifi.Install(wifiPhyHelper, wifiMacHelper, initiator);

    InternetStackHelper internet;
    internet.Install(nodes);
    internet.Install(BCnodes);
    internet.Install(initiator);
    
    Ipv4AddressHelper ipAddrs;
    ipAddrs.SetBase("192.168.0.0", "255.255.0.0");
    Ipv4InterfaceContainer nodesIpIfaces = ipAddrs.Assign(nodeDevices);
    ipAddrs.Assign(initiatorDevice);
    // NS_LOG_INFO("ip adress "<< nodesIpIfaces.GetAddress(4));
    
    // ipAddrs.SetBase("192.168.1.0","255.255.255.0");
    Ipv4InterfaceContainer BCnodesIpIfaces = ipAddrs.Assign(BCnodeDevices);

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
    mobility.Install(BCnodes);
    mobility.Install(initiator);
    
     //--------------------------------------------
    //-- Create a custom traffic source and sink
    //--------------------------------------------
    NS_LOG_INFO("installing apps.");



    Blockchain* blockchain = Blockchain::getInstance();
    blockchain->SetBCAddressContainer(BCnodesIpIfaces);
    blockchain->SetFLAddressContainer(nodesIpIfaces);
    blockchain->SetTargetAcc(targetAccuracy);
    blockchain->setNumFLNodes(numFlNodes);
    blockchain->setNumBCNodes(numBCNodes);
    blockchain->SetMaxFLRound(flrounds);
    blockchain->setNumAggregators(numAggregators);
    blockchain->setNumTrainers(numParticipants);
    blockchain->SetModelsToAggAtOnce(x);
    blockchain->SetRandomBCStream();
    blockchain->SetModelSize(modelSize);
    blockchain->simulation_start_time = starting_sim ;
  
    for(uint i =0; i< numFlNodes; i++){
      blockchain->AddNodeInfo(nodesInfo[i]);
    }

    Ptr<Node> bcnode;
    Ptr<BCNode> BC ;
    for (uint32_t i = 0; i < BCnodes.GetN(); ++i) {
      bcnode = BCnodes.Get(i);
      BC = CreateObject<BCNode>();
      bcnode->AddApplication(BC);
      BC->SetStartTime(Seconds(0));
      // BC->SetStopTime(Seconds(10));
  
    }


    Ptr<Node> flNode;
    Ptr<FLNode> FL ;
    for (uint32_t i = 0; i < nodes.GetN(); ++i) {
      flNode = nodes.Get(i);
      FL = CreateObject<FLNode>();
      flNode->AddApplication(FL);
      FL->SetStartTime(Seconds(0));
      // FL->SetStopTime(Seconds(10));
      //setting the caracteristics of the nodes
      FL->Init(nodesInfo[i],modelSize,testPartitionSize);
    }

      
    
    Ptr<Node> initnode = initiator.Get(0); 
    Ptr<Initiator> flInitTask = CreateObject<Initiator>();
    initnode->AddApplication(flInitTask);
    flInitTask->SetStartTime(Seconds(1));

    flInitTask->setNumNodes(numFlNodes);
    flInitTask->setRounds(flrounds);
    flInitTask->setTargetAcc(targetAccuracy);
    flInitTask->setNumParticipants(numParticipants);
    flInitTask->setNumAggregators(numAggregators);
    flInitTask->setNodesInfo(nodesInfo, numFlNodes);
    //set the destination adresses
    std::vector<Ipv4Address> adrs;
    adrs.push_back(blockchain->getBCAddress()); 
   
    for(int i=0; i<numFlNodes; i++){
      adrs.push_back(nodesIpIfaces.GetAddress(i));
    }
    flInitTask->setDestAddrs(adrs);

    
    //--------------------------------------------------------------
    //---- tracing 
    //-------------------------------------------------------------
   
    
    if (tracing == true)
    {
      
        // AsciiTraceHelper ascii;
        // wifiPhyHelper.EnableAsciiAll(ascii.CreateFileStream("main.tr"));


        // To do-- enable an IP-level trace that shows forwarding events only
    }
   
    AnimationInterface anim(animFile);
    for(uint32_t i=0; i<BCnodes.GetN();i++){
      anim.UpdateNodeColor(BCnodes.Get(i),0,0,255);
    }
    anim.UpdateNodeColor(initiator.Get(0),0,255,0);
    anim.EnablePacketMetadata (true);
    
    anim.SetMaxPktsPerTraceFile(100000000000);
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
        node.availability = (std::stoi(field) == 1);

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
        node.dropout = (std::stoi(field) == 1);

        std::getline(ss, field); // Read the malicious field
        node.malicious = (std::stoi(field) == 1);

        // Resize the array for each new node
        nodeList = (FLNodeStruct*)realloc(nodeList, (count + 1) * sizeof(FLNodeStruct));
        nodeList[count++] = node;
    }

    file.close();
    // std::cerr << "file closed "<<count << std::endl;
    numNodes = count;
    return nodeList;
}
    

   

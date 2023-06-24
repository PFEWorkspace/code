#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/applications-module.h"
#include "ns3/wifi-module.h"
#include "ns3/netanim-module.h"

#include <string>
#include <ctime>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FLExperimentSimulation");

int
main(int argc, char* argv[])
{

  LogComponentEnable("TaskInitiatorApp",LOG_LEVEL_ALL);
  LogComponentEnable("FLExperimentSimulation",LOG_LEVEL_ALL);
  Time::SetResolution(Time::NS);
  
    int numFlNodes= 100;
    const uint16_t Port = 8833;
    double xPosMin = 0;
    double xPosMax = 300;
    double yPosMin = 0;
    double yPosMax = 300;
    double zPosMin = 0;
    double zPosMax = 0;
    std::string animFile = "FL-animation.xml";

   //--------------------------------------------
    //-- Create nodes and network stacks
    //--------------------------------------------
    NS_LOG_INFO("Creating nodes.");
    NodeContainer nodes;
    nodes.Create(numFlNodes);

    // TODO once the blockchain nodes application is changed to one more suitable to our sinario
    // we'll replace this one node acting as a server with the BC nodes
    NodeContainer initiator;
    initiator.Create(1);

    NS_LOG_INFO("Installing WiFi and Internet stack.");
    WifiHelper wifi;
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    YansWifiPhyHelper wifiPhy;
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiPhy.SetChannel(wifiChannel.Create());
    NetDeviceContainer nodeDevices = wifi.Install(wifiPhy, wifiMac, nodes);
    NetDeviceContainer initiatorDevice = wifi.Install(wifiPhy, wifiMac, initiator);
    
    InternetStackHelper internet;
    internet.Install(nodes);
    internet.Install(initiator);
    Ipv4AddressHelper ipAddrs;
    ipAddrs.SetBase("192.168.0.0", "255.255.0.0");
    Ipv4InterfaceContainer nodesIpIfaces = ipAddrs.Assign(nodeDevices);
    ipAddrs.NewAddress();
    Ipv4InterfaceContainer initiatorIpIfaces = ipAddrs.Assign(initiatorDevice);
    Ipv4Address initiatorAddr = initiatorIpIfaces.GetAddress(0);

  
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
    mobility.Install (initiator);

     //--------------------------------------------
    //-- Create a custom traffic source and sink
    //--------------------------------------------
    NS_LOG_INFO("installing apps.");
    Ptr<Node> appSource = initiator.Get(0);
    Ptr<Initiator> flInitTask = CreateObject<Initiator>();
    appSource->AddApplication(flInitTask);
    

    // Ptr<Node> appSink = NodeList::GetNode(1);
    // Ptr<Receiver> receiver = CreateObject<Receiver>();
    // appSink->AddApplication(receiver);
    // receiver->SetStartTime(Seconds(0));

    // Config::Set("/NodeList/*/ApplicationList/*/$Sender/Destination",
    //             Ipv4AddressValue("192.168.0.2"));

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
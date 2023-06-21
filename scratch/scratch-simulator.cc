#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/stats-module.h"
#include "ns3/wifi-module.h"

#include <ctime>
#include <sstream>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FLSimulation");

int
main(int argc, char* argv[])
{
   //--------------------------------------------
    //-- Create nodes and network stacks
    //--------------------------------------------
    NS_LOG_INFO("Creating nodes.");
    NodeContainer nodes;
    nodes.Create(100);

    NS_LOG_INFO("Installing WiFi and Internet stack.");
    WifiHelper wifi;
    WifiMacHelper wifiMac;
    wifiMac.SetType("ns3::AdhocWifiMac");
    YansWifiPhyHelper wifiPhy;
    YansWifiChannelHelper wifiChannel = YansWifiChannelHelper::Default();
    wifiPhy.SetChannel(wifiChannel.Create());
    NetDeviceContainer nodeDevices = wifi.Install(wifiPhy, wifiMac, nodes);

    InternetStackHelper internet;
    internet.Install(nodes);
    Ipv4AddressHelper ipAddrs;
    ipAddrs.SetBase("192.168.0.0", "255.255.255.0");
    ipAddrs.Assign(nodeDevices);

    //--------------------------------------------
    //-- Setup physical layout
    //--------------------------------------------
     // Create a position allocator
    Ptr<RandomDiscPositionAllocator> positionAlloc = CreateObject<RandomDiscPositionAllocator> ();
    // positionAlloc->SetRho (100.0); // Set the radius of the disc to 100.0

    // Set the position allocator for the nodes
    MobilityHelper mobility;
    mobility.SetPositionAllocator (positionAlloc);
    mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
    mobility.Install (nodes);

  return 0 ;
}
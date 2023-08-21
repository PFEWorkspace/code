#include "FL-task-initiator.h"
#include "FL-node.h"

#include "BC-node.h"



namespace ns3{
    
NS_LOG_COMPONENT_DEFINE("BCNodeApp");

TypeId BCNode::GetTypeId() {
  static TypeId tid = TypeId("BCNode")
                          .SetParent<Application>()
                          .SetGroupName("Applications")
                          .AddConstructor<BCNode>()
                          .AddAttribute("Port", "Listening port", UintegerValue(8833),
                                        MakeUintegerAccessor(&BCNode::m_port),
                                        MakeUintegerChecker<uint32_t>())
                          .AddAttribute("Destination",
                                          "Target host address.",
                                          Ipv4AddressValue(),
                                          MakeIpv4AddressAccessor(&BCNode::m_destAddr),
                                          MakeIpv4AddressChecker());
  return tid;
}

BCNode::BCNode(){
   NS_LOG_FUNCTION_NOARGS();

}

BCNode::~BCNode(){
  NS_LOG_FUNCTION_NOARGS();
}


void BCNode::DoDispose() {
   NS_LOG_FUNCTION_NOARGS();
    m_socket = nullptr;
    // chain up
    Application::DoDispose();
}

void BCNode::StartApplication() {
  NS_LOG_FUNCTION_NOARGS();

    if (!m_socket)
    {
     
        Ptr<SocketFactory> socketFactory =
            GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
        m_socket = socketFactory->CreateSocket();
        InetSocketAddress local = InetSocketAddress(Ipv4Address::GetAny(), m_port);
        m_socket->Bind(local);
    }

    m_socket->SetRecvCallback(MakeCallback(&BCNode::Receive, this));

}

void BCNode::StopApplication() {
  NS_LOG_FUNCTION_NOARGS();
}

void BCNode::Receive(Ptr<Socket> socket) {
    
    Ptr<Packet> packet;
    Address from;
  
    while ((packet = socket->RecvFrom(from)))
    {
        unsigned char *packetInfo = new unsigned char[packet->GetSize()];
       
        if (InetSocketAddress::IsMatchingType(from))
        {
            packet->CopyData(packetInfo, packet->GetSize ());
            // NS_LOG_INFO("I'm a bc node id: "<< GetNode()->GetId() << " received " << packet->GetSize() << " bytes from "
            //                         << InetSocketAddress::ConvertFrom(from).GetIpv4()
            //                         << " content: "<< packetInfo) ;
            
            std::string data(reinterpret_cast<char*>(packetInfo), packet->GetSize()) ;
            rapidjson::Document d;
            Blockchain* bc = Blockchain::getInstance();
           MLModel model;
            if(ParseJSON(data,d)){
                   
                if(d.HasMember("message_type") && d["message_type"].IsInt()){
                    switch (d["message_type"].GetInt())
                    {
                        case NEWTASK : //NEWTASK
                         NS_LOG_INFO("I'm a bc node id: "<< GetNode()->GetId() << " received " << packet->GetSize() << " bytes from "
                                    << InetSocketAddress::ConvertFrom(from).GetIpv4()
                                    << " content: "<< packetInfo) ;
            
                        if(d.HasMember("task_id") && d["task_id"].IsInt()){
                            bc->setTaskId( d["task_id"].GetInt());
                            bc->SetActualFLRound(0);
                            bc->SetCurrentBlockId(); 
                        }
                        // the max delay for receiving candidature is: size_message=120 / smallest_trans_rate=150 + 2 seconds for
                        Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(120/150 + 2), [this]() { Selection();});
                        break;
                        case CANDIDATURE : 
                        // receive les candidatures and treat them
                        TreatCandidature(d);
                        break;
                        case MODEL: //MODEL
                             NS_LOG_INFO("I'm a bc node id: "<< GetNode()->GetId() << " received " << packet->GetSize() << " bytes from "
                                    << InetSocketAddress::ConvertFrom(from).GetIpv4()
                                    << " content: "<< packetInfo) ;
            
                            model = DocToMLModel(d);
                            if(model.modelId!=0) TreatModel(model,InetSocketAddress::ConvertFrom(from).GetIpv4(),false);
                            break;                    
                        default:
                        // NS_LOG_INFO("default");
                        break;
                    }
                
                }
                // saving the transaction in the blockchain
                bc->WriteTransaction(bc->getCurrentBlockId(),GetNode()->GetId(),d);
            }
        }

    }
}

void BCNode::Send(rapidjson::Document &d, Ipv4Address adrs) {
 
 if (!m_socket){
    Ptr<SocketFactory> socketFactory = GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
    m_socket = socketFactory->CreateSocket();
    m_socket->Bind();
 }
    rapidjson::StringBuffer packetInfo;
    rapidjson::Writer<rapidjson::StringBuffer> writer(packetInfo);
    d.Accept(writer);

    Ptr<Packet> packet = Create<Packet>(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize());
    m_socket->SendTo(packet,0,InetSocketAddress(adrs, m_port));
    
}


void BCNode::SendTo( rapidjson::Document &d, std::vector<Ipv4Address> &addresses) {
    if (!m_socket) {
        Ptr<SocketFactory> socketFactory = GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
        m_socket = socketFactory->CreateSocket();
        m_socket->Bind();
    }
    
    rapidjson::StringBuffer packetInfo;
    rapidjson::Writer<rapidjson::StringBuffer> writer(packetInfo);
    d.Accept(writer);

    Ptr<Packet> packet = Create<Packet>(reinterpret_cast<const uint8_t*>(packetInfo.GetString()), packetInfo.GetSize());
   
    for (Ipv4Address destAddr : addresses) {
       m_socket->SendTo(packet, 0, InetSocketAddress(destAddr, m_port));
        // NS_LOG_INFO("sent traininig task to "<< destAddr);
    }
}


void
BCNode::SendBroadCast(rapidjson::Document &d, Ipv4Address adrs) {
 NS_LOG_INFO("inside send broadcast");
 if (!m_socket){
    Ptr<SocketFactory> socketFactory = GetNode()->GetObject<SocketFactory>(UdpSocketFactory::GetTypeId());
    m_socket = socketFactory->CreateSocket();
    m_socket->SetAllowBroadcast(true);
    m_socket->Bind();
 }
    rapidjson::StringBuffer packetInfo;
    rapidjson::Writer<rapidjson::StringBuffer> writer(packetInfo);
    d.Accept(writer);

    Ptr<Packet> packet = Create<Packet>(reinterpret_cast<const uint8_t*>(packetInfo.GetString()),packetInfo.GetSize());
    m_socket->SendTo(packet,0,InetSocketAddress(adrs, m_port));
    
}


FLNodeStruct 
BCNode::docToFLNodeStruct(rapidjson::Document &d){
    FLNodeStruct node = FLNodeStruct();
    if(d.HasMember("nodeId") && d["nodeId"].IsInt()){
        node.nodeId = d["nodeId"].GetInt();
    }
    if(d.HasMember("availability") && d["availability"].IsBool()){
        node.availability = d["availability"].GetBool();
    }
    if(d.HasMember("honesty") && d["honesty"].IsDouble()){
        node.honesty = d["honesty"].GetDouble();
    }
    if(d.HasMember("datasetSize")&& d["datasetSize"].IsInt()){
        node.datasetSize = d["datasetSize"].GetInt();
    }
    if(d.HasMember("freq") && d["freq"].IsDouble()){
        node.freq = (int)d["freq"].GetDouble();
    }
    if(d.HasMember("transRate") && d["transRate"].IsDouble()){
        node.transRate = (int)d["transRate"].GetDouble();
    }
    if(d.HasMember("task") && d["task"].IsInt()){
        node.task = d["task"].GetInt();
    }
    return node;
}

void
BCNode::TreatCandidature(rapidjson::Document &d){
    
    Blockchain* bc = Blockchain::getInstance();
    //save the candidature in the blockchain
    bc->AddNodeInfo( docToFLNodeStruct(d));
    FLNodeStruct node = bc->GetNodeInfo(bc->GetReceivedCandidatures());
    bc->IncReceivedCandidatures(); 
   
}

void
BCNode::Selection(){
    //selecting nodes
    Blockchain* bc = Blockchain::getInstance();
    NS_LOG_INFO("received " << bc->GetReceivedCandidatures() << " on " << bc->getNumFLNodes() << " ,starting the selection");
    AiHelper* ai = AiHelper::getInstance();
    ai->Selection(); //the selected nodes are set in the bc
    // write the selection message 
    rapidjson::Document d;
    rapidjson::Value value;
    d.SetObject(); 
    value = SELECTION;
    d.AddMember("message_type", value, d.GetAllocator());
    value = TRAIN;
    d.AddMember("task", value, d.GetAllocator());
    // get selected nodes addresses
    std::vector<Ipv4Address> adrs;
    int numSelectedTrainers = bc->getNumTrainers();
    int id;
    for(int i=0; i < numSelectedTrainers; i++){
        id =bc->getTrainer(i);
        adrs.push_back(bc->getFLAddress(id));
    } 
        
    //sending the message to all selected trainers
    SendTo(d,adrs);
    bc->StartTraining();
}

MLModel BCNode::DocToMLModel(rapidjson::Document &d) {
    MLModel model = MLModel();

    if (d.HasMember("modelId") && d["modelId"].IsInt()) {
        model.modelId = d["modelId"].GetInt();
    }

    if (d.HasMember("nodeId") && d["nodeId"].IsInt()) {
        model.nodeId = d["nodeId"].GetInt();
    }

    if (d.HasMember("taskId") && d["taskId"].IsInt()) {
        model.taskId = d["taskId"].GetInt();
    }

    if (d.HasMember("round") && d["round"].IsInt()) {
        model.round = d["round"].GetInt();
    }

    if (d.HasMember("type") && d["type"].IsInt()) {
        model.type = d["type"].GetInt();
    }

    if (d.HasMember("positiveVote") && d["positiveVote"].IsInt()) {
        model.positiveVote = d["positiveVote"].GetInt();
    }

    if (d.HasMember("negativeVote") && d["negativeVote"].IsInt()) {
        model.negativeVote = d["negativeVote"].GetInt();
    }

    if (d.HasMember("evaluator1") && d["evaluator1"].IsInt()) {
        model.evaluator1 = d["evaluator1"].GetInt();
    }

    if (d.HasMember("evaluator2") && d["evaluator2"].IsInt()) {
        model.evaluator2 = d["evaluator2"].GetInt();
    }

    if (d.HasMember("evaluator3") && d["evaluator3"].IsInt()) {
        model.evaluator3 = d["evaluator3"].GetInt();
    }

    if (d.HasMember("aggregated") && d["aggregated"].IsBool()) {
        model.aggregated = d["aggregated"].GetBool();
    }

    if (d.HasMember("aggModelId") && d["aggModelId"].IsInt()) {
        model.aggModelId = d["aggModelId"].GetInt();
    }

    if (d.HasMember("accuracy") && d["accuracy"].IsDouble()) {
        model.accuracy = d["accuracy"].GetDouble();
    }

    if (d.HasMember("acc1") && d["acc1"].IsDouble()) {
        model.acc1 = d["acc1"].GetDouble();
    }

    if (d.HasMember("acc2") && d["acc2"].IsDouble()) {
        model.acc2 = d["acc2"].GetDouble();
    }

    if (d.HasMember("acc3") && d["acc3"].IsDouble()) {
        model.acc3 = d["acc3"].GetDouble();
    }

    return model;
}

void
BCNode::TreatModel(MLModel model, Ipv4Address source, bool reschedule){
    /* switch model type:
        case type 0 (local):
            if next step is evaluation:               
                if it has been evaluated before (came from evaluator and not a trainer) :
                    if previous task found in tasks list: (if not found means this part have been done by another node or after a reschedule)
                        select a new aggregator not busy
                        if no available node reschedule this methode 
                        sendTask(evaluation, model, aggregatorid, aggregatoradrs)
                        take it off
                    
                else if it is its first evaluation:
                    select an aggregator that is not already busy and sendTask(...)                        
            else : (done with evaluations next aggregation)
                if model valid (positiveVote > negativeVote)
                    save it to models to aggregate
                    Aggregation(model)
        case type 1 (intermediaire):
            if it came from an aggregatation task and not en evaluation:
                save it to list of models to aggregate
                aggregation(model)
            select another aggregator
            if dispo send evaluation task (its less prioritized, if enough resource do it else no need)
        case type 2 (global):
            newRound
    */ 
    
    Blockchain* bc = Blockchain::getInstance();
    int aggId;
    std::vector<MLModel> models;
    int sourceId= bc->getFLNodeId(source);
    // NS_LOG_INFO("treat model");
    if(bc->hasPreviousTask(sourceId) && not reschedule){
        bc->RemoveTask(sourceId);
        // NS_LOG_INFO("task removed");
    }
    NS_LOG_INFO("model type "<< model.type << " positive vote "<< model.positiveVote<< " negative vote "<< model.negativeVote);
    
    switch(model.type){
    case LOCAL:
    //  NS_LOG_INFO("it's local model");
        if(model.positiveVote+model.negativeVote < 2 || model.positiveVote - model.negativeVote == 0){
                aggId = bc->GetAggregatorNotBusy(model.evaluator1, model.evaluator2);
                // NS_LOG_INFO("node "<<aggId<<" will be evaluating the model "<< model.modelId);
                if(aggId == -1){ //no available nodes
                    // NS_LOG_INFO("reschedule treatmodel");
                    Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(10),[this, model,source](){TreatModel(model,source,true);});
                    break;
                }
                else{
                    Evaluation(model,aggId);                   
                }
        }else{ //done with evaluation next is aggregation
            if(model.positiveVote>model.negativeVote){ 
                if(!reschedule){
                    bc->AddModelToAgg(model);
                }
                // if(bc->GetModelsToAggAtOnce()<bc->getModelsToAggSize() ){ //les aggregation local parmi les premier, certain it's not a global agg
                //     aggId = bc->GetAggregatorNotBusy(-1,-1); //doesn't matter if the aggregation is done by one of the evaluators
                //     if(aggId == -1){ //no available nodes
                    
                //         Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(10),[this, model,source](){TreatModel(model,source,true);});
                //         break;
                //     }else{
                //         models =bc->getxModelsToAgg(bc->GetModelsToAggAtOnce());
                //         // NS_LOG_INFO("models to agg size "<< models.size() << " first: "<< models[0].modelId << " second "<<models[1].modelId << " third "<<models[2].modelId);
                //         Aggregation(models,aggId,INTERMEDIAIRE);
                //     }                          
               
                aggId = bc->GetAggregatorNotBusy(-1,-1);
                if(aggId == -1){ //no available nodes
                    Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(10),[this, model,source](){TreatModel(model,source, true);});
                    break;
                }else{  
                    // NS_LOG_INFO("max delay "<<bc->MaxDelayPassed());  
                    if(bc->GetModelsToAggAtOnce()<= bc->getModelsToAggSize() && (bc->getNumAggTasksAwaiting()>0 || (bc->getNumAggTasksAwaiting()==0 && bc->firstagg))){
                            if(bc->firstagg){bc->firstagg=false;}
                            models =bc->getxModelsToAgg(bc->GetModelsToAggAtOnce());
                             NS_LOG_INFO("AGGREGATION INTERMEDIAIRE");
                            Aggregation(models,aggId,INTERMEDIAIRE);
                    }else{
                        // NS_LOG_INFO("models size" <<bc->getModelsToAggSize() << " tasks awaiting "<< bc->getNumAggTasksAwaiting());
                        if(bc->getModelsToAggSize()>0 && bc->GetModelsToAggAtOnce()>= bc->getModelsToAggSize() && bc->getNumAggTasksAwaiting()==0 && bc->MaxDelayPassed() && !bc->lastagg){
                                models =bc->getxModelsToAgg(bc->GetModelsToAggAtOnce());
                                NS_LOG_INFO("AGGREGATION GLOBAL ");
                                Aggregation(models,aggId,GLOBAL);
                        }else if(bc->getModelsToAggSize()+ bc->getNumAggTasksAwaiting()>0 && !bc->lastagg){
                             NS_LOG_INFO("reschedule local");
                            Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(bc->GetStillDelay()),[this, model,source](){TreatModel(model,source, true);});
                        }
                    }
                }    
            }    
        }
    break;
    case INTERMEDIAIRE:   
        if(model.positiveVote+model.negativeVote==0){//not evaluated yet and came from an aggregation task
            if(!reschedule){
                bc->AddModelToAgg(model);
            }
            aggId = bc->GetAggregatorNotBusy(-1,-1);
            if(aggId == -1){ //no available nodes
                Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(10),[this, model,source](){TreatModel(model,source, true);});
                break;
            }else{    
                if(bc->GetModelsToAggAtOnce()<=bc->getModelsToAggSize() && bc->getNumAggTasksAwaiting()>0){
                        models =bc->getxModelsToAgg(bc->GetModelsToAggAtOnce());
                        // NS_LOG_INFO("models to agg size "<< models.size() << " first: "<< models[0].modelId);
                        NS_LOG_INFO("AGGREGATION INTERMEDIAIRE");
                        Aggregation(models,aggId,INTERMEDIAIRE);
                }else {
                    NS_LOG_INFO("models size " <<bc->getModelsToAggSize() << " tasks awaiting "<< bc->getNumAggTasksAwaiting() <<" "<< bc->MaxDelayPassed() << bc->lastagg);

                    if(bc->getModelsToAggSize()>0 && bc->GetModelsToAggAtOnce()>= bc->getModelsToAggSize() && bc->getNumAggTasksAwaiting()==0 && bc->MaxDelayPassed() && !bc->lastagg){
                        models =bc->getxModelsToAgg(bc->GetModelsToAggAtOnce());
                        NS_LOG_INFO("AGGREGATION GLOBAL");
                        Aggregation(models,aggId,GLOBAL);
                    }else {
                        if(bc->getModelsToAggSize()+ bc->getNumAggTasksAwaiting()>0 && !bc->lastagg){
                        NS_LOG_INFO("reschedule inter");
                        Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(bc->GetStillDelay()+10),[this, model,source](){TreatModel(model,source, true);});
                        }
                    }
                }
                
                   
            } 
            // send it for evaluation if there's some nodes available
            if(bc->evaluate()){
                aggId = bc->GetAggregatorNotBusy(model.evaluator1, model.evaluator2);
                if(aggId != -1){ //no available nodes
                    //  Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(2), [this, model, aggId](){ Evaluation(model,aggId);});                   
                    Evaluation(model,aggId);
                }
            }

        }else{
         if(model.negativeVote == 1){ //evaluated once and they found it was wrong, evaluated again 
            aggId = bc->GetAggregatorNotBusy(model.evaluator1,model.evaluator2);

            if(aggId == -1){ //no available nodes
                Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(10),[this, model,source](){TreatModel(model,source, true);});
                break;           
            }else Evaluation(model,aggId);   
        }  //else : it's either evaluated once and its true or twice and the decision is 2vs1 (2 evals + the actual model)
        }
    break;
    case GLOBAL:
    NewRound(model);
    //    Simulator::ScheduleWithContext(GetNode()->GetId(), Seconds(30),[this,model](){ NewRound(model);});
    break;
    default:
    break;
 }

   

}

void
BCNode::Evaluation(MLModel model, int nodeId){
    // save task in tasks         
    // send a new task  
    //schedule few seconds later to check if this task was done (still in list or not) if still in list resend the message,
    Blockchain* bc = Blockchain::getInstance();
    AggregatorsTasks task = AggregatorsTasks();
    task.nodeId = nodeId;
    task.task = EVALUATE;
    task.models[0] = model;
    bc->AddTask(task);
    
    rapidjson::Document d;
    rapidjson::Value value;
    d.SetObject(); 
    value = EVALUATION;
    d.AddMember("message_type", value, d.GetAllocator());
    value = model.modelId ;
    d.AddMember("modelId", value, d.GetAllocator());
    value = model.nodeId;
    d.AddMember("nodeId", value, d.GetAllocator());
    value = model.taskId ;
    d.AddMember("taskId", value, d.GetAllocator());
    value = model.round ;
    d.AddMember("round", value, d.GetAllocator());
    value = model.type; //LOCAL 
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

    Ipv4Address nodeAdrs = bc->getFLAddress(nodeId);
    Send(d,nodeAdrs);
    //schedule a check if task is done or not (detect dropouts)
    Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(50),[this,task](){DetectDropOut(task);});

}
    
void
BCNode::Aggregation(std::vector<MLModel> models, int nodeId, int type){
    // NS_LOG_INFO("Aggregation in BCNODE "<<type);
    Blockchain* bc = Blockchain::getInstance();
    AggregatorsTasks task = AggregatorsTasks();
    if(type==GLOBAL) {
        bc->lastagg = true;
    } else{
        task.nodeId = nodeId;
        task.task = AGGREGATE;
        for(uint i=0;i<models.size();i++){
            task.models[i] = models[i];
        }    
        bc->AddTask(task);
    }   
    
    
    rapidjson::Document d;
    rapidjson::Value value;
    rapidjson::Document::AllocatorType& allocator = d.GetAllocator();
    d.SetObject(); 
    value = AGGREGATION;
    d.AddMember("message_type", value, allocator);
    value = type ;
    d.AddMember("aggregation_type",value,allocator);
    
    rapidjson::Value jsonModels(rapidjson::kArrayType);
    
    for(MLModel model : models){
        rapidjson::Value m(rapidjson::kObjectType);
        value = model.modelId ;
        m.AddMember("modelId", value, allocator);
        value = model.nodeId;
        m.AddMember("nodeId", value, allocator);
        value = model.taskId ;
        m.AddMember("taskId", value, allocator);
        value = model.round ;
        m.AddMember("round", value, allocator);
        value = model.type; //LOCAL 
        m.AddMember("type", value, allocator);
        value = model.positiveVote;
        m.AddMember("positiveVote", value, allocator);
        value = model.negativeVote ;
        m.AddMember("negativeVote", value, allocator);
        value = model.evaluator1 ;
        m.AddMember("evaluator1", value, allocator);
        value = model.evaluator2 ;
        m.AddMember("evaluator2", value, allocator);
        value = model.evaluator3 ;
        m.AddMember("evaluator3", value, allocator);
        value = model.aggregated ;
        m.AddMember("aggregated", value, allocator);
        value = model.aggModelId ;
        m.AddMember("aggModelId", value, allocator);
        value = model.accuracy ; 
        m.AddMember("accuracy", value, allocator);
        value = model.acc1;
        m.AddMember("acc1", value, allocator);
        value = model.acc2;
        m.AddMember("acc2", value, allocator);
        value = model.acc3;
        m.AddMember("acc3", value, allocator);

        jsonModels.PushBack(m,allocator);
    }
    d.AddMember("models",jsonModels,allocator);
    Ipv4Address nodeAdrs = bc->getFLAddress(nodeId);
    Send(d,nodeAdrs);
     //schedule a check if task is done or not (detect dropouts)
    Simulator::ScheduleWithContext(GetNode()->GetId(),Seconds(50),[this,task](){DetectDropOut(task);});
}

void BCNode::NewRound(MLModel globalModel){
    //reset blockchain
    Blockchain* bc = Blockchain::getInstance();
    bool finished = bc->ResetRound(globalModel);
    //reset Aihelper and reset pythonside (generating new env: update node's infos)
    AiHelper* ai = AiHelper::getInstance();
    ai->ResetRound();
    if( not finished ){
       //send the newround message
        rapidjson::Document d;
        rapidjson::Value value;
        d.SetObject(); 
        value = NEWROUND;
        d.AddMember("message_type", value, d.GetAllocator());
    
        std::vector<Ipv4Address> adrs;
   
        for(int i=0; i<bc->getNumFLNodes(); i++){
        adrs.push_back(bc->getFLAddress(i));
        }
        SendTo(d,adrs);
        Simulator::ScheduleWithContext(GetNode()->GetId(), Seconds(120/150 + 2),[this](){ Selection();});
    }else{
        NS_LOG_INFO("******** FL Task done**************");
    }

    

}

void BCNode::DetectDropOut(AggregatorsTasks task){

}
}

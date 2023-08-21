import os 
import torch as T
import torch.nn.functional as F
import numpy as np
from .node_selection_buffer import ReplayBuffer
from .node_selection_networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent ():
    def __init__(self,env,alpha=0.003,beta=0.0003,input_shape=[8],gamma = 0.99,n_actions=2,max_size=1000000,tau=0.005,
    layer1_size=256,layer2_size=256,batch_size=256,reward_scale=2):
        os.makedirs('tmp/sac', exist_ok=True)
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_shape=input_shape, n_actions=n_actions)
        # print ("replay buffer passed")
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.scale = reward_scale
        # self.actor = ActorNetwork(alpha,input_shape,n_actions=n_actions,name='actor',max_actions=env.action_space.high) #type: ignore
        
        self.actor = ActorNetwork(alpha, input_shape,  n_actions=n_actions, name='actor', max_actions=env.action_space.high.shape[0])
        # print ("actor passed")
        self.critic_1 = CriticNetwork(beta, input_shape , n_actions=n_actions, name='critic_1')
        # print ("critic 1 passed")
        self.critic_2 = CriticNetwork(beta, input_shape, n_actions=n_actions, name='critic_2')
        # print ("critic2 passed")
        self.value = ValueNetwork(beta, input_shape, name='value')
        # print ("value passed")
        self.target_value = ValueNetwork(beta, input_shape, name='target_value')
        # print ("target value passed")
        self.update_network_parameters(tau=1)

    def get_selected(self ,input_list):
        indices_of_ones = [index for index, value in enumerate(input_list) if value == 1.0]
        return indices_of_ones
    def choose_action(self,observation):
        # print("in choose action printing observation",observation)
        state = T.tensor(observation,dtype=T.float).to(self.actor.device)
        print("state in choose action", state.shape)
        actions = self.actor.sample_normal(state,self.n_actions)
        print("got the action passe sample_normal :",actions)

        return actions

    def remember(self,state,action,reward,next_state,done):
        # print("in remember checking state" , state)
        self.memory.store_transition(state,action,reward,next_state,done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()
        
        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()


    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state,action,reward,new_state,done = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward,dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state,dtype=T.float).to(self.actor.device)
        state = T.tensor(state,dtype=T.float).to(self.actor.device)
        action = T.tensor(action,dtype=T.float).to(self.actor.device)

        value = self.value(state).view(-1)
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state,self.n_actions)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state,actions)
        q2_new_policy = self.critic_2.forward(state,actions)
        critic_value = T.min(q1_new_policy,q2_new_policy) # to set more stability and get rid of over-estimating bias
        critic_value = critic_value.view(-1)

        self.value_optimizer.zero_grad() # type: ignore
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value,value_target)
        value_loss.backward (retain_graph=True)
        self.value_optimizer.step() # type: ignore 

        actions ,log_probs = self.actor.sample_normal (state,self.n_actions)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state,actions)
        q2_new_policy = self.critic_2.forward(state,actions)
        critic_value = T.min(q1_new_policy,q2_new_policy)
        critic_value = critic_value.view(-1)    

        actor_loss = log_probs - critic_value   
        actor_loss = T.mean(actor_loss) 
        self.actor_optimizer.zero_grad()     # type: ignore
        actor_loss.backward(retain_graph=True)  
        self.actor_optimizer.step()  # type: ignore


        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_
        q1_old_policy = self.critic_1.forward(state,action).view(-1)
        q2_old_policy = self.critic_2.forward(state,action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy,q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy,q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()  
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()  

        self.update_network_parameters()
        

        


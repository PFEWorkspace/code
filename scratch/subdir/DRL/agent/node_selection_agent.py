import os
from sqlite3 import SQLITE_CREATE_INDEX 
import torch as T
import torch.nn.functional as F
import numpy as np
from .node_selection_buffer import ReplayBuffer
from .node_selection_networks import ActorNetwork, CriticNetwork, ValueNetwork
import drl_utils as dr

ALPHA_INITIAL = 1.
DISCOUNT_RATE = 0.99
LEARNING_RATE = 10 ** -4
SOFT_UPDATE_INTERPOLATION_FACTOR = 0.01
class Agent ():
    def __init__(self,env,alpha=ALPHA_INITIAL,beta=LEARNING_RATE,input_shape=[8],gamma = DISCOUNT_RATE,n_actions=2,max_actions=1,max_size=200,tau=SOFT_UPDATE_INTERPOLATION_FACTOR,
    layer1_size=256,layer2_size=256,batch_size=1,reward_scale=2):
        os.makedirs('./tmp/sac', exist_ok=True)
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_shape=input_shape, n_actions=n_actions,max_action=max_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_actions= max_actions
        self.scale = reward_scale

        self.actor = ActorNetwork(alpha, input_shape,  n_actions=n_actions, name='actor', max_actions=self.max_actions)

        self.critic_1 = CriticNetwork(beta, input_shape , n_actions=max_actions, name='critic_1')

        self.critic_2 = CriticNetwork(beta, input_shape, n_actions=max_actions, name='critic_2')

        self.value = ValueNetwork(beta, input_shape, name='value')

        self.target_value = ValueNetwork(beta, input_shape, name='target_value')

        self.update_network_parameters(tau=0.6)

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)
        actions, log_probs = self.actor.sample_normal(state, self.max_actions)
        return actions

    def remember(self,state,action,reward,next_state,done):
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
            print('not enough memories to learn from!')
            return
        print("in learn")
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.actor.device)
        state = T.tensor(state, dtype=T.float).to(self.actor.device)
        action = T.tensor(action, dtype=T.float).to(self.actor.device)
        flat_state = dr.flatten_nodes(state)
        flat_state_= dr.flatten_nodes(state_)


        value = self.value(flat_state).view(-1)
        value_ = self.target_value(flat_state_).view(-1)
        value_[done] = 0.0
        print("going into sample normal in learn")
        actions, log_probs = self.actor.sample_normal(state, self.max_actions)
        # actions = actions[:self.max_actions]
        actions = T.tensor(actions)
        print("going into forward critic")
        q1_new_policy = self.critic_1.forward(flat_state, actions)
        q2_new_policy = self.critic_2.forward(flat_state, actions)
        print("problem with critic forward")
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)
        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(flat_state, actions)
        q2_new_policy = self.critic_2.forward(flat_state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(flat_state, action).view(-1)
        q2_old_policy = self.critic_2.forward(flat_state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()
        self.save_models()
        print('updated the networks')
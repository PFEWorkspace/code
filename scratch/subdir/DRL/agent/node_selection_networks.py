from collections import OrderedDict
import os
import select
import torch as T
from torch.distributions.utils import logits_to_probs
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
import drl_utils as dr

class CriticNetwork(nn.Module):
    #beta learning rate, number of input dimensions from the environment
    #
    def __init__ (self,beta,input_shape,n_actions,
    fc1_dims=256 , fc2_dims = 256, name='critic',chkpt_dir='tmp/sac' ):
        super(CriticNetwork,self).__init__()
        self.input_dims = np.prod(input_shape)
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.name+'_sac')
        #layer 1
        self.fc1 = nn.Linear(self.input_dims+n_actions,self.fc1_dims)
        #layer 2
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        #layer 3
        self.q = nn.Linear(self.fc2_dims,1)
        #optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=beta)
        #device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #to device
        self.to(self.device)

    def forward(self,state,action):
        #layer 1
        
        input = T.cat([state,action],dim=0)
        action_value = self.fc1(input)
        action_value = F.relu(action_value)
        #layer 2
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)
        #layer 3
        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))

class ValueNetwork(nn.Module):
    def __init__(self,beta,input_shape,fc1_dims=256,fc2_dims=256,name='value',chkpt_dir='tmp/sac'):
        super(ValueNetwork,self).__init__()
        # print("in init value networks")
        self.input_dims = np.prod(input_shape)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.name+'_sac')
        #layer 1
        # print("value network input dimensions", self.input_dims)
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        #layer 2
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        #layer 3
        self.v = nn.Linear(self.fc2_dims,1)
        #optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=beta)
        #device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #to device
        self.to(self.device)

    def forward(self,state):
        #layer 1
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        #layer 2
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)
        #layer 3
        v = self.v(state_value)

        return v

    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self,alpha,input_shape,max_actions,fc1_dims=256,fc2_dims=256,n_actions=2,name='actor',chkpt_dir='tmp/sac'):
        super(ActorNetwork,self).__init__()
        self.input_dims = np.prod(input_shape)
        self.input_shape = input_shape
        self.max_actions = max_actions # number of selected nodes
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.name+'_sac')
        # print("checpoint file", self.checkpoint_file)
        self.reparam_noise = 1e-6
        #layer 1
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        # self.fc1 = nn.Linear(*self.input_dims,out_features=self.fc1_dims)
        #layer 2
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        #layer 3


        self.mu = nn.Linear(self.fc2_dims,self.n_actions)
        #layer 4
        self.sigma = nn.Linear(self.fc2_dims,self.n_actions)
        #optimizer
        self.optimizer = optim.Adam(self.parameters(),lr=alpha)
        #device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #to device
        self.to(self.device)
        #normal distribution
        # self.distribution = Normal
    # ... (previous code) ...

    def sample_normal(self, state, num_selected_nodes, exploration_noise= 0.6):
        action_probs_nn, action_mean, action_log_std = self.forward(state)
        print("action probs in sample normal", action_probs_nn)

        action_std = action_log_std.exp()
        print("action std after exp()", action_std)

        state = dr.array_to_state(state, 8)
        print("state after array_to_state", state)

        availability_mask = state[:, 1] != 0
        availability_mask = availability_mask.long()
        print("availability mask", availability_mask)

        # Add exploration noise to logits
        noisy_logits = action_mean + exploration_noise * action_std * T.randn_like(action_mean)

        # Convert action_probs to a clean tensor with NaN replaced by 0
        

        non_nan = np.nan_to_num(T.transpose(action_probs_nn, 0, -1).detach().numpy(), nan=0.0)
        action_probs_nn_clean = T.tensor(non_nan)
        action_probs_nn_clean *= availability_mask
        print("action_probs_clean after nan_to_num and availability mask", action_probs_nn_clean)


        action_dist = Categorical(action_probs_nn_clean)
        print("action_dist after categorical", action_dist)

        # Sample actions from the Categorical distribution
        
        sampled_actions = action_dist.sample(sample_shape=(num_selected_nodes,))
        print("sampled_actions from distribution", sampled_actions)
        log_probs = action_dist.log_prob(sampled_actions)
    
       
        print("final selected_indices", sampled_actions)

        return sampled_actions, log_probs



    def forward(self, state):
      # Layer 1
      prob = self.fc1(state)
      prob = F.relu(prob)
      prob = self.fc2(prob)
      prob = F.relu(prob)

      # Layer 3
      mu = self.mu(prob)
      sigma = self.sigma(prob)

      # Apply availability mask
      state = dr.array_to_state(state, 8)
      availability_mask = state[:, 1] != 0
      availability_mask = availability_mask.float()

      # Apply ReLU activation to mu to ensure non-negative values
      mu = F.relu(mu)

      # Scale the mu values to the range [0, 1]
      scaled_actions = mu / self.input_shape[0]

      # Calculate the probability of not selecting the node
      prob_not_selected = 1 - scaled_actions

      # Normalize the probabilities
      total_probabilities = prob_not_selected + scaled_actions
      selection_probabilities = scaled_actions / total_probabilities  # Only the probability of selecting the node

      return selection_probabilities, mu, sigma





    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))


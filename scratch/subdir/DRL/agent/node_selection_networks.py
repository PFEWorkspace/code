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
        action_value = self.fc1(T.cat([state,action],dim=1))
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
        self.input_dims = np.prod(input_shape)

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.name+'_sac')
        #layer 1
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
        print("checpoint file", self.checkpoint_file)
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
    
    def sample_normal(self, state, num_selected_nodes, exploration_noise=0.025):
        # print("in sample_normal")
        action_probs, action_mean, action_log_std = self.forward(state)
        action_std = action_log_std.exp()
        state = dr.array_to_state(state, 8)
        # print("after array of array ",state)
        availability_mask = state[:, 1] != 0
        availability_mask = availability_mask.long()
        # print("availability mask", availability_mask)
        # Add exploration noise to logits
        noisy_logits = action_mean + exploration_noise * action_std * T.randn_like(action_mean)

        # Create a Categorical distribution using the noisy logits
        action_dist = Categorical(action_probs)
        
        # Sample actions from the Categorical distribution
        sampled_actions = action_dist.sample()

        # Apply availability mask
        sampled_actions = sampled_actions * availability_mask


        # Get indices of selected nodes
        selected_indices = sampled_actions.nonzero().squeeze()
        for i in range(10):
            new_samples = action_dist.sample()
            new_samples = new_samples*availability_mask  # Apply availability mask
            new_indices = new_samples.nonzero().squeeze()

            # Convert the selected_indices list back to a tensor
            selected_indices_tensor = T.tensor(selected_indices)

            # Combine the selected indices and new indices while removing duplicates
            combined_indices = T.cat((selected_indices_tensor, new_indices))
            unique_combined_indices = T.unique(combined_indices)

            # Convert the unique indices back to a Python list
            selected_indices = unique_combined_indices
        print(selected_indices)
        selected_indices = np.array(selected_indices)
        print(selected_indices)
        if len(selected_indices) < num_selected_nodes:
            additional_indices = np.random.choice(availability_mask.nonzero().squeeze(), size=num_selected_nodes - len(selected_indices), replace=False)
            print(additional_indices)
            selected_indices = np.concatenate((selected_indices, additional_indices))
            print("finished sample_normal")
        # Calculate log probabilities for the new sampled actions
        log_probs = action_dist.log_prob(new_samples)
        print("log probs", log_probs)
        return selected_indices, log_probs



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
        selection_probabilities = T.stack(((prob_not_selected / total_probabilities), scaled_actions / total_probabilities), dim=1)
        selection_probabilities = T.clamp(selection_probabilities, min=0)

        # print("probabilities", selection_probabilities)

        return selection_probabilities, mu, sigma






   
    def sample_Relu(self, state, num_selected_nodes,reparameterize=True):
        action_probs ,action_mean, action_log_std = self.forward(state)  # Output of the actor network
        action_std = action_log_std.exp()
        
        # Sample actions from the Gaussian distribution
        normal_distribution = Normal(action_mean, action_std)
        if reparameterize:
            sampled_actions = normal_distribution.rsample()  # sample with additional noise
        else:
            sampled_actions = normal_distribution.sample()
        
        total_nodes = self.input_shape[0]

        # Apply ReLU activation to actions to increase the range
        transformed_actions = F.relu(sampled_actions)

        # Scale the transformed actions to a smaller range
        scaled_transformed_actions = transformed_actions * 0.1  # Adjust the scaling factor as needed

        # Scale the transformed actions to the range [0, total_nodes) for node selection
        scaled_actions = scaled_transformed_actions * (total_nodes - 1)

        # Convert scaled actions to integer indices
        selected_indices = scaled_actions.type(T.int64)

        # Ensure no repetition in selected indices
        selected_indices = np.unique(selected_indices.cpu().numpy())
        selected_indices = selected_indices[selected_indices < total_nodes]
        # Ensure the number of selected indices matches num_selected_nodes
        selected_indices = selected_indices[:num_selected_nodes]

        # Truncate selected indices if they exceed the total number of nodes
        

        return selected_indices





    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(),self.checkpoint_file)
    
    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))
   

    
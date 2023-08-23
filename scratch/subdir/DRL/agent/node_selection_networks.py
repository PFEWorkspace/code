from collections import OrderedDict
import os
import select
import torch as T
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
        print("started init actor n_actions",  n_actions, "max axctiosn", max_actions)
        self.input_dims = np.prod(input_shape)
        self.input_shape = input_shape
        self.max_actions = max_actions # number of selected nodes
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,self.name+'_sac')
        self.reparam_noise = 1e-6
        #layer 1
        print("input dims for FC1 in Actor",self.input_dims)
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims) 
        # self.fc1 = nn.Linear(*self.input_dims,out_features=self.fc1_dims)
        print("fc1 passed actor")
        #layer 2
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        #layer 3
        print("fc2 passed actor")

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
        action_probs, action_mean, action_log_std = self.forward(state)
        action_std = action_log_std.exp()

        # Add exploration noise to logits
        noisy_logits = action_mean + exploration_noise * action_std * T.randn_like(action_mean)
        print("noisy shit", noisy_logits)
        # Create a Categorical distribution using the noisy logits
        action_dist = Categorical(action_probs)
        print("actiondist", action_dist)
        # Sample actions from the Categorical distribution
        sampled_actions = action_dist.sample()
        print("sampledddd",sampled_actions)
        selected_indices = sampled_actions.nonzero().squeeze()
        print("indices",len(selected_indices),"num_selected", num_selected_nodes)
        count=1
        while (len(selected_indices) < num_selected_nodes) or count <10:
            count+=1
            print("len", len(selected_indices), "num", num_selected_nodes)
            print("pas assez de noeuds")
            new_samples = action_dist.sample()
            new_indices = new_samples.nonzero().squeeze()
            print("new indices", new_indices)

            # Convert the selected_indices list back to a tensor
            selected_indices_tensor = T.tensor(selected_indices)

            # Combine the selected indices and new indices while removing duplicates
            combined_indices = T.cat((selected_indices_tensor, new_indices))
            unique_combined_indices = T.unique(combined_indices)

            # Convert the unique indices back to a Python list
            selected_indices = unique_combined_indices
            print("indices", selected_indices)

        return selected_indices


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






    
    # def sample_normal(self,state,reparameterize=True):
    #     mu,sigma = self.forward(state)
    #     probabilities = Normal(mu,sigma)
    #     if reparameterize:
    #         actions = probabilities.rsample()  # sample with additional noise
    #     else:
    #         actions = probabilities.sample()
    #     action = T.tanh(actions) * T.tensor(self.max_actions, dtype=T.float32).to(self.device)  # Convert to float32
    #     log_probs = probabilities.log_prob(actions)
    #     log_probs -= T.log(1-action.pow(2)+self.reparam_noise) # type: ignore
    #     log_probs = log_probs.sum(1,keepdim=True)
    #     return action,log_probs
    # def sample_normal(self, state, num_selected_nodes, exploration_noise=0.8, temperature=1.0):
        # action_mean, action_log_std = self.forward(state)
        # action_std = action_log_std.exp()

        # # Sample actions from the Gaussian distribution
        # normal_distribution = Normal(action_mean, action_std)
        # sampled_actions = normal_distribution.rsample()

        # # Add exploration noise
        # sampled_actions += T.tensor(exploration_noise).to(self.device) * T.randn_like(sampled_actions)
        # print("sampled_actions", sampled_actions)

        # # Transform actions to be between -1 and 1 using tanh
        # transformed_actions = T.tanh(sampled_actions)
        # print("transformed_actions", transformed_actions)

        # # Scale the transformed actions to the range [0, total_nodes) for node selection
        # total_nodes = self.input_shape[0]
        # scaled_actions = (transformed_actions + 1.0) * (total_nodes - 1) / 2.0
        # print("scaled_actions", scaled_actions)

        # # Convert scaled actions to integer indices
        # selected_indices = scaled_actions.type(T.int64)
        # print("selected_indices", selected_indices)

        # # Ensure no repetition in selected indices
        # selected_indices = np.unique(selected_indices.cpu().numpy())
        # print("selected_indices after unique", selected_indices)

        # # Randomly sample more indices without repetition if needed
        # while len(selected_indices) < num_selected_nodes:
        #     additional_indices = np.random.choice(total_nodes, size=num_selected_nodes - len(selected_indices), replace=False)
        #     selected_indices = np.concatenate((selected_indices, additional_indices))
        # print("selected_indices after while", selected_indices)

        # # Truncate selected indices if exceeded the total number of nodes
        # selected_indices = selected_indices[:total_nodes]

        # return selected_indices
    def sample_Relu(self, state, num_selected_nodes,reparameterize=True):
        action_probs ,action_mean, action_log_std = self.forward(state)  # Output of the actor network
        action_std = action_log_std.exp()
        
        print("action_probs:", action_probs)
        print("action_mean:", action_mean)
        print("action_log_std:", action_log_std)
        # Sample actions from the Gaussian distribution
        normal_distribution = Normal(action_mean, action_std)
        if reparameterize:
            sampled_actions = normal_distribution.rsample()  # sample with additional noise
        else:
            sampled_actions = normal_distribution.sample()
        
        total_nodes = self.input_shape[0]

        # Apply ReLU activation to actions to increase the range
        transformed_actions = F.relu(sampled_actions)
        print("transformed_actions", transformed_actions)

        # Scale the transformed actions to a smaller range
        scaled_transformed_actions = transformed_actions * 0.1  # Adjust the scaling factor as needed
        print("scaled_transformed_actions", scaled_transformed_actions)

        # Scale the transformed actions to the range [0, total_nodes) for node selection
        scaled_actions = scaled_transformed_actions * (total_nodes - 1)
        print("scaled_actions", scaled_actions)

        # Convert scaled actions to integer indices
        selected_indices = scaled_actions.type(T.int64)
        print("selected indices before unique", selected_indices)

        # Ensure no repetition in selected indices
        selected_indices = np.unique(selected_indices.cpu().numpy())
        print("selected indices after unique", selected_indices)
        selected_indices = selected_indices[selected_indices < total_nodes]
        print("selected indices after truncating to total_nodes", selected_indices)
        # Ensure the number of selected indices matches num_selected_nodes
        selected_indices = selected_indices[:num_selected_nodes]
        print("selected indices after truncating to num_selected_nodes", selected_indices)

        # Truncate selected indices if they exceed the total number of nodes
        

        return selected_indices

    def sample_normal1(self, state, num_selected_nodes, reparameterize=True, exploration_noise=0.8):
        action_probs, action_mean, action_log_std = self.forward(state)
        action_std = action_log_std.exp()

        print("Action mean:", action_mean)
        print("Action log std:", action_log_std)

        # Sample actions from the Gaussian distribution
        if reparameterize:
            sampled_actions = action_mean + action_std * T.randn_like(action_mean)  # Reparameterization trick
        else:
            sampled_actions = action_mean

        print("Sampled actions before threshold:", sampled_actions)

        # Clip actions to ensure they are within the desired range
        sampled_actions = T.clamp(sampled_actions, -1.0, 1.0)

        print("Sampled actions after clamp:", sampled_actions)

        # Apply the threshold to convert actions into 0s and 1s
        threshold = 0.5  # Adjust this threshold as needed
        selected_nodes = (sampled_actions > threshold).type(T.int64)

        print("Selected nodes after threshold:", selected_nodes)

        # Ensure the selected nodes are within the valid range of available nodes
        state = dr.array_to_state(state,8)
        availability_mask = state[:, 1] != 0
        availability_mask = availability_mask.long()

        selected_nodes *= availability_mask.view(-1, 1)

        print("Selected nodes after availability mask:", selected_nodes)

        # Count the number of selected nodes for each sample in the batch
        num_selected = selected_nodes.sum(dim=1)

        print("Number of selected nodes:", num_selected)

        # If the number of selected nodes is less than the desired number, select additional nodes
        remaining_nodes = num_selected_nodes - num_selected
        additional_indices = T.randint(0, availability_mask.sum(), (selected_nodes.shape[0], remaining_nodes.max()))
        available_indices = T.nonzero(availability_mask)[:, 0]
        selected_indices = available_indices[additional_indices]
        for i in range(selected_nodes.shape[0]):
            selected_nodes[i, selected_indices[i]] = 1

        print("Additional selected indices:", selected_indices)

        return selected_nodes
    
    def sample_normal2(self, state, num_selected_nodes, reparameterize=True, exploration_noise=0.8):
        action_probs, action_mean, action_log_std = self.forward(state)  # Output of the actor network
        action_std = action_log_std.exp()
        print('zction prob s from forward',action_probs)
        # Sample actions from the Gaussian distribution
        normal_distribution = Normal(action_mean, action_std)
        if reparameterize:
            sampled_actions = normal_distribution.rsample()  # sample with additional noise
        else:
            sampled_actions = normal_distribution.sample()
        print("sampled before tensore", sampled_actions)

        # Apply exploration noise
        sampled_actions += T.tensor(exploration_noise).to(self.device) * T.randn_like(sampled_actions)
        print("sampled", sampled_actions)
        # Apply tanh activation to actions to ensure they are between -1 and 1
        transformed_actions = T.tanh(sampled_actions)
        print("transformened",transformed_actions)
        # Scale the transformed actions to the range [0, 1] for node selection
        scaled_actions = (transformed_actions + 1.0) / 2.0
        print(scaled_actions)
        # Apply availability mask
        state = dr.array_to_state(state,8)
        availability_mask = state[:, 1] != 0
        availability_mask = availability_mask.float()
        scaled_actions *= availability_mask.view(-1, 1)

        # Select the top `num_selected_nodes` indices with the highest scaled action values
        selected_indices = scaled_actions.argsort(descending=True)[:, :num_selected_nodes]

        return selected_indices
    





    def save_checkpoint(self):
        print('...saving checkpoint...')
        T.save(self.state_dict(),self.checkpoint_file)
    
    def load_checkpoint(self):
        print('...loading checkpoint...')
        self.load_state_dict(T.load(self.checkpoint_file))
   

    
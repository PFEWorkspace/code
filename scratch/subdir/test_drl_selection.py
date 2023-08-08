import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

# Mockup environment and data
class MockEnvironment:
    def __init__(self):
        self.numNodes = 5
        self.nodes = [{'nodeId': i} for i in range(self.numNodes)]

# Mockup configuration
class MockConfig:
    nodes = lambda: None
    nodes.selection = "score"

# Mockup AiHelperAct
class AiHelperAct:
    selectedAggregators = [0] * 5
    selectedTrainers = [0] * 5

def compute_loss(node_index):
    # Simulated loss calculation
    return torch.tensor(np.random.rand())

class ActorCritic(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_size):
        super(ActorCritic, self).__init__()
        
        # Define actor and critic networks
        self.actor = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_nodes),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


# Initialize Actor-Critic network
num_nodes = 5
num_features = 7
hidden_size = 64

# Initialize Actor-Critic network
actor_critic = ActorCritic(num_nodes=num_nodes, num_features=num_features, hidden_size=hidden_size)


# Initialize lambda value and learning rate
lambda_value = 0.5
learning_rate = 0.001


# Load CSV data and preprocess
data = pd.read_csv("generated_nodes.csv")
features = data[["Availability", "Honesty", "Dataset Size", "Frequency", "Transmission Rate", "Task", "Dropout"]]
numerical_cols = ["Dataset Size", "Frequency", "Transmission Rate", "Task"]
features[numerical_cols] = (features[numerical_cols] - features[numerical_cols].mean()) / features[numerical_cols].std()

# Convert state representation to tensor
state_representation = features.to_numpy(dtype=np.float32)  # Convert DataFrame to numpy array

state = torch.tensor(state_representation, dtype=torch.float32)

# Get action probabilities and state value from the actor-critic
action_probs, state_value = actor_critic(state)

# Sample an action based on action probabilities
selected_node_index = torch.multinomial(action_probs, 1).item()
print(f"Selected Node Index: {selected_node_index}")

# Simulate DRL selection
act = AiHelperAct()
act.selectedAggregators = [0] * 5
act.selectedTrainers = [0] * 5

# Calculate reward
reward = -lambda_value * compute_loss(selected_node_index)
print(f"Reward: {reward}")

# Simulate actor and critic updates
action_log_prob = torch.log(action_probs[selected_node_index])
advantage = reward - state_value
actor_loss = -action_log_prob * advantage
critic_loss = nn.MSELoss()(state_value, reward)
optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate)
optimizer.zero_grad()
total_loss = actor_loss + critic_loss
total_loss.backward()
optimizer.step()

print("Actor-Critic Training Completed")

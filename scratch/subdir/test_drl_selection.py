import pandas as pd
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_size):
        super(ActorCritic, self).__init__()

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

# Mockup data generation
np.random.seed(42)
num_nodes = 5
data = {
    "ID": np.arange(num_nodes),
    "Availability": np.random.choice([True, False], num_nodes),
    "Honesty": np.zeros(num_nodes),
    "Dataset Size": np.random.randint(100, 1000, num_nodes),
    "Frequency": np.random.randint(100, 1000, num_nodes),
    "Transmission Rate": np.random.randint(300, 1000, num_nodes),
    "Task": np.zeros(num_nodes),
    "Dropout": np.random.choice([True, False], num_nodes)
}

# Create a DataFrame
df = pd.DataFrame(data)
df.to_csv("generated_nodes.csv", index=False)



# Load CSV data and preprocess
data = pd.read_csv("generated_nodes.csv")
features = data[["Availability", "Honesty", "Dataset Size", "Frequency", "Transmission Rate", "Task", "Dropout"]]
numerical_cols = ["Dataset Size", "Frequency", "Transmission Rate", "Task"]
features[numerical_cols] = (features[numerical_cols] - features[numerical_cols].mean()) / features[numerical_cols].std()

# Convert state representation to tensor
state_representation = features.to_numpy(dtype=np.float32)
state = torch.tensor(state_representation, dtype=torch.float32)

# Initialize Actor-Critic network
num_features = state.shape[1]
hidden_size = 64
actor_critic = ActorCritic(num_nodes, num_features, hidden_size)

# Initialize lambda value and learning rate
lambda_value = 0.5
learning_rate = 0.001

# Get action probabilities and state value from the actor-critic
action_probs, state_value = actor_critic(state)

# Sample an action based on action probabilities
selected_node_index = torch.multinomial(action_probs, 1).item()
print(f"Selected Node Index: {selected_node_index}")

# Calculate reward (simulated loss calculation)
reward = -lambda_value * torch.tensor(np.random.rand())
print(f"Reward: {reward}")

# Simulate actor and critic updates
action_log_prob = torch.log(action_probs[selected_node_index])
advantage = reward - state_value
actor_loss = -action_log_prob * advantage
critic_loss = nn.MSELoss()(state_value, reward)
optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)
optimizer.zero_grad()
total_loss = actor_loss + critic_loss
total_loss.backward()
optimizer.step()

print("Actor-Critic Training Completed")

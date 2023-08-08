# Define the ActorCritic class
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



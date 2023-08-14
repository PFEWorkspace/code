
import gym
import numpy as np
from gym import spaces

class CustomActionSpace(spaces.Space):
    def __init__(self, total_nodes, num_selected):
        self.total_nodes = total_nodes
        self.num_selected = num_selected

    @property
    def shape(self):
        return (self.num_selected,)

    def sample(self):
        selected_indices = np.random.choice(self.total_nodes, size=self.num_selected, replace=False)
        action = np.zeros(self.total_nodes, dtype=np.int32)
        action[selected_indices] = 1
        return action

    def contains(self, x):
        if len(x) != self.total_nodes:
            return False
        selected_indices = np.where(x)[0]
        return len(selected_indices) == self.num_selected

# Create the custom action space
# total_nodes = 15
# num_selected = 5
# custom_action_space = CustomActionSpace(total_nodes, num_selected)

# # Test the custom action space
# for _ in range(5):
#     action = custom_action_space.sample()
#     print("Selected nodes:", np.where(action)[0])

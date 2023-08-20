
import gym
import numpy as np
from gym import spaces

class CustomActionSpace(spaces.Space):
    def __init__(self, total_nodes, num_selected):
        self.total_nodes = total_nodes
        self.num_selected = num_selected
        self.high = np.ones(self.num_selected)  

    @property
    def shape(self):
        return (self.num_selected,)

    def sample(self):
        action = np.random.choice(self.total_nodes, size=self.num_selected, replace=False)
        return action

# Create the custom action space
# total_nodes = 15
# num_selected = 5
# custom_action_space = CustomActionSpace(total_nodes, num_selected)


# # Initialize self.current_state with zeros for the availability column and specific values for the rest of the columns
# current_state = np.zeros((total_nodes, num_selected + 1))
# current_state[:, 0] = np.arange(total_nodes)  # Set node IDs

# print(current_state)
# # Test the custom action space
# for _ in range(5):
#     action = custom_action_space.sample()
#     print("Selected nodes:", np.where(action)[0])

# import numpy as np

# num_selected = 5  # Replace with the desired length

# random_array = np.random.random(num_selected)
# print(random_array)

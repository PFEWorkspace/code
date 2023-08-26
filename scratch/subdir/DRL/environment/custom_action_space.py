
import numpy as np
from gym import spaces


class CustomActionSpace(spaces.Space):
    def __init__(self, total_nodes, num_selected):
        self.total_nodes = total_nodes
        self.num_selected = num_selected
        self.high = np.ones(self.num_selected)

    @property
    def shape(self):
        return (self.total_nodes)

    def sample(self):
        action = np.random.choice(self.total_nodes, size=self.num_selected, replace=False)
        result = np.zeros(self.total_nodes)
        for i in action :
            result[i] = 1
        return result, action
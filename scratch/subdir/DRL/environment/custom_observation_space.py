from time import process_time_ns
import gym
import numpy as np
from gym import spaces
import csv

class CustomObservationSpace(spaces.Dict):
    def __init__(self, total_nodes):
        self.total_nodes = total_nodes
        # Define the min and max values for each feature
        low_freq = 50.0
        high_freq = 300.0
        low_rate = 150.0
        high_rate = 1000.0
        min_honesty = -500.0
        max_honesty = 500.0
        min_data = 100.0
        max_data = 1000.0
        feature_min_values = np.array([0,0.0, min_honesty, min_data, low_freq, low_rate, 0.0,0.0], dtype=np.float32)
        feature_max_values = np.array([total_nodes,1.0, max_honesty, max_data, high_freq, high_rate,1.0 ,100.0], dtype=np.float32)
        observation_low = np.tile(feature_min_values, (total_nodes, 1))
        observation_high = np.tile(feature_max_values, (total_nodes, 1))

        current_state_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)

        observation_space_dict = spaces.Dict(
            {
                "current_state":current_state_space, # tableau de noeuds with features
                "FL_accuracy": spaces.Box(low=0.0, high=1.0, dtype=np.float32),
            }
        )
        super().__init__(observation_space_dict) # type : ignore
    def sample(self):
        obs= super().sample()
        indexes = np.arange(obs["current_state"].shape[0])

        # Replace the first column of current_state with the indexes
        obs["current_state"][:, 0] = indexes
        return obs
    def preprocess_observation(self, current_state, fl_accuracy):
        flattened_current_state = current_state.flatten()
        processed_observation = np.hstack((flattened_current_state, fl_accuracy))

        return processed_observation

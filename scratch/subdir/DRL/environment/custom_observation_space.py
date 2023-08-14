import gym
import numpy as np
from gym import spaces
import csv

class CustomObservationSpace(gym.spaces.Dict):
    def __init__(self, total_nodes, num_selected, num_features):
        self.total_nodes = total_nodes
        self.num_selected = num_selected
        self.num_features = num_features

        # Define the min and max values for each feature
        #TODO define the min maxes according to config
        low_freq = 0.0 
        low_rate = 0.0
        max_honesty = 10.0
        max_data = 1.0
        high_freq = 1.0
        high_rate = 1.0
        feature_min_values = np.array([0.0, 0.0, 0.0, low_freq, low_rate, 0.0], dtype=np.float32)
        feature_max_values = np.array([1.0, max_honesty, max_data, high_freq, high_rate, 1.0], dtype=np.float32)

        # Create arrays for the low and high values for each node's features
        observation_low = np.tile(feature_min_values, (total_nodes, 1))
        observation_high = np.tile(feature_max_values, (total_nodes, 1))
        selection_observation_low = np.tile(feature_min_values, (num_selected, 1))
        selection_observation_high = np.tile(feature_max_values, (num_selected, 1))

        current_state_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)
        selected_previous_space = spaces.Box(low=selection_observation_low, high=selection_observation_high, dtype=np.float32)
        
        # observation_space_dict = {
        #     "current_state": current_state_space,
        #     "previous_state": {
        #         "selected_nodes": selected_previous_space,
        #         "FL_accuracy": spaces.Box(low=0.0, high=1.0, shape=(num_selected,), dtype=np.float32)
        #     }
        # }
        observation_space_dict = spaces.Dict(
            {
                "current_state":current_state_space,
                "previous_state": spaces.Dict(
                    {
                        "selected_nodes": selected_previous_space,
                        "FL_accuracy": spaces.Box(low=0.0, high=1.0, dtype=np.float32),
                    }
                ),
            }
        )
        super().__init__(observation_space_dict)

# Example usage
total_nodes = 15
num_selected = 10
num_features = 6
########################reset function test#############################################
def reset_test()
    csv_filename = "environment/observations.csv"  # Replace with your CSV file name
    with open(csv_filename, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        rows = list(csv_reader)
    # Extract the observations from the CSV rows
    current_state_rows = rows[1:total_nodes+1] # has true and false in it
    #prepocessing the data from csv to change Bool to int
    current_state_preprocessed = []
    for row in current_state_rows:
        preprocessed_row = []
        for value in row[:num_features]:
            if value == "True":
                preprocessed_row.append(1)
            elif value == "False":
                preprocessed_row.append(0)
            else:
                preprocessed_row.append(value)  # Keep other values unchanged
        current_state_preprocessed.append(preprocessed_row)
    # Convert the preprocessed rows to a NumPy array
    current_state = np.array(current_state_preprocessed, dtype=np.float32)[:, :num_features]
    # current_state = np.array(current_state_rows, dtype=np.float32)[:, :num_features]
    print(current_state)
    # Create initial values for other parts of the observation
    initial_selected_previous = np.zeros((num_selected, num_features), dtype=np.float32)
    initial_fl_accuracy = 0.0
    initial_observation = {
        "current_state": current_state,
        "previous_state": {
            "selected_nodes": initial_selected_previous,
            "FL_accuracy": initial_fl_accuracy,
        }
    }
    print(initial_observation)
# custom_observation_space = CustomObservationSpace(total_nodes, num_selected, num_features)
# print(custom_observation_space)
# print("#########################################################")
# print(custom_observation_space.sample())
import gym
import numpy as np
from gym import spaces
import csv

class CustomObservationSpace(gym.spaces.Dict):
    def __init__(self, total_nodes):
        self.total_nodes = total_nodes

        # Define the min and max values for each feature
        #TODO set honesty interval
        low_freq = 50.0 
        high_freq = 300.0
        low_rate = 150.0
        high_rate = 1000.0
        min_honesty = -500.0
        max_honesty = 500.0
        min_data = 100.0
        max_data = 1000.0
        feature_min_values = np.array([0.0, min_honesty, min_data, low_freq, low_rate, 0.0], dtype=np.float32)
        feature_max_values = np.array([1.0, max_honesty, max_data, high_freq, high_rate, 1.0], dtype=np.float32)
        #[availability,honesty,datasize,frequency,transmissionrate,modelaccuracy]
        # Create arrays for the low and high values for each node's features
        observation_low = np.tile(feature_min_values, (total_nodes, 1))
        observation_high = np.tile(feature_max_values, (total_nodes, 1))

        current_state_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float32)        
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
                "FL_accuracy": spaces.Box(low=0.0, high=1.0, dtype=np.float32),
            }
        )
        super().__init__(observation_space_dict)

# Example usage
total_nodes = 15
num_selected = 10
num_features = 6
########################reset function test#############################################
csv_filename = "observations.csv"  # Replace with your CSV file name
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
# Create initial values for other parts of the observation
initial_fl_accuracy = 0.0
initial_observation = {
    "current_state": current_state,
    "FL_accuracy": initial_fl_accuracy
}
# print(initial_observation)
# print("availibitility",initial_observation["current_state"][:,1])
print("current state", current_state)
import random

print("current state", current_state)
action = [2, 5, 7, 10, 11]
print("action", action)
unavailable_indices = []  # Collect indices of nodes with availability 0
current_state_availability = current_state[:, 1]
print("current_state_availability", current_state_availability)

# Check availability and adjust the action
adjusted_action = []
for node_index in action:
    if current_state_availability[node_index] == 1:  # Check for availability
        adjusted_action.append(node_index)
    else:
        # Find a replacement node that is available and not already selected
        available_nodes = np.where(current_state_availability == 1)[0]
        available_nodes = np.setdiff1d(available_nodes, adjusted_action)  # Exclude already selected nodes
        if len(available_nodes) > 0:
            replacement_node = random.choice(available_nodes)
            adjusted_action.append(replacement_node)
            print(f"Node {node_index} is not available. Replaced with Node {replacement_node}")
        else:
            print(f"No available nodes to replace Node {node_index}. Keeping it in the action.")

print("adjusted_action", adjusted_action)


# print("*********************************")
# print("inititla selected nodes" , initial_observation["previous_state"]["selected_nodes"])
# custom_observation_space = CustomObservationSpace(total_nodes, num_selected, num_features)

# # should return the observation of the initial state
# # We need the following line to seed self.np_random
# csv_filename = "observations.csv"  # Replace with your CSV file name
# with open(csv_filename, "r") as csv_file:
#     csv_reader = csv.reader(csv_file)
#     rows = list(csv_reader)
# # Extract the observations from the CSV rows
# current_state_rows = rows[1:total_nodes+1] # has true and false in it we remove first row having the name of the features
# #prepocessing the data from csv to change Bool to int
# current_state_preprocessed = []
# for row in current_state_rows:
#     preprocessed_row = []
#     for value in row[:num_features]:
#         if value == "True":
#             preprocessed_row.append(1)
#         elif value == "False":
#             preprocessed_row.append(0)
#         else:
#             preprocessed_row.append(value)  # Keep other values unchanged
#     current_state_preprocessed.append(preprocessed_row)
# # Convert the preprocessed rows to a NumPy array
# current_state = []
# new_current_state = np.array(current_state_preprocessed, dtype=np.float32)[:, :num_features]
# current_state[:total_nodes] = new_current_state
# # print(current_state)
# # Create initial values for other parts of the observation
# # Access the current_state attribute
# unavailable_indices = []  # Collect indices of nodes with availability 0
# availability_values = [node[1] for node in current_state]
# print("#########################################################")
# print(availability_values)
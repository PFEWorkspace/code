import gym
from gym.spaces import Box, Dict
import numpy as np
from custom_observation_space import CustomObservationSpace
from custom_action_space import CustomActionSpace


class FLNodeSelectionEnv(gym.Env):
    def __init__(self,total_nodes,num_selected , num_features,aggregator_ratio=0.3):
        super().__init__() 
        self.total_nodes = total_nodes
        self.num_selected = num_selected
        self.num_features = num_features
        self.aggregator_ratio = aggregator_ratio
        # Calculate the number of aggregators and trainers based on the ratio
        num_aggregators = int(num_selected * aggregator_ratio)
        num_trainers = num_selected - num_aggregators
        self.num_aggregators = num_aggregators
        self.num_trainers = num_trainers
        self.current_state = np.zeros(self.total_nodes + num_features)
        self.observation_space = CustomObservationSpace(total_nodes,num_selected, num_features)
        self.action_space = CustomActionSpace(total_nodes, num_selected)
    
    def __reset__(self): #CALLED TO INITIATE NEW EPISODE
        #should return the observation of the initial state
        # We need the following line to seed self.np_random
        csv_filename = "environment/observations.csv"  # Replace with your CSV file name
        with open(csv_filename, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
        # Extract the observations from the CSV rows
        current_state_rows = rows[1:self.total_nodes+1] # has true and false in it we remove first row having the name of the features
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
        current_state = np.array(current_state_preprocessed, dtype=np.float32)[:, :self.num_features]
        # print(current_state)
        # Create initial values for other parts of the observation
        initial_selected_previous = np.zeros((self.num_selected, self.num_features), dtype=np.float32)
        initial_fl_accuracy = 0.0
        initial_observation = {
            "current_state": current_state,
            "previous_state": {
                "selected_nodes": initial_selected_previous,
                "FL_accuracy": initial_fl_accuracy,
            }
        }
        self._observation = initial_observation
        return initial_observation
    

    def step(self, action):
        # Use self.current_state_availability as an internal attribute
        current_observation= _get_obs()
        # current_observation = self.env.step(action)  # Corrected line
        self.current_state = current_observation["current_state"]  # Access the current_state attribute
        selected_indices = np.where(action)[0]
        unavailable_indices = []  # Collect indices of nodes with availability 0
        current_state_availability = self.current_state[:self.total_nodes]
        print("current state tout court")
        print(current_state)
        print("********************************************************")
        print("current state availibility")
        print(current_state_availability)
        for node_index in selected_indices:
            if current_state_availability[node_index] == 0:  # Use self.current_state_availability
                unavailable_indices.append(node_index)

        # Remove unavailable nodes from the selected_indices
        selected_indices = np.setdiff1d(selected_indices, unavailable_indices)

        # Adjust the action to only include available nodes
        adjusted_action = np.zeros(self.total_nodes, dtype=np.int32)
        adjusted_action[selected_indices] = 1

        # Proceed with the environment dynamics using the adjusted action
        # Compute the next state, reward, done flag, and other information
        next_state, reward, done, info = ...

        return next_state, reward, done, info

    def __render__(self, mode="human"):
        #should render the environment
        pass

    def __close__(self):
        #should close the environment
        pass
    
    def __seed__(self, seed=None):
        #should set the seed for this env's random number generator(s)
        pass

    def _get_obs(self):
        return self._observation
# Create a sample environment instance
total_nodes = 15
num_selected = 5
num_features = 6
env = FLNodeSelectionEnv(total_nodes, num_selected, num_features)

# Define the availability for each node (replace this with your data)
availability = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0])

# Test actions with different availability constraints
for _ in range(5):
    action = env.action_space.sample()
    print("Selected nodes:", np.where(action)[0])
    
    # Set availability to the current state's availability
    # Pass the availability and action to the environment step function
    next_state, reward, done, info = env.step(action)
    
    print("Next state:", next_state)
    print("Reward:", reward)
    print("Done:", done)
    print()

# # import gym
# # from gym.spaces import Box, Dict
# # import numpy as np

# # num_nodes = 3  # Number of nodes in the network
# # num_features = 6  # Number of features for each node
# # num_selected =10  # Number of nodes to be selected
# # low_freq= 0.0 
# # low_rate=0.0
# # max_honesty=10.0
# # max_data= 1.0
# # high_freq=1.0
# # high_rate=1.0
# # # Define the min and max values for each feature
# # #Availability[0,1],Honesty[],Dataset Size,Frequency,Transmission Rate, model accuracy in round t-1
# # feature_min_values = np.array([0.0, 0.0        , 0.0     , low_freq , low_rate ,0.0], dtype=np.float32)
# # feature_max_values = np.array([1.0, max_honesty, max_data, high_freq, high_rate,1.0], dtype=np.float32)

# # # Create arrays for the low and high values for each node's features
# # observation_low = np.tile(feature_min_values, (num_nodes, 1))
# # observation_high = np.tile(feature_max_values, (num_nodes, 1))


# # # Concatenate the observation and additional information
# # # total_low = np.hstack((observation_low.flatten(), additional_low))
# # # total_high = np.hstack((observation_high.flatten(), additional_high))

# # # Define the observation shape
# # observation_shape = (num_nodes, num_features)
# # previous_selected_shape = (num_selected,num_features)
# # observation_space = Box(low=observation_low, high=observation_high, dtype=np.float32)
# # # observation_space = Box(low=feature_min_values, high=feature_max_values,shape = observation_shape ,dtype=np.float32)
# # print(observation_space.sample())



# ##########################################################################################################################
# import gym
# from gym import spaces
# import numpy as np

# class LimitSelectedNodesWrapper(gym.Wrapper):
#     def __init__(self, env, num_selected):
#         super().__init__(env)
#         self.num_selected = num_selected
#         self.action_space = spaces.MultiBinary(num_selected)

#     def step(self, action):
#         selected_indices = np.where(action)[0]
#         if len(selected_indices) > self.num_selected:
#             # If more nodes are selected than allowed, randomly choose num_selected nodes
#             np.random.shuffle(selected_indices)
#             selected_indices = selected_indices[:self.num_selected]
#             action = np.zeros(self.num_nodes, dtype=np.int)
#             action[selected_indices] = 1

#         return self.env.step(action)

# class CustomEnvironment(gym.Env):
#     def __init__(self, total_nodes, num_selected):
#         self.total_nodes = total_nodes
#         self.num_selected = num_selected
#         self.action_space = spaces.MultiBinary(total_nodes)
#         self.observation_space = spaces.Discrete(1)  # Just a placeholder observation space

#     def step(self, action):
#         selected_indices = np.where(action)[0]
#         if len(selected_indices) > self.num_selected:
#             # If more nodes are selected than allowed, randomly choose num_selected nodes
#             np.random.shuffle(selected_indices)
#             selected_indices = selected_indices[:self.num_selected]
#             action = np.zeros(self.num_nodes, dtype=np.int)
#             action[selected_indices] = 1
    
#         return self.env.step(action)
#     def reset(self):
#         # Your environment reset logic here
#         return self.observation_space.sample()

# # Create a custom environment
# total_nodes = 15
# num_selected = 10
# custom_env = CustomEnvironment(total_nodes, num_selected)

# # Wrap the custom environment with the selection limit wrapper
# wrapped_env = LimitSelectedNodesWrapper(custom_env, num_selected)

# # Test the wrapped environment
# obs = wrapped_env.reset()
# for _ in range(5):
#     action = wrapped_env.action_space.sample()
#     obs, reward, done, _ = wrapped_env.step(action)
#     selected_nodes = np.where(action)[0]
#     print("Selected nodes:", selected_nodes)
#     print("Observation:", obs)
#     print("Reward:", reward)
#     print("Done:", done)
#     print()




import gym
from gym import spaces
import numpy as np

class LimitSelectedNodesWrapper(gym.Wrapper):
    def __init__(self, env, num_selected):
        super().__init__(env)
        self.num_selected = num_selected
        self.action_space = spaces.MultiBinary(num_selected)

    def step(self, action):
        selected_indices = np.where(action)[0]
        
        if len(selected_indices) != self.num_selected:
            available_indices = np.where(action == 0)[0]
            if len(available_indices) < self.num_selected:
                # If there are fewer available indices, select all of them
                selected_indices = available_indices
            else:
                selected_indices = np.random.choice(available_indices, size=self.num_selected, replace=False)
            action = np.zeros(self.env.total_nodes, dtype=np.int32)
            action[selected_indices] = 1

        return self.env.step(action)
class CustomEnvironment(gym.Env):
    def __init__(self, total_nodes, num_selected):
        self.total_nodes = total_nodes
        self.num_selected = num_selected
        self.action_space = spaces.MultiBinary(total_nodes)
        self.observation_space = spaces.Discrete(1)  # Just a placeholder observation space

    def step(self, action):
        # Your environment logic here
        return self.observation_space.sample(), 0, False, {}

    def reset(self):
        # Your environment reset logic here
        return self.observation_space.sample()

# Create a custom environment
total_nodes = 15
num_selected = 10
custom_env = CustomEnvironment(total_nodes, num_selected)

# Wrap the custom environment with the selection limit wrapper
wrapped_env = LimitSelectedNodesWrapper(custom_env, num_selected)

# Test the wrapped environment
obs = wrapped_env.reset()
# for _ in range(5):
#     action = wrapped_env.action_space.sample()
#     obs, reward, done, _ = wrapped_env.step(action)
#     selected_nodes = np.where(action)[0]
#     print("Selected nodes:", selected_nodes)
#     print("Observation:", obs)
#     print("Reward:", reward)
#     print("Done:", done)
#     print()


# import csv 
# num_features = 5
# csv_filename = "environment/observations.csv"  # Replace with your CSV file name
# with open(csv_filename, "r") as csv_file:
#         csv_reader = csv.reader(csv_file)
#         rows = list(csv_reader)
#         # Extract the observations from the CSV rows
#         current_state_rows = rows[1:15+1] 
#         print(current_state_rows)
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
# new_current_state = np.array(current_state_preprocessed, dtype=np.float32)[:, :num_features]
# print(new_current_state)
import csv
total_nodes = 15
num_selected = 10
num_features = 5
# We need the following line to seed self.np_random
csv_filename = "environment/observations.csv"  # Replace with your CSV file name
with open(csv_filename, "r") as csv_file:
    csv_reader = csv.reader(csv_file)
    rows = list(csv_reader)
# Extract the observations from the CSV rows
current_state_rows = rows[1:total_nodes+1] # has true and false in it we remove first row having the name of the features
#prepocessing the data from csv to change Bool to int
current_state_preprocessed = []
for row in current_state_rows:
    preprocessed_row = []
    for value in row[:num_features+1]:
        if value == "True":
            preprocessed_row.append(1)
        elif value == "False":
            preprocessed_row.append(0)
        else:
            preprocessed_row.append(value)  # Keep other values unchanged
    current_state_preprocessed.append(preprocessed_row)
# Convert the preprocessed rows to a NumPy array
new_current_state = np.array(current_state_preprocessed, dtype=np.float32)[:, :num_features+1]
new_column = np.zeros((new_current_state.shape[0], 1))
# Append the new column to the existing array
current_state = np.append(new_current_state, new_column, axis=1) #added the accuracy of local model column
current_state[:total_nodes] = current_state
# Create initial values for other parts of the observation
current_observation = {
    "current_state": current_state,
    "FL_accuracy": 0.0
}
print(current_observation)
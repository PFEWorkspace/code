import gym
from gym.spaces import Box, Dict
import numpy as np
from custom_observation_space import CustomObservationSpace
from custom_action_space import CustomActionSpace
from numpy.random import default_rng


class FLNodeSelectionEnv(gym.Env):
    def __init__(self,total_nodes,num_selected , num_features,target,aggregator_ratio=0.3):
        super().__init__() 
        self.total_nodes = total_nodes
        self.num_selected = num_selected
        self.num_features = num_features
        self.aggregator_ratio = aggregator_ratio
        self.target_accuracy = target
        # Calculate the number of aggregators and trainers based on the ratio
        num_aggregators = int(num_selected * aggregator_ratio)
        num_trainers = num_selected - num_aggregators
        self.num_aggregators = num_aggregators
        self.num_trainers = num_trainers
        self.current_state = current_state = np.zeros((total_nodes, num_features+2)) # room for id and node accuracy
        self.current_state[:, 0] = np.arange(total_nodes)  # Set node IDs

        self.observation_space = CustomObservationSpace(total_nodes)
        self.action_space = CustomActionSpace(total_nodes, num_selected)
        # setting the initial state
        self.fl_accuracy = 0.0
        self.rng = default_rng()
        self.current_round=0

    
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
        new_current_state = np.array(current_state_preprocessed, dtype=np.float32)[:, :self.num_features+1]
        new_column = np.zeros((new_current_state.shape[0], 1))
        # Append the new column to the existing array
        current_state = np.append(new_current_state, new_column, axis=1) #added the accuracy of local model column
        self.current_state[:self.total_nodes] = current_state
        # Create initial values for other parts of the observation
        current_observation = {
            "current_state": current_state,
            "FL_accuracy": 0.0
        }
        self._observation = current_observation
        return current_observation
    
    def step(self, action):
        #Adjusting action 
        current_observation = _get_obs()
        self.current_state = current_observation["current_state"]  # Access the current_state attribute
        self.current_fl_accuracy = current_observation["FL_accuracy"]
         # Perform DRL selection
        self.send_selected_nodes_to_network(action) # action is the list of selected nodes

        # Wait for network to provide updates
        updated_accuracies = self.receive_updated_accuracies_from_network()
        updated_nodes = self.receive_updated_nodes_from_network()
        updated_losses = self.receive_updated_losses_from_network()
        updated_fl_accuracy = self.receive_updated_fl_accuracy_from_network()

        # Update the state of the environment with received updates
        self.update_environment_state_with_network_updates(updated_nodes, updated_fl_accuracy, updated_accuracies)

        # Simulate FL round and get rewards
        node_rewards = self.calculate_reward(selected_nodes,updated_losses)
        agent_reward = sum(node_rewards)# or agent_reward = self.agent_reward(node_rewards) in case we change the way we calcultae the agent reward
        # Update the state of the environment
        next_observation = self.update_environment_state()
        self.current_round += 1
        # Check if the maximum number of rounds is reached or the target accuracy is achieved
        done = self.current_round >= self.max_rounds or self.target_accuracy_achieved(updated_fl_accuracy)
        return next_observation, rewards, done, info

    def send_selected_nodes_to_network(selected)
        # Send the selected nodes to the network
        selected_aggregators= np.random.choice(selected, size=self.num_aggregators, replace=False)
        selected_trainers = np.setdiff1d(selected, selected_aggregators)
        #communicate to aiHelperAct
        accuracies = np.random.random(len(selected_trainers))
        fl_accuracy = np.random.random()
        return fl_accuracy , accuracies , selected_trainers
    def update_environment_state_with_network_updates()
        # Update the state of the environment with received updates
        pass
    
    
    def agent_reward(self, node_rewards):
        return sum(node_rewards)

    def calculate_reward(self, selected_nodes, updated_losses):
        node_rewards = np.zeros(self.total_nodes)
        for node_index, loss in zip(selected_nodes, updated_losses):
            node_rewards[node_index] = -loss  # Use -loss as a simple example
        return node_rewards
    def target_accuracy_achieved(self, updated_accuracy):
        return updated_accuracy >= self.target_accuracy

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
        return self.current_observation
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

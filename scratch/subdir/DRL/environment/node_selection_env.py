import csv
import os
import gym
import numpy as np
from .custom_observation_space import CustomObservationSpace
from .custom_action_space import CustomActionSpace
from numpy.random import default_rng
from DRL import drl_utils as dr

class FLNodeSelectionEnv(gym.Env):
    def __init__(self,total_nodes,num_selected_agg, num_selected_trainer , num_features,target,max_rounds,aggregator_ratio=0.3):
        super().__init__()
        self.total_nodes = total_nodes
        self.num_selected = num_selected_agg + num_selected_trainer
        self.num_features = num_features
        self.aggregator_ratio = aggregator_ratio
        self.target_accuracy = target
        # Calculate the number of aggregators and trainers based on the ratio
        num_aggregators = num_selected_agg
        num_trainers = num_selected_trainer
        self.num_aggregators = num_aggregators
        self.num_trainers = num_trainers
        self.current_state =  np.zeros((total_nodes, num_features+1)) # room for id and node accuracy
        self.current_state[:, 0] = np.arange(total_nodes)  # Set node IDs
        self.observation_space = CustomObservationSpace(total_nodes)
        self.action_space = CustomActionSpace(total_nodes, self.num_selected)
        # setting the initial state
        self.fl_accuracy = 0.0
        self.current_observation = {
                "current_state":self.current_state,
                "FL_accuracy": self.fl_accuracy}

        self.current_round=0
        self.max_rounds: int =max_rounds

    def set_act(self, act):
        self.act = act
    def reset(self, filename): #CALLED TO INITIATE NEW EPISODE
        #should return the observation of the initial state
        # We need the following line to seed self.np_random
        csv_filename = filename  # Replace with your CSV file name
        with open(csv_filename, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            rows = list(csv_reader)
        # Extract the observations from the CSV rows
        current_state_rows = rows[1:self.total_nodes+1] # has true and false in it we remove first row having the name of the features
        #prepocessing the data from csv to change Bool to int
        current_state_preprocessed = []
        for row in current_state_rows:
            preprocessed_row = []
            for value in row[:self.num_features]:
                if value == "true":
                    preprocessed_row.append(1)
                elif value == "false":
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
        info = {"msg" : "success"}
        return current_observation , info

    def step(self, action, accuracies, nodes, losses, fl_accuracy):
        updated_nodes =dr.get_nodes_withaccuracy(nodes, self.total_nodes,accuracies)
        updated_fl_accuracy =fl_accuracy

        # Update the state of the environment with received updates
        next_observation = self.update_environment_state_with_network_updates(updated_nodes, fl_accuracy)

        
        # Simulate FL round and get rewards
        node_rewards = self.calculate_reward(action,accuracies)
        agent_reward = sum(node_rewards)# or agent_reward = self.agent_reward(node_rewards) in case we change the way we calcultae the agent reward
        # Update the state of the environment
        self.current_round += 1
        # Check if the maximum number of rounds is reached or the target accuracy is achieved
        done = self.current_round >= self.max_rounds or self.target_accuracy_achieved(updated_fl_accuracy)

        return next_observation, agent_reward,done


    def update_environment_state_with_network_updates(self,nodes,FL_accuracy):
        # Update the state of the environment with received updates
        #check the shape of nodes if it includes accuracy
        nodes= nodes.astype(np.float32) # cast the array to accept npfloat
        obs= {
            "current_state": nodes,
            "FL_accuracy": FL_accuracy
        }
        self.current_observation = obs
        return obs


    def agent_reward(self, node_rewards):
        return sum(node_rewards)

    def calculate_reward(self, selected_nodes, updated_accuracies):
        #todo to change to accuracy
        # print("nodes in reward", self.current_observation)
        # print("selected nodes",  selected_nodes)
        state = self.current_observation["current_state"]
        node_rewards = np.zeros(self.total_nodes)
        # print ("updated_losses", updated_losses)
        for node_index in selected_nodes:
            node_rewards[node_index] = state[node_index][2]
            if (updated_accuracies[node_index] != 0) :
                node_rewards[node_index] *= updated_accuracies[node_index]  # Use loss as a simple example because loss is positive
            # print("node reward", node_rewards[node_index])
        return node_rewards
    def target_accuracy_achieved(self, updated_accuracy):
        return updated_accuracy >= self.target_accuracy

    # def __render__(self, mode="human"):
    #     #should render the environment
    #     pass

    # def __close__(self):
    #     #should close the environment
    #     pass

    # def __seed__(self, seed=None):
    #     #should set the seed for this env's random number generator(s)
    #     pass

    def _get_obs(self):
        return self.current_observation
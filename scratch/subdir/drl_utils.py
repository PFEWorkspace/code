from typing import List
import numpy as np
def flatten_nodes(nodes):
    return nodes.flatten()
def flatten_observation(observation):
    flattened_nodes = flatten_nodes(observation['current_state'])
    flattened_observation = np.concatenate((flattened_nodes, [observation["FL_accuracy"]]))
    return flattened_observation
def get_observation(nodes):
    obs = []
    for node in nodes:
        availability = 1.0 if node.availability =="True" else 0.0
        obs.append(np.array([node.nodeId, availability, node.honesty, node.datasetSize, node.freq, node.transRate, 0.0,0.0]))

    obs_array = np.array(obs)
    return obs_array

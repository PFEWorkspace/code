from typing import List
import numpy as np
from torch import dropout_
def flatten_nodes(nodes):
    return nodes.flatten()
def flatten_observation(observation):
    flattened_nodes = flatten_nodes(observation['current_state'])
    flattened_observation = np.concatenate((flattened_nodes, [observation["FL_accuracy"]]))
    print("flat obs", flattened_observation)
    return flattened_observation

def get_obs(nodes):
    obs = []
    for node in nodes:
        availability = 1.0 if node.availability ==True else 0.0
        dropout = 1.0 if node.dropout == True else 0.0 
        obs.append(np.array([node.nodeId, availability, node.honesty, node.datasetSize, node.freq, node.transRate, dropout,0.0]))

    obs_array = np.array(obs)
    return obs_array
def get_observation(nodes, target_size):
    obs=[]
    print("target size in get osb", target_size)
    for i, node in enumerate(nodes):
        availability = 1.0 if node.availability ==True else 0.0
        dropout = 1.0 if node.dropout == True else 0.0 
        obs.append(np.array([i, availability, node.honesty, node.datasetSize, node.freq, node.transRate, dropout, 0.0]))

    obs_array = np.array(obs)
    # print("nulber of nodes passed in arguments", len(nodes))
    if len(obs_array) < target_size:
        last_node_id = len(nodes) - 1
        num_fill = target_size - len(obs_array)
        fill_array = np.zeros((num_fill, obs_array.shape[1]))
        fill_array[:, 0] = np.arange(last_node_id + 1, last_node_id + num_fill + 1)
        obs_array = np.vstack((obs_array, fill_array))
    print("in get observation chekcing the obs", obs_array)
    return obs_array

def get_nodes_withaccuracy(nodes, target_size, acc):
    obs=[]
    print("target size in get osb", target_size)
    for i, node in enumerate(nodes):
        availability = 1.0 if node[1] ==True else 0.0
        dropout = 1.0 if node[6] == True else 0.0 
        obs.append(np.array([i, availability, node[2], node[3], node[4], node[5], dropout, acc[i]]))

    obs_array = np.array(obs)
    print("nulber of nodes passed in arguments", len(nodes))
    if len(obs_array) < target_size:
        last_node_id = len(nodes) - 1
        num_fill = target_size - len(obs_array)
        fill_array = np.zeros((num_fill, obs_array.shape[1]))
        fill_array[:, 0] = np.arange(last_node_id + 1, last_node_id + num_fill + 1)
        obs_array = np.vstack((obs_array, fill_array))
    print("in get observation chekcing the obs", obs_array)
    return obs_array

def adjust_observation_with_nodes(observation, nodes):
    node_ids = [int(node.nodeId) for node in nodes]
    
    for row in observation:
        node_id = int(row[0])
        if node_id not in node_ids:
            row[1] = 0.0
            
    return observation
def array_to_dict(tab, features):
    state_rows = np.int16((len(tab)-1)/features)
    state = tab[0:(len(tab)-1)].reshape(state_rows,features)
    dict_obs={
        "current_state" : state,
        "FL_accuracy" : tab[-1]
    }
    return dict_obs
def array_to_state(tab, features):
    state_rows = np.int16((len(tab))/features)
    state = tab[0:(len(tab))].reshape(state_rows,features)
    # print("here si ur state", state)
    return state
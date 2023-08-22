import numpy as np
np.set_printoptions(precision=2, suppress=True)

def adjust_observation_with_nodes(observation, nodes):
    node_ids = [int(node.nodeId) for node in nodes]
    
    for row in observation:
        node_id = int(row[0])
        if node_id not in node_ids:
            row[1] = 0.0
            
    return observation

# Sample observation and nodes
observation = np.array([
    [0, 1.0, -27650.73, 460.0, 70.0, 950.0, 1.0, 0.0],
    [1, 1.0, 42.03, 910.0, 180.0, 250.0, 1.0, 0.0],
    [2, 1.0, 45.95, 840.0, 140.0, 530.0, 1.0, 0.0],
    [3, 1.0, -10.63, 230.0, 290.0, 390.0, 0.0, 0.0],
    [4, 0.0, 41.85, 900.0, 140.0, 640.0, 0.0, 0.0],
    [5, 1.0, 47.91, 670.0, 180.0, 720.0, 1.0, 0.0],
])
print(observation)

class Node:
    def __init__(self, nodeId, availability):
        self.nodeId = nodeId
        self.availability = availability

# Sample nodes
nodes = [
    Node(nodeId=3, availability="False"),
    Node(nodeId=4, availability="True"),
    Node(nodeId=5, availability="True"),
]

# Adjust the observation
adjusted_observation = adjust_observation_with_nodes(observation, nodes)

print("Original Observation:")
print(observation)
print("\nAdjusted Observation:")
print(adjusted_observation)

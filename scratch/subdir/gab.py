import numpy as np
np.set_printoptions(precision=2, suppress=True)

# def adjust_observation_with_nodes(observation, nodes):
#     node_ids = [int(node.nodeId) for node in nodes]
    
#     for row in observation:
#         node_id = int(row[0])
#         if node_id not in node_ids:
#             row[1] = 0.0
            
#     return observation

# # Sample observation and nodes
# observation = np.array([
#     [0, 1.0, -27650.73, 460.0, 70.0, 950.0, 1.0, 0.0],
#     [1, 1.0, 42.03, 910.0, 180.0, 250.0, 1.0, 0.0],
#     [2, 1.0, 45.95, 840.0, 140.0, 530.0, 1.0, 0.0],
#     [3, 1.0, -10.63, 230.0, 290.0, 390.0, 0.0, 0.0],
#     [4, 0.0, 41.85, 900.0, 140.0, 640.0, 0.0, 0.0],
#     [5, 1.0, 47.91, 670.0, 180.0, 720.0, 1.0, 0.0],
# ])
# print(observation)

# class Node:
#     def __init__(self, nodeId, availability):
#         self.nodeId = nodeId
#         self.availability = availability

# # Sample nodes
# nodes = [
#     Node(nodeId=3, availability="False"),
#     Node(nodeId=4, availability="True"),
#     Node(nodeId=5, availability="True"),
# ]

# # Adjust the observation
# adjusted_observation = adjust_observation_with_nodes(observation, nodes)

# print("Original Observation:")
# print(observation)
# print("\nAdjusted Observation:")
# print(adjusted_observation)
def array_to_dict(tab, features):
    state_rows = np.int16((len(tab)-1)/features)
    state = tab[0:(len(tab)-1)].reshape(state_rows,features)
    dict_obs={
        "current_state" : state,
        "FL_accuracy" : tab[-1]
    }
    return dict_obs

ar=  np.array([   0.,      1.,      9.69,  390.,    260.,    920.,      0.,      0.,
 1. ,     1. ,     0.   , 820. ,   290. ,   960. ,     0. ,     0. , 
 2. ,     0. ,     9.26 , 620. ,   150. ,   910. ,     0. ,     0. , 
 3. ,     1. ,    11.1  , 670. ,   180. ,   990. ,     0. ,     0. , 
 4. ,     1. ,   -75.43 , 780. ,   130. ,   990. ,     0. ,     0. , 
 5. ,     1. ,  -176.35 , 310. ,   290. ,   970. ,     0. ,     0. , 
 6. ,     1. ,     4.24 , 310. ,   130. ,   980. ,     0. ,     0. , 
 7. ,     0. ,     9.55 , 190. ,    60. ,   960. ,     0. ,     0. , 
 8. ,     1. ,     9.93 , 910. ,   190. ,   740. ,     0. ,     0. , 
 9. ,     0. ,     0.   , 220. ,   160. ,   980. ,     0. ,     0. , 
10. ,     1. ,   -54.17 , 170. ,   140. ,   900. ,     0. ,     0. , 
11. ,     1. ,     3.05 , 170. ,    60. ,   820. ,     0. ,     0. , 
12. ,     0. ,     0.13 , 700. ,   220. ,   850. ,     0. ,     0. , 
13. ,     1. ,     2.46 , 680. ,    60. ,   860. ,     0. ,     0. , 
14. ,     0. ,     5.86 , 650. ,   190. ,   680. ,     0. ,     0. , 
15. ,     1. ,     7.08 , 480. ,   170. ,   830. ,     0. ,     0. , 
16. ,     1. ,     5.62 , 270. ,    90. ,   780. ,     0. ,     0. , 
17. ,     0. ,   -83.   , 240. ,   230. ,   780. ,     0. ,     0. , 
18. ,     1. ,     0.   , 520. ,   290. ,   770. ,     0. ,     0. , 
19. ,     0. ,    10.64 , 260. ,   120. ,   800. ,     0. ,     0. , 
20. ,     1. ,     7.54 , 320. ,   110. ,   790. ,     0. ,     0. , 
21. ,     0. ,     9.26 , 260. ,   180. ,   760. ,     0. ,     0. , 
22. ,     1. ,     0.   , 310. ,   100. ,   730. ,     0. ,     0. , 
23. ,     0. ,    10.67 , 520. ,   300. ,   760. ,     0. ,     0. , 
24. ,     1. ,     8.3  , 310. ,    70. ,   680. ,     0. ,     0. , 35.])

di = array_to_dict(ar,8)

print(di)


 def sample_normal(self, state, num_selected_nodes,reparameterize=True, exploration_noise=0.8):
        print("debut of sample normal")
        action_probs,action_mean, action_log_std = self.forward(state)  # Output of the actor network
        action_std = action_log_std.exp()
        print ("in sample normal got mean",action_mean)
        # Sample actions from the Gaussian distribution
        normal_distribution = Normal(action_mean, action_std)
        if reparameterize:
            sampled_actions = normal_distribution.rsample()  # sample with additional noise
        else:
            sampled_actions = normal_distribution.sample()
        total_nodes = self.input_shape[0]
        sampled_actions += T.tensor(exploration_noise).to(self.device) * T.randn_like(sampled_actions)

        # Apply tanh activation to actions to ensure they are between -1 and 1
        transformed_actions = T.tanh(sampled_actions)
        print("transformed_actions sample normal", transformed_actions)

        # Scale the transformed actions to the range [0, total_nodes) for node selection
        scaled_actions = (transformed_actions + 1.0) * (total_nodes - 1) / 2.0
        print("scaled_actions sample normal", scaled_actions)

        # Convert scaled actions to integer indices
        selected_indices = scaled_actions.type(T.int64)
        # print("selected indices before unique", selected_indices)

        # Ensure no repetition in selected indices
        selected_indices = np.unique(selected_indices.cpu().numpy())
        # print("selected indices after unique", selected_indices)

        # Find the indices that are most closely aligned with the transformed actions
        additional_indices = np.argsort(np.abs(scaled_actions.detach().cpu().numpy() - selected_indices[:, None]), axis=1)

        # Select additional indices that are unique and not already selected
        additional_indices = additional_indices.flatten()
        additional_indices = additional_indices[np.isin(additional_indices, selected_indices, invert=True)]
        additional_indices = additional_indices[:num_selected_nodes - len(selected_indices)]

        # Add the additional indices to the selected indices
        selected_indices = np.concatenate((selected_indices, additional_indices))

        # Truncate selected indices if exceeded the total number of nodes
        selected_indices = selected_indices[:total_nodes]
        # print("selected indices after adding additional indices", selected_indices)

        return selected_indices




selection prob  tensor([[1.0000, 0.0000],[0,
        [0.8902, 0.1098], 1, 
        [0.4701, 0.5299],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [0.3797, 0.6203],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [0.7726, 0.2274],
        [0.0103, 0.9897],
        [0.7630, 0.2370],
        [0.5081, 0.4919],
        [1.0000, 0.0000],
        [0.7663, 0.2337],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [0.0000, 1.0000],
        [0.8155, 0.1845],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [0.2141, 0.7859],
        [1.0000, 0.0000],
        [0.5642, 0.4358],
        [1.0000, 0.0000],
        [0.3468, 0.6532],
        [0.0000, 1.0000],
        [0.4658, 0.5342],
        [1.0000, 0.0000],
        [0.0000, 1.0000],
        [1.0000, 0.0000],
        [0.6254, 0.3746],
        [0.1949, 0.8051],
        [1.0000, 0.0000],
        [0.4156, 0.5844],
        [0.1511, 0.8489],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [1.0000, 0.0000],
        [0.4860, 0.5140]], 
        
        
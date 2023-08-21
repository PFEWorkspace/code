# # Import necessary libraries
# import numpy as np
# import gym
# from environment.node_selection_env import FLNodeSelectionEnv  # Replace with your actual module name
# from agent.node_selection_agent import Agent  # Replace with your actual module name

# # Set random seed for reproducibility (optional but recommended)
# np.random.seed(42)

# # Create an instance of the environment
# total_nodes = 15
# num_selected = 5
# num_features = 6
# env = FLNodeSelectionEnv(total_nodes, num_selected, num_features, 0.89,40)

# # Create an instance of the agent
# agent = Agent(alpha=0.0003, beta=0.0003, input_dims=[total_nodes + num_features + 1],
#               env=env, gamma=0.99, n_actions=num_selected, max_size=1000000,
#               tau=0.005, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2)

# # Training loop
# num_episodes = 1000  # Adjust the number of episodes as needed

# for episode in range(num_episodes):
#     observation = env.reset()  # Reset the environment and get the initial observation
#     done = False
#     episode_reward = 0
    
#     while not done:
#         action = agent.choose_action(observation)  # Choose an action using the actor network
#         # next_observation, reward, done, _ = env.step(action)  # Take the chosen action
        
#         # Store the experience in the replay buffer
#         agent.remember(observation, action, reward, next_observation, done)
        
#         # Update the agent's networks
#         agent.learn()
        
#         observation = next_observation
#         episode_reward += reward
    
#     print(f"Episode: {episode+1}, Reward: {episode_reward:.2f}")

# # Save the trained models
# agent.save_models()

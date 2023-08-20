from environment.node_selection_env import FLNodeSelectionEnv
import gym
import numpy as np
from agent.node_selection_agent import Agent
from agent.utils import plot_learning_curve
from environment.node_selection_env import FLNodeSelectionEnv  # Replace with the actual import path of your custom environment class
class MockAct:
    def __init__(self, accuracies, nodes, losses, fl_accuracy):
        self.accuracies = accuracies
        self.nodes = nodes
        self.losses = losses
        self.fl_accuracy = fl_accuracy



def flatten_nodes(nodes):
    return nodes.flatten()
def flatten_observation(observation):
    flattened_nodes = flatten_nodes(observation['current_state'])
    flattened_observation = np.concatenate((flattened_nodes, [observation["FL_accuracy"]]))
    return flattened_observation
def main():
    # Register your custom environment
    
    env_name = 'CustomNodeSelection-v0'
    gym.register(id=env_name, entry_point=FLNodeSelectionEnv)

    # Create an instance of your custom 
    # Usage 
    total_nodes = 50
    num_selected = 30
    num_features = 6
    accuracies = np.array([0.85, 0.91, 0.78, 0.92])
    nodes = np.array([[   0.,    1.,    0.,  100.,  300., 1000. ,   0.],
 [   1.,    1.,    0.,  900.,  200.,  530.,    0.],
 [   2.,    1.,    0.,  350.,  150.,  690.,    0.],
 [   3.,    1.,    0.,  740.,  140.,  680.,    0.],
 [   4.,    0.,    0.,  300.,  100.,  390.,    0.],
 [   5.,    1.,    0.,  810.,  190.,  860.,    0.],
 [   6.,    1.,    0.,  690.,  210.,  460.,    0.],
 [   7.,    0.,    0.,  830.,  290.,  300.,    0.],
 [   8.,    1.,    0.,  450.,  300.,  710.,    0.],
 [   9.,    1.,    0.,  810.,  210.,  390.,    0.],
 [  10.,    1.,    0., 1000.,  190.,  190.,    0.],
 [  11.,    1.,    0.,  770.,   60.,  640.,    0.],
 [  12.,    0.,    0.,  680.,  140.,  370.,    0.],
 [  13.,    0.,    0.,  140.,  170.,  750.,    0.],
 [  14.,    0.,    0.,  500.,  160.,  210.,    0.],
 [  15.,    1.,    0.,  540.,  220.,  730.,    0.],
 [  16.,    1.,    0.,  930.,  210.,  960.,    0.],
 [  17.,    1.,    0.,  340.,  210.,  880.,    0.],
 [  18.,    0.,    0.,  270.,  140.,  930.,    0.],
 [  19.,    0.,    0.,  260.,  260.,  660.,    0.],
 [  20.,    1.,    0.,  310.,  190.,  680.,    0.],
 [  21.,    1.,    0.,  730.,  150.,  930.,    0.],
 [  22.,    0.,    0.,  120.,   50.,  470.,    0.],
 [  23.,    1.,    0.,  380.,  220.,  740.,    0.],
 [  24.,    1.,    0.,  160.,  290.,  460.,    0.],
 [  25.,    1.,    0.,  360.,  300.,  620.,    0.],
 [  26.,    1.,    0.,  370.,  270.,  940.,    0.],
 [  27.,    1.,    0.,  820.,  110.,  220.,    0.],
 [  28.,    1.,    0.,  480.,  240.,  510.,    0.],
 [  29.,    1.,    0.,  580.,  150.,  310.,    0.],
 [  30.,    1.,    0.,  980.,  130.,  710.,    0.],
 [  31.,    0.,    0.,  880.,  270.,  220.,    0.],
 [  32.,    1.,    0.,  420.,  290.,  950.,    0.],
 [  33.,    0.,    0.,  270.,  170.,  770.,    0.],
 [  34.,    1.,    0.,  490.,   50.,  860.,    0.],
 [  35.,    1.,    0.,  340.,  250.,  220.,    0.],
 [  36.,    1.,    0.,  940.,  120., 1000.,    0.],
 [  37.,    1.,    0.,  330.,  210.,  600.,    0.],
 [  38.,    1.,    0.,  860.,  130.,  830.,    0.],
 [  39.,    1.,    0.,  490.,  140.,  990.,    0.],
 [  40.,    1.,    0.,  900.,  150.,  650.,    0.],
 [  41.,    1.,    0.,  750.,  230.,  620.,    0.],
 [  42.,    1.,    0.,  880.,  200.,  530.,    0.],
 [  43.,    1.,    0.,  570.,  160.,  840.,    0.],
 [  44.,    0.,    0.,  680.,  220.,  350.,    0.],
 [  45.,    1.,    0.,  790.,  220.,  890.,    0.],
 [  46.,    1.,    0.,  410.,  220.,  350.,    0.],
 [  47.,    1.,    0.,  360.,  240.,  480.,    0.],
 [  48.,    1.,    0.,  100.,  120.,  320.,    0.],
 [  49.,    1.,    0.,  890.,   80.,  250.,    0.]])
    losses = np.array([0.15, 0.09, 0.22, 0.08])
    fl_accuracy = 0.89
    num_additional_nodes = len(nodes) - len(accuracies)
    additional_accuracies = np.random.uniform(0.5, 0.95, num_additional_nodes)
    additional_losses = np.random.uniform(0.05, 0.3, num_additional_nodes)

# Combine the existing and additional data
    all_accuracies = np.concatenate([accuracies, additional_accuracies])
    all_losses = np.concatenate([losses, additional_losses])
    env = FLNodeSelectionEnv(total_nodes= total_nodes, num_selected=num_selected, num_features=num_features, target=0.8, max_rounds=150)
    mock_act = MockAct(all_accuracies, nodes, all_losses, fl_accuracy)
    # print("mock_act type : ",type(mock_act))
    # env = gym.make(env_name, total_nodes= total_nodes, num_selected=num_selected, num_features=num_features, target=0.8, max_rounds=150)
    obs_shape = env.observation_space.sample()["current_state"].shape
    # print("shape of the obs sample",obs_shape)
    # agent = Agent(input_shape=(total_nodes * num_features,),input_dims=total_nodes*num_features, n_actions=env.action_space.shape[0], env=env)
    
    agent = Agent(input_shape=obs_shape ,n_actions=env.action_space.shape[0], env=env)   
    n_games = 200
    filename = 'node_selection.png'
    figure_file = 'plots/' + filename
    best_score=env.reward_range[0]
    score_history = []
    load_checkpoint = False
    if (load_checkpoint):
        agent.load_models()    
    for i in range(n_games):
        observation, _= env.reset()
        
        done = False
        score = 0
        # flat_obs = env.observation_space.preprocess_observation(observation["current_state"],observation["FL_accuracy"])
        flat_obs= observation["current_state"]
        flat = flatten_nodes(flat_obs) # array or arrays becomes array 
        while not done:
            action = agent.choose_action(flat)
            print("in main selection checking action" , action)
            mock_act = MockAct(all_accuracies, nodes, all_losses, fl_accuracy)

            observation_, reward, done, info = env.step(action,mock_act)
            score += reward
            agent.remember(flat_obs, action, reward, observation_["current_state"], done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

if __name__ == "__main__":
    main()






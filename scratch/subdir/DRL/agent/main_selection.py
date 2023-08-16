from node_selection_env import FLNodeSelectionEnv
import gym
import numpy as np
from node_selection_agent import Agent
from utils import plot_learning_curve
from environment.node_selection_env import FLNodeSelectionEnv  # Replace with the actual import path of your custom environment class

def main():
    # Register your custom environment
    env_name = 'CustomNodeSelection-v0'
    gym.register(id=env_name, entry_point=FLNodeSelectionEnv)

    # Create an instance of your custom environment
    env = gym.make(env_name)
    agent = Agent(input_dims=env.observation_space.shape,
                  env=env, n_actions=env.action_space.shape[0])
    n_games = 200
    filename = 'node_selection.png'
    figure_file = 'plots/' + filename
    best_score=env.reward_range[0]
    score_history = []
    load_checkpoint = False
    if (load_checkpoint):
        agent.load_models()
        env.render(mode='human')
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
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






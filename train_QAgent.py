# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:07:27 2020

@author: RezaKakooee
"""

#%%
import numpy as np

from agents import QAgent
from grid_world_general_env import Environment
from visutils import plot_obs_history

#%%
def save_values(agent, env):
    print('Final Q-Table is: \n', agent.q_table)         
    np.save('q_table', agent.q_table)
    Q = agent.q_table.reshape((env.height, env.width, agent.action_size))
    
    V = np.zeros((env.height, env.width))
    for i in range(env.height):
        for j in range(env.width):
            V[i][j] = np.max(Q[i,j,:])
            
#%%
def main():
    env = Environment(default=5)
    agent = QAgent(env) 
    
    total_reward = []
    view_freq = 500
    n_episodes = 2400
    obs_history = {ep:[] for ep in range(n_episodes)}
    for ep in range(n_episodes):
        if ep % view_freq == 0:
            print('---------- The episode number is: ', ep)
        episode_reward = []
        state = env.reset()
        done = False
        t = 0
        while not done:
            t += 1
            obs_history[ep].append(state)
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            experience = (state, action, next_state, reward, done)
            agent.train(experience)
            episode_reward.append(reward)
            
            # experience_list = list(experience)
            # for str_a, int_a in env.actions.items():  
            #     if action == int_a:
            #         experience_list[1] = str_a
            # print('Experience: ', experience_list)
            
            # env.render()
            state = next_state
            
            if done:
                if ep % view_freq == 0:
                    print("Episode finished after {} timesteps".format(t))
                    print("Current epsilon is: ", agent.epsilon)
                    print('Episode Reward: {}'.format(np.sum(episode_reward)))
                break
        total_reward.append(np.sum(episode_reward))
    ###
    saving = True
    if saving:
        save_values(agent, env)
    return agent, env, obs_history

#%%
if __name__ == '__main__':
    agent, env, obs_history = main()
    plot_obs_history(env, obs_history)
    
        
    

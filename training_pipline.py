# -*- coding: utf-8 -*-
"""
Created on Mon May  4 06:31:30 2020

@author: RezaKakooee
"""
#%%
from grid_world_general_env import Environment
from agents import QAgent
from train_QAgent import train
from test_QAgent import test
import visutils

#%% Instantiate the Environment class
## There are two pre-defined puzzles inside the MdpMetaData class: a 5*5 and a 10*10 maze. Here we instantiate the Environment class with a 5*5 maze
env = Environment(default=5)

#%% Instantiate the agent
agent = QAgent(env) 

#%% Train Q-Learning Agent
q_table, total_reward, obs_history = train(env, agent, n_episodes=2400, render=False)

#%% Plot the total reward per episode
visutils.plot_reward(total_reward)

#%% Check the agent performance in some episodes
visutils.plot_obs_history(env, obs_history)

#%% Test the trained Q-Learning Agent
agent.q_table = q_table
test(env, agent, n_episodes=2, render=True)
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 21:58:47 2020

@author: RezaKakooee
"""

#%%
from gym import spaces
from gym.core import Env
from gym.utils import seeding

from mdp_meta_data import MdpMetaData
from plot_environment import PlotEnvironment

#%%
class Environment(Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self, default=None, n_rows=5, n_cols=5, 
                 inner_wall_coords=[[1,2],[2,2],[2,3],[2,4]], 
                 startX=3, startY=4, goalX=1, goalY=3):
        self.default = default
        self.mdp_metadate = MdpMetaData(default, n_rows, n_cols, inner_wall_coords, 
                                        startX, startY, goalX, goalY)
        
        self.height = self.mdp_metadate.n_rows
        self.width = self.mdp_metadate.n_cols

        self.action_space = spaces.Discrete(self.mdp_metadate.num_actions)
        self.observation_space = spaces.Discrete(self.mdp_metadate.num_states)
        self.actions = self.mdp_metadate.actions
        
        self.start_state = self.mdp_metadate.start_state
        self.goal_state = self.mdp_metadate.goal_state
        
        self.inner_wall = self.mdp_metadate.inner_wall
        
        self.T, self.R, self.P = self.mdp_metadate.make_mdp()

        self.S = self.mdp_metadate.goal_state
        
        self.view = PlotEnvironment(self.mdp_metadate.n_rows, self.mdp_metadate.n_cols, 
                                    self.start_state, self.goal_state, self.mdp_metadate.inner_wall)
        
        self.seed()
        self.reset()
    
    def reset(self):
        # self.reset_frame_counter()
        self.S = self.start_state
        return self.S
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def move(self, action):
        # print('current s: ', self.S)
        # print('T', self.T.iloc[self.S, action])
        self.new_S = self.T.iloc[self.S, action]
        # print('new s: ', new_S)
        return self.new_S
    
    def step(self, action):
        self.old_S = self.S
        self.S = self.move(action)
        if self.S == self.goal_state:
            reward = 0
            return self.S, reward, True, {}
        else:
            reward = -1
            return self.S, reward, False, {}
    
    def render(self, mode='human'):
        self.metadata["render.frame_counter"] = self.view.plot_map(self.old_S, self.new_S)

#%%
def main():
    from agents import Random_Agent
    env = Environment()
    agent = Random_Agent(env)
    
    n_episodes = 10
    state = env.reset()
    for i in range(n_episodes):
        action = agent.get_random_action(state)
        next_state, reward, done, info = env.step(action)
        experience = (state, action, next_state, reward, done)
        experience_list = list(experience)
        for str_a, int_a in env.actions.items():  
            if action == int_a:
                experience_list[1] = str_a
        
        print('Experience: ', experience_list)
        if not done:
            env.render()
            state = next_state
        else:
            state = env.reset()

#%%
if __name__ == '__main__':
    main()
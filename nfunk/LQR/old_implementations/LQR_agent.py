'''
Class which implements an agent, based on an "optimal" LQR policy
This LQR agent stabilizes the pendulum on top!
'''

import numpy as np

class LQR_agent:
    def __init__(self):
        #self.K = np.array([19.80210901,  6.03515527])
        #self.K = np.array([9.8469272, 2.60201742]) 
        #self.K = np.array([10.79086181, 2.84513479]) # 200Hz
        self.K = np.array([10.44231477, 2.75595235]) # 50Hz


    def __call__(self, obs):
        comm = 1
        theta = (np.arctan2(obs[1],obs[0]))
        theta_dot = obs[2]
        u = -(self.K[0]*theta+self.K[1]*theta_dot)

        return [u], comm, 0
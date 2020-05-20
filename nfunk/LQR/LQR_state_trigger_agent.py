'''
Class which implements an agent, based on an "optimal" LQR policy
This LQR agent stabilizes the pendulum on top!
'''

import numpy as np

class LQR_state_trigger_agent:
    # This is the implementation of a standart LQR agent -> always communicates
    def __init__(self):
        self.K = self.K = np.array([9.8469272, 2.60201742])
        self.thres = 2.0 * 0.017453292
        self.last_act = [0.0]

    def get_option(self,obs):
        # always returns 1 -> always communicate
        theta = (np.arctan2(obs[1],obs[0]))
        theta_dot = obs[2]
        if (True):
            # this represents the 2 norm scheduling
            theta = np.sqrt(theta**2+theta_dot**2)
        if (np.abs(theta)>self.thres):
            return 1
        else:
            return 0

    def act(self,obs,option):
        if (option==1):
            theta = (np.arctan2(obs[1],obs[0]))
            theta_dot = obs[2]
            u = [-(self.K[0]*theta+self.K[1]*theta_dot)]
            self.last_act = u

        return self.last_act

    def reset_last_act(self,obs):
        self.last_act = [0.0]
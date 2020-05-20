'''
Class which implements an agent, based on an "optimal" LQR policy
This LQR agent stabilizes the pendulum on top!
'''

import numpy as np

class LQR_state_diff_trigger_agent:
    # This is the implementation of a standart LQR agent -> always communicates
    def __init__(self):
        self.K = self.K = np.array([9.8469272, 2.60201742])
        self.thres = 0.75
        self.last_act = [0.0]
        self.last_state = [0.0,0.0]
        self.first = True

    def get_option(self,obs):
        if (self.first):
            self.first = False
            return 1
        # always returns 1 -> always communicate
        theta = (np.arctan2(obs[1],obs[0]))
        theta_dot = obs[2]
        theta_diff = self.last_state[0]-theta
        theta_dot_diff = self.last_state[1]-theta_dot
        if (np.sqrt(theta_diff**2+theta_dot_diff**2)<self.thres*np.sqrt(theta**2+theta_dot**2)):
                return 0
        else:
            return 1

    def act(self,obs,option):
        if (option==1):
            theta = (np.arctan2(obs[1],obs[0]))
            theta_dot = obs[2]
            u = [-(self.K[0]*theta+self.K[1]*theta_dot)]
            self.last_act = u
            self.last_state[0] = theta
            self.last_state[1] = theta_dot

        return self.last_act

    def reset_last_act(self,obs):
        self.first = True
        self.last_act = [0.0]
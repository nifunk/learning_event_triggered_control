# Copyright (c) 2020 Max Planck Gesellschaft

'''
Class which implements an agent, based on an "optimal" LQR policy
This LQR agent stabilizes the pendulum on top!
'''

import numpy as np

class LQR_input_trigger_agent:
    # This is the implementation of a standart LQR agent -> always communicates
    def __init__(self):
        self.K = self.K = np.array([9.8469272, 2.60201742])
        self.thres = 0.98
        self.last_act = [0.0]
        self.first = True

    def get_option(self,obs):
        if (self.first):
            self.first = False
            return 1
        # always returns 1 -> always communicate
        theta = (np.arctan2(obs[1],obs[0]))
        theta_dot = obs[2]
        curr_act = [-(self.K[0]*theta+self.K[1]*theta_dot)]
        if (abs(curr_act[0]-self.last_act[0])<self.thres*abs(curr_act[0])):
                return 0
        else:
            return 1

    def act(self,obs,option):
        if (option==1):
            theta = (np.arctan2(obs[1],obs[0]))
            theta_dot = obs[2]
            u = [-(self.K[0]*theta+self.K[1]*theta_dot)]
            self.last_act = u

        return self.last_act

    def reset_last_act(self,obs):
        # if self.first is set to true here then we denote this as the "first" mode
        # -> this means always communicate at the first spot!
        self.first = True
        act = 2*np.random.rand()-1
        # dont reset to completely zero as then the trigger does not work
        self.last_act = [0.01*act]
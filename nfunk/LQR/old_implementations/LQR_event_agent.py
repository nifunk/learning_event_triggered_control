'''
Class which implements an agent, based on an "optimal" LQR policy
This LQR agent stabilizes the pendulum on top!
'''

import numpy as np

class LQR_event_agent:
    def __init__(self):
        self.K = np.array([19.80210901,  6.03515527])   # used to be: [19.80210901,  6.03515527] 
        self.thres = 5 * 0.017453292 # in degrees () # was 5 deg
        self.prev_u = 0.0 # previous input applied, will be kept if no communication
        self.prev_comm = 0 #previous communitcation decision

    def __call__(self, obs):
        comm = 1    # assume communication is needed
        term = 0    # assume that current option does not terminata
        theta = (np.arctan2(obs[1],obs[0]))
        theta_dot = obs[2]
        if (abs(theta)<self.thres):  # this means no communication needed...
            u = self.prev_u
            comm = 0
        else:
            u = -(self.K[0]*theta+self.K[1]*theta_dot)
            self.prev_u = u
        if not(self.prev_comm==comm):
            term = 1
        self.prev_comm = comm

        return [u], comm, term
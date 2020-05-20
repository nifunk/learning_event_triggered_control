'''
Class which implements an agent, based on an "optimal" LQR policy
This LQR agent stabilizes the pendulum on top!
'''

import numpy as np

class LQR_standart_agent:
    # This is the implementation of a standart LQR agent -> always communicates
    def __init__(self):
        self.K = np.array([9.8469272, 2.60201742]) #20Hz
        #self.K = np.array([9.07370428, 2.39729361])  #10Hz
        #self.K = np.array([5.33746719, 0.68598507])  #1Hz
        #self.K = np.array([8.01220661, 2.09628154])  #5Hz
        #self.K = np.array([6.71241598, 1.61604534])  #2.5Hz
        #self.K = np.array([10.67033427, 2.81435885]) #100Hz  

        self.last_act = [0.0]

    def get_option(self,obs):
        # always returns 1 -> always communicate
        return 1

    def act(self,obs,option):
        theta = (np.arctan2(obs[1],obs[0]))
        theta_dot = obs[2]
        u = -(self.K[0]*theta+self.K[1]*theta_dot)

        return [u]

    def reset_last_act(self,obs):
        self.last_act = [0.0]
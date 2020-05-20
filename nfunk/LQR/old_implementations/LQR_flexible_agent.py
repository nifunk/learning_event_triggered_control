'''
Class which implements a more flexible LQR agent which offers various decision boundaries which we can choose from
This class is more or less explicitly made for the pendulum gym environment
'''

import numpy as np

class LQR_flexible_agent:
    def __init__(self,decision_fct, prev_obs, prev_act):
        '''
        initial observation, previous action
        '''
        #self.K = np.array([19.80210901,  6.03515527])
        #self.K = np.array([9.8469272, 2.60201742])    # 20Hz
        #self.K = np.array([10.79086181, 2.84513479]) # 200Hz
        self.K = np.array([10.44231477, 2.75595235]) # 50Hz
        #self.K = np.array([10.90333898, 2.87380076]) # 2000Hz


        self.dec_fct = decision_fct
        self.prev_obs = prev_obs
        self.prev_act = prev_act

    def __call__(self, observation, thresh):
        # Thres defines the threshold when one should communicate....
        comm = 1
        cont_comm = False
        cont_comm_val = 0.0

        # decide whether communication is needed or not:
        theta = (np.arctan2(observation[1],observation[0]))
        theta_dot = observation[2]

        theta_prev = (np.arctan2(self.prev_obs[1],self.prev_obs[0]))
        theta_dot_prev = self.prev_obs[2] 

        # action based on the current measurement
        curr_action = [-(self.K[0]*theta+self.K[1]*theta_dot)]

        comm_needed = True  # first assume communication is needed
        # NOW CONSIDER THE AVAILABLE DECISION FCTS ONE BY ONE:
        # input crit:
        if (self.dec_fct=='input_abs'):
            if (abs(curr_action[0]-self.prev_act[0])<thresh*abs(curr_action[0])):
                comm_needed = False
        elif (self.dec_fct=='input_abs_rev'):
            if (abs(curr_action[0]-self.prev_act[0])<thresh*abs(self.prev_act[0])):
                comm_needed = False
        # state crit:
        elif (self.dec_fct=='state_sqrt'):
            if (np.sqrt((theta-theta_prev) ** 2 + (theta_dot-theta_dot_prev) ** 2)<thresh*np.sqrt((theta) ** 2 + (theta_dot) ** 2)):
                comm_needed = False
        elif (self.dec_fct=='state_sqrt_rev'):
            if (np.sqrt((theta-theta_prev) ** 2 + (theta_dot-theta_dot_prev) ** 2)<thresh*np.sqrt((theta_prev) ** 2 + (theta_dot_prev) ** 2)):
                comm_needed = False
        # anglevel abs crit:
        elif (self.dec_fct=='angvel_abs'):
            if ((abs(theta/0.1) + abs(theta_dot/0.5))<thresh):  # this means no communication needed...
                comm_needed = False
        # anglevel sqrroot crit:
        elif (self.dec_fct=='angvel_sqrt'):
            if (np.sqrt((theta/0.1) ** 2 + (theta_dot/0.5) ** 2)<thresh):  # this means no communication needed...
                comm_needed = False
        elif (self.dec_fct=='angvel_sqrt_cont'):
            if (thresh>0.00001):
                cont_comm_val = (np.sqrt((theta/0.1) ** 2 + (theta_dot/0.5) ** 2))/thresh
            else: 
                cont_comm_val = 1.0
            cont_comm = True 
            if (np.sqrt((theta/0.1) ** 2 + (theta_dot/0.5) ** 2)<thresh):  # this means no communication needed...
                comm_needed = False
        elif (self.dec_fct=='ang_abs'):
        # angle crit
            if ((abs(theta/0.1))<thresh):  # this means no communication needed...
                comm_needed = False
        else:
            print ("Invalid decision function inside agent -> STOP THE CODE!!!!")

        if not(comm_needed):
            action = self.prev_act
            comm = 0
        else:
            # This means communication happens:
            action = curr_action
            self.prev_obs = observation
            self.prev_act = action
            comm = 1
        if (cont_comm):
            comm = cont_comm_val


        return action, comm, 0 # tells input, communication or not, if policy finished or not

    def set_initial_states(self,prev_obs,prev_act):
        self.prev_obs = prev_obs
        self.prev_act = prev_act
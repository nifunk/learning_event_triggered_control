import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import path
import cv2

# This class is a simple render to illustrate whether we communicate or not
class AddRender:
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0, **kwargs):
        self.num_envs=1 # just that it still runs with basic openai
        self.max_speed=12
        self.max_torque=2.
        self.dt=0.05#.02
        self.g = g
        self.viewer = None
        self.option = 0
        self.last_opt = 0

    def set_option(self,option):
        self.last_opt = self.option
        self.option = option


    def render_new(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            # file assumes that there is a folder called assets where pictures illsutrating communication or no communication are placed
            fname1 = "assets/1.png"
            self.img1 = rendering.Image(fname1, 2., 2.)
            fname2 = "assets/2.png"
            self.img2 = rendering.Image(fname2, 2., 2.)
            self.img1_raw = cv2.imread('assets/1.png',0)
            self.img0_raw = cv2.imread('assets/0.png',0)
            self.im_win = plt.imshow(self.img1_raw)
            fname0 = "assets/0.png"
            self.img0 = rendering.Image(fname0, 0., 0.)

        # Plot the picture depending on which option is chosen
        if (self.option==0):
            self.viewer.add_onetime(self.img0)
        elif (self.option==1):
            self.viewer.add_onetime(self.img1)
        else:
            self.viewer.add_onetime(self.img2)
        return self.viewer.render()


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

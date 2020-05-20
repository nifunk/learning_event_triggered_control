"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from os import path

class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self,g=10.0, **kwargs):
        self.max_speed=8. #angular of pole
        self.max_x=1. #Maybe need to set bigger!!!!
        self.max_xdot=30. #Maybe need to set bigger!!!!
        self.max_torque=10. #Maybe need to set bigger!!!!
        self.dt=.025 #was .05 
        self.viewer = None

        high = np.array([np.pi, self.max_speed, self.max_x, self.max_xdot]) #changed for 4dim
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high) #change?

        self.seed()

#was c_phi = 1, c_phidot = 0.1 
    #def init(self, c_x=0.75, c_phi=4, c_u=0.001, noise_level=1e-4, obs_noise=False, process_noise=False):
        #self.obs_noise = obs_noise
        self.obs_noise = True
        #self.process_noise = process_noise
        self.process_noise = True
        #self.noise_level = noise_level
        self.noise_level = 1e-2
        #self.c_u = c_u
        self.c_u = 0.001
        #self.c_x = c_x
        self.c_x = 0.75
        #self.c_phi = c_phi
        self.c_phi = 4

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot, x, xdot = self.state # th := theta  was th, thdot = self.state
	
#	end_cost = 0.
#	done = False	
#	if (th < -0.1) or (th > 0.1): #to stop the episode if th is big
#            done = True
#	    end_cost = 5000.

        Ip = 7.88e-003
        Mp = 0.230
        lp = 0.6413
        Bp = 0.0024
        g = 9.81

        Beq = 5.4
        Mc = 0.94
        
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        
        costs = self.c_phi * angle_normalize(th)**2 + self.c_x * (x**2) + self.c_u * (u**2) #+ end_cost # end_cost if stop episode


        A = -(Ip+Mp*lp**2)*Beq*xdot-(Mp**2*lp**3+Ip*Mp*lp)*np.sin(th)*(thdot)**2-Mp*lp*np.cos(th)*Bp*thdot  \
            +(Ip+Mp*lp**2)*u+Mp**2*lp**2*g*np.cos(th)*np.sin(th)
        B = (Mc+Mp)*Ip+Mc*Mp*lp**2+Mp**2*lp**2*np.sin((th)**2)
        
        C = (Mc+Mp)*Mp*g*lp*np.sin(th)-(Mc+Mp)*Bp*(thdot)-Mp**2*lp**2*np.sin(th)*np.cos(th)*(thdot)**2 \
            -Mp*lp*np.cos(th)*Beq*xdot+u*Mp*lp*np.cos(th)
        D = (Mc+Mp)*Ip+Mc*Mp*lp**2+Mp**2*lp**2*np.sin((th)**2)
        
        xdotdot = A/B
        thdotdot = C/D
        
        newthdot = thdot + thdotdot * dt
        newth = th + newthdot*dt
        newth = angle_normalize(newth)
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        
        newxdot = xdot + xdotdot * dt
        newx = x + newxdot*dt
        newxdot = np.clip(newxdot, -self.max_xdot, self.max_xdot)
#        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
#        newth = th + newthdot*dt
#        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        proc_noise = self.noise_level * np.random.randn(4, )

        if self.process_noise:
            self.state = np.array([newth, newthdot, newx, newxdot]) + proc_noise
        else:
            self.state = np.array([newth, newthdot, newx, newxdot])

        if (newth < -0.8) or (newth > 0.8): #to stop the episode if th is big
            return self._get_obs(), -3000, True, {}
#            done = True
#       end_cost = 5000.

        return self._get_obs(), -costs+1.0, False, {}

    def reset(self):
        s_init = [0, 0, 0, 0] + self.noise_level * np.random.randn(4, )
        # high = np.array([np.pi, 1])
        self.state = s_init
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        meas_noise = self.noise_level * np.random.randn(4, )
        if self.obs_noise:
            theta, thetadot, x, xdot = self.state + meas_noise
        else:
            theta, thetadot, x, xdot = self.state
        return np.array([theta, thetadot, x, xdot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

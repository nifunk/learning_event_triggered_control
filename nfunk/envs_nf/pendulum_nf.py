# Copyright (c) 2020 Max Planck Gesellschaft
# Modified the pendulum environment from https://github.com/openai/gym 
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class PendulumEnv(gym.Env):
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

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

        self.use_noise = True
        #self.use_noise = True 
        # Initialize a few variables:
        #self.rew_scale = kwargs['reward_scale']
        self.rew_scale = 1.0
        self.state_cost_scale = 1.0
        if (self.use_noise):
            self.obs_noise = 1e-4
            self.process_noise = 1e-4
            self.noise_level = 1e-4
        else:
            self.obs_noise = 0.0
            self.process_noise = 0.0
            self.noise_level = 0.0

        #self.start_top = bool(kwargs['env_start_top'])
        self.start_top = True
        #self.start_top = True
        #self.use_early_reset = bool(kwargs['env_early_reset'])
        self.use_early_reset = False
        self.on_top = True

        # Default cost values:
        self.c_phi = 1.0/self.state_cost_scale
        self.c_phidot = 0.1/self.state_cost_scale
        self.c_u = 0.1/self.state_cost_scale
        self.c_comm = 0.0   # comm penalty should be applied in main learning file   


    def seed(self, seed=None):
        seed_orig = seed
        self.np_random, seed = seeding.np_random(seed)
        # from here on was not there before:
        if (seed_orig is None):
            seed_no = 0
        else:
            seed_no = seed_orig

        np.random.seed(seed_no)
        return [seed]

    def step(self,u):
        comm = None
        if not((np.shape(u)) == self.action_space.shape):
            comm = u[1]
            u = u[0]
        #print (comm)

        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        #if (abs(u[0])>self.max_torque):
        #    print ("applied too much torque")
        orig_u = u[0]
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        
        # Adaptive cost function here:
        costs = self.c_phi * angle_normalize(th)**2 + self.c_phidot * thdot**2 + self.c_u * (orig_u**2)
        if not(comm is None):
            costs += self.c_comm * comm
        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        #newthdot = thdot + (3*g/(2*l) * th + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        if (abs(newthdot)>self.max_speed):
            print ("TOO FAST -> CLIPPING")
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        #if (abs(angle_normalize(newth))>15*0.0174533):
        #    print ("PENDULUM DOWN")
        #    print (th)

        if (abs(angle_normalize(newth))>45*0.0174533 and self.use_early_reset):
            if (self.on_top):
                print ("Pendulum fell down!")
            self.on_top = False

        proc_noise = self.process_noise * np.random.randn(2, )
        self.state = np.array([newth, newthdot]) + proc_noise

        # RETURN argument: observation, Kosten, Termination?, Dict um Werte hin und her zu geben!
        if not(self.on_top):
            return self._get_obs(), -3000, True, {}    #was 100 or other reward 10000
        else:
            # +0.1 was previously not there...
            return self._get_obs(), -costs*self.rew_scale, False, {} # in other rew setting +0.1

    def reset(self):
        # Previous version: start really anywhere, totally at random:
        #high = np.array([np.pi, 1])
        #self.state = self.np_random.uniform(low=-high, high=high)
        #self.last_u = None

        # NEW:
        if (self.start_top):
            #s_init = [0, 0] + self.noise_level * np.random.randn(2, )
            if (self.use_noise):
                s_init = [0, 0] + np.random.normal(0.0,1e-2,(2, )) # can also use e-1 -> more disturbed
            else:
                #s_init = [0.01, 0.0]
                s_init = [0, 0] + np.random.normal(0.0,1e-2,(2, ))
        else:
            s_init = [np.pi, 0] + self.noise_level * np.random.randn(2, )
            #s_init = [np.pi, 0] + np.random.normal(0.0,1e-2,(2, ))
            #high = np.array([np.pi, 1])
            #s_init = self.np_random.uniform(low=-high, high=high)

        self.state = s_init
        self.last_u = None
        self.on_top = True

        return self._get_obs()

    def _get_obs(self):
        meas_noise = self.obs_noise * np.random.randn(2, )
        theta, thetadot = self.state + meas_noise
        return np.array([np.cos(theta), np.sin(theta), thetadot])

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
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
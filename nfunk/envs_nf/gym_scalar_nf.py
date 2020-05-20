import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from math import exp

class GymScalarEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0, **kwargs):
        self.num_envs=1 # just that it still runs with basic openai
        self.max_speed=8
        self.max_torque=10.
        self.dt=.05 # equals 20 Hz
        self.g = g
        self.viewer = None

        high = np.array([10.])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()

        self.Ad = exp(self.dt)
        self.Bd = exp(self.dt)-1

        self.use_noise = True
        print ("--- Using own custom env ---")
        input ("WAAITT")
        # Initialize a few variables:
        #self.rew_scale = kwargs['reward_scale']
        self.rew_scale = 0.75

        if (self.use_noise):
            #self.obs_noise = kwargs['env_obs_noise']
            #self.process_noise = kwargs['env_process_noise']
            #self.noise_level = kwargs['env_noise_level']
            self.obs_noise = 5*1e-2
            self.process_noise = 5*1e-2
            self.noise_level = 5*1e-2
        else:
            self.obs_noise = 0.0
            self.process_noise = 0.0
            self.noise_level = 0.0

        # Default cost values:
        self.c_phi = 1.0
        self.c_phidot = 0.1
        self.c_u = 0.1
        self.c_comm = 1.0
        #if (kwargs['env_cost_mat'] is not None):
        #    split_itm = kwargs['env_cost_mat'].split(" ")
        #    self.c_phi = float(split_itm[0])
        #    self.c_phidot = float(split_itm[1])
        #    self.c_u = float(split_itm[2])
        #    self.c_comm = float(split_itm[3])

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

        th = self.state[0] # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        #if (abs(u[0])>self.max_torque):
        #    print ("applied too much torque")
                # deploy prestabilization:
        #u[0] = u[0] - 2 * th


        orig_u = u[0]
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        #print ("u: " + str(u))
        #print ("th: " + str(th))
        self.last_u = u # for rendering
        
        # Adaptive cost function here:
        costs = self.c_phi * (th)**2 + self.c_u * (orig_u**2)
        if not(comm is None):
            costs += self.c_comm * comm

        newth = self.Ad*th + self.Bd * u

        proc_noise = self.process_noise * np.random.randn(1, )
        self.state = np.array([newth]) + proc_noise

        if (abs(newth)>10):
            print ("Terminating earlier")
            return self._get_obs(), -1000, True, {}  # was -100            

        return self._get_obs(), -costs*self.rew_scale, False, {}    # was 0.1 

    def reset(self):

        s_init = -2*np.random.rand(1,)+1.0  # start between [-1,1]
        s_init = -4*np.random.rand(1,)+2.0  # start between [-1,1]

        self.state = s_init
        self.last_u = None
        self.on_top = True

        return self._get_obs()

    def _get_obs(self):
        meas_noise = self.obs_noise * np.random.randn(1, )
        theta = self.state + meas_noise
        return np.array(theta)

    def render(self, mode='human'):
        # TODO: maybe implement more nicely -> so far not working reasonably
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
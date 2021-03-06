# Copyright (c) 2020 Max Planck Gesellschaft

from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np
import pdb

# In this file the parametrization of the policies is implemented

def dense3D2(x, size, name, option, num_options=1, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [num_options, x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w[option[0]])
    if bias:
        b = tf.get_variable(name + "/b", [num_options,size], initializer=tf.zeros_initializer())
        return ret + b[option[0]]

    else:
        return ret


class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True, num_options=2,dc=0):
        assert isinstance(ob_space, gym.spaces.Box)

        # Define the dimensions
        self.ac_space_dim = ac_space.shape[0]
        self.ob_space_dim = ob_space.shape[0]
        self.dc = dc
        self.last_action = tf.zeros(ac_space.shape, dtype=tf.float32)
        self.last_action_init = tf.zeros(ac_space.shape, dtype=tf.float32)
        self.num_options = num_options
        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        option =  U.get_placeholder(name="option", dtype=tf.int32, shape=[None])

        # create a filter for the pure shape, meaning excluding u[k-1]
        obs_shape_pure = ((self.ob_space_dim - self.ac_space_dim),)

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)
        with tf.variable_scope("obfilter_pure"):
            self.ob_rms_only = RunningMeanStd(shape=obs_shape_pure)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        obz_pure = tf.clip_by_value((ob[:,:-self.ac_space_dim] - self.ob_rms_only.mean) / self.ob_rms_only.std, -5.0, 5.0)

        # implementation of the Q-funtion:
        last_out0 = obz # for option 0
        last_out1 = obz_pure # for option 1
        for i in range(num_hid_layers):
            last_out0 = tf.nn.tanh(U.dense(last_out0, hid_size, "vffc0%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            last_out1 = tf.nn.tanh(U.dense(last_out1, hid_size, "vffc1%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        last_out0 = U.dense(last_out0, 1, "vfff0", weight_init=U.normc_initializer(1.0))
        last_out1 = U.dense(last_out1, 1, "vfff1", weight_init=U.normc_initializer(1.0))


        # presents the value of (state,option) -> denoted as Q-fct in report        
        self.vpred = U.switch(option[0], last_out1, last_out0)[:,0]

        # Implementation of the policy over options:
        last_out0 = obz # for option 0
        last_out1 = obz_pure # for option 1
        for i in range(num_hid_layers):
            last_out0 = tf.nn.tanh(U.dense(last_out0, hid_size, "oppi0%i"%(i+1), weight_init=U.normc_initializer(1.0)))
            last_out1 = tf.nn.tanh(U.dense(last_out1, hid_size, "oppi1%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        last_out0 = U.dense(last_out0, 1, "oppif0", weight_init=U.normc_initializer(1.0))
        last_out1 = U.dense(last_out1, 1, "oppif1", weight_init=U.normc_initializer(1.0))
        last_out = tf.concat([last_out0, last_out1], 1)
        # this is the output of the policy over options:
        self.op_pi = tf.nn.softmax(last_out)

        self.tpred = tf.nn.sigmoid(dense3D2(tf.stop_gradient(last_out), 1, "termhead", option, num_options=num_options, weight_init=U.normc_initializer(1.0)))[:,0]
        
        # Always terminate
        termination_sample = tf.constant([True])
        

        # calculate the control action: -> implementation of intra option policy
        last_out = obz_pure
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense3D2(last_out, pdtype.param_shape()[0]//2, "polfinal", option, num_options=num_options, weight_init=U.normc_initializer(0.01),bias=False)
            mean = tf.nn.tanh(mean)
            logstd = tf.get_variable(name="logstd", shape=[num_options, 1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = U.concatenate([mean, mean * 0.0 + logstd[option[0]]], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))


        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # if stochastic is true, we sample around the mean, this corresponds to the exploration at the action level
        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        
        # determine the control action to be applied. In case of ZOH == opt 0 just use u[k-1]
        ac = U.switch(option[0], ac, tf.stop_gradient(ob[:,-self.ac_space_dim:]))
        ac = tf.clip_by_value(ac,-1.0,1.0)

        self.last_action = tf.stop_gradient(ac)
        self._act = U.function([stochastic, ob, option], [ac, self.vpred, last_out, logstd])

        self._get_v = U.function([ob, option], [self.vpred])
        self.get_term = U.function([ob, option], [termination_sample])
        self.get_tpred = U.function([ob, option], [self.tpred])
        self.get_vpred = U.function([ob, option], [self.vpred])        
        self._get_op = U.function([ob], [self.op_pi])


    def act(self, stochastic, ob, option):
        # function returns, action and Q-fct prediction
        ac1, vpred1, feats, logstd =  self._act(stochastic, ob[None], [option])
        return ac1[0], vpred1[0], feats[0], logstd[option][0]


    def get_option(self,ob):
        # function which returns which option is to be executed
        op_prob = self._get_op([ob])[0][0]
        return np.random.choice(range(len(op_prob)), p=op_prob)


    def get_term_adv(self, ob, curr_opt):
        # this function calculates the terminal advantage. Not needed as we always terminate...
        vals = []
        for opt in range(self.num_options):
            vals.append(self._get_v(ob,[opt])[0])

        vals=np.array(vals)
        op_prob = self._get_op(ob)[0].transpose()
        return (vals[curr_opt[0]] - np.sum((op_prob*vals),axis=0) + self.dc),  ( vals[curr_opt[0]] - np.sum((op_prob*vals),axis=0) )


    def get_opt_adv(self, ob, curr_opt):
        # This function returns the advantage over the options, where we use the maximum Q-value
        vals = []
        for opt in range(self.num_options):
            vals.append(self._get_v(ob,[opt])[0])

        vals=np.array(vals)
        # choose max value as reference:
        vals_max = np.amax(vals,axis=0)

        return ((vals[curr_opt[0]] - vals_max+ self.dc ),  (vals[curr_opt[0]] - vals_max) )

    def get_opt_adv_oldpi(self, ob, curr_opt, oldpi):
        # This function returns the advantage over the options, where we use the max value as the reference
        vals = []
        for opt in range(self.num_options):
            vals.append(self._get_v(ob,[opt])[0])

        vals=np.array(vals)
        # choose max value as reference:
        vals_max = np.amax(vals,axis=0)

        return ((vals[curr_opt[0]] - vals_max+ self.dc ),  (vals[curr_opt[0]] - vals_max) )


    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

    def reset_last_act(self):
        # in case of starting a new rollout, set u[k-1] to 0,...
        self.last_action = self.last_action_init
        return self.last_action


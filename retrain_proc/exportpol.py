# Copyright (c) 2020 Max Planck Gesellschaft
from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import pdb
import os
import shutil
from scipy import spatial
import gym
import pickle as pkl
import matplotlib.pyplot as plt
from gym import spaces
from utils.writeNNet import writeNNet


def do_export(yolo,env,pi):
    weigh_list = [] # this denotes the weights of the policy (control policy!)
    bias_list = []  # this denotes the bias of the control policy
    pol_ov_opt_w_0 = [] # weights for propability of choosing opt 0
    pol_ov_opt_b_0 = []
    pol_ov_opt_w_1 = [] # weights for propability of choosing opt 1
    pol_ov_opt_b_1 = []


    for v in yolo:
        if ('pol' in v.name):
            if ('final' in v.name):
                abc = tf.transpose(v[1,:,:]).eval()
                abc = np.concatenate((abc,-1*abc),axis=0)
                weigh_list.append(abc)
                bias_list.append(np.zeros(np.shape(abc)[0]))
            elif ('w' in v.name):
                weigh_list.append((tf.transpose(v).eval()))
            else:
                bias_list.append((tf.transpose(v).eval()))

        # search for pol over options parameter
        if ('oppi' in v.name):
            if ('f' in v.name):
                # this means that we are dealing with the final layer:
                if ('w' in v.name):
                    # here we are dealing with weights
                    abc = tf.transpose(v).eval()
                    #abc = abc[:32,:]
                    abc = np.concatenate((abc,-1*abc),axis=0)

                    if ('f0' in v.name):
                        pol_ov_opt_w_0.append(abc)
                    else:
                        pol_ov_opt_w_1.append(abc)

                else:
                    # here we are dealing with bias
                    abc = tf.transpose(v).eval()
                    #abc = abc[:32]
                    abc = np.concatenate((abc,-1*abc),axis=0)

                    if ('f0' in v.name):
                        pol_ov_opt_b_0.append(abc)
                    else:
                        pol_ov_opt_b_1.append(abc)

            else:
                if ('w' in v.name):
                    # here we are dealing with weights
                    abc = tf.transpose(v).eval()
                    #abc = abc[:32,:]
                    if ('i0' in v.name):
                        pol_ov_opt_w_0.append(abc)
                    else:
                        pol_ov_opt_w_1.append(abc)

                else:
                    # here we are dealing with bias
                    abc = tf.transpose(v).eval()
                    #abc = abc[:32]
                    if ('i0' in v.name):
                        pol_ov_opt_b_0.append(abc)
                    else:
                        pol_ov_opt_b_1.append(abc)


    # here manually the clipping is added for the control network
    # This is not needed for the policy that outputs the probabilities
    if (True):
        again = np.zeros((2,2))
        again[0,0] = -1
        again[1,0] = -1
        again[0,1] = 1
        again[1,1] = 1
        again_b = np.zeros((2))
        again_b[0] = 1
        again_b[1] = -1
        weigh_list.append(again)
        bias_list.append(again_b)
        again = np.zeros((2,2)) # on top is the right, bottom is the inverted sign
        again[0,0] = -1
        again[0,1] = 1
        again[1,0] = 1
        again[1,1] = -1
        again_b = np.zeros((2))
        again_b[0] = 1
        again_b[1] = -1
        weigh_list.append(again)
        bias_list.append(again_b)

    # as the control network initially only takes 3 inputs correct this:
    zero_layer = np.zeros((np.shape(weigh_list[0])[0],1))
    weigh_list[0] = np.concatenate((weigh_list[0],zero_layer),axis=1)
    # also correct for this with the decision for communicating:
    zero_layer = np.zeros((np.shape(pol_ov_opt_w_1[0])[0],1))
    pol_ov_opt_w_1[0] = np.concatenate((pol_ov_opt_w_1[0],zero_layer),axis=1)
    


    # there should be 4 input values (cos th, sin th, th dot, u(k-1))
    pend = True
    inMins = []
    inMins.append(-5.0)
    if (pend):
        inMins.append(-5.0)
        inMins.append(-5.0)
        inMins.append(-1.0)
    inMaxs = []
    inMaxs.append(5.0)
    if (pend):
        inMaxs.append(5.0)
        inMaxs.append(5.0)
        inMaxs.append(1.0)
    inmeans = []
    means_from_torch = pi.ob_rms.mean.eval()
    #print (means_from_torch)
    
    inmeans.append(means_from_torch[0])#inmeans.append(0.0)
    if (pend):
        inmeans.append(means_from_torch[1])#inmeans.append(0.0)
        inmeans.append(means_from_torch[2])#inmeans.append(0.0)
    inmeans.append(means_from_torch[3])#inmeans.append(0.0)
    inmeans.append(0.0) # for the output
 
    inranges = []
    std_from_torch = pi.ob_rms.std.eval()
    print (std_from_torch)
    inranges.append(std_from_torch[0])#inranges.append(1.0)
    if (pend):
        inranges.append(std_from_torch[1])#inranges.append(1.0)
        inranges.append(std_from_torch[2])#inranges.append(1.0)
    inranges.append(std_from_torch[3])#inranges.append(1.0)
    inranges.append(1.0)

    # idea: dump all the needed files into a pickle object that can later be processed efficiently then
    if (os.path.exists("NN_retrain_analysis.pkl")):
        os.remove("NN_retrain_analysis.pkl")
    f = open("NN_retrain_analysis.pkl", "wb")
    print (len(weigh_list))
    pkl.dump(weigh_list, f)
    print (len(bias_list))
    pkl.dump(bias_list, f)
    pkl.dump(inMins, f)
    pkl.dump(inMaxs, f)
    pkl.dump(inmeans, f)
    pkl.dump(inranges, f)
    pkl.dump(pi.ob_rms.mean.eval(),f)
    pkl.dump(pi.ob_rms.std.eval(),f)
    pkl.dump(pi.ob_rms_only.mean.eval(),f)
    pkl.dump(pi.ob_rms_only.std.eval(),f)
    pkl.dump(env.action_space.low,f)
    pkl.dump(env.action_space.high,f)
    
    # additionally write the propabilities:
    pkl.dump(pol_ov_opt_w_0,f)
    pkl.dump(pol_ov_opt_b_0,f)
    pkl.dump(pol_ov_opt_w_1,f)
    pkl.dump(pol_ov_opt_b_1,f)


    f.close()
    print ("raw data sucessfully exported")


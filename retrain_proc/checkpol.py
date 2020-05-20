# Copyright (c) 2020 Max Planck Gesellschaft
import sys
import warnings
import numpy as np
import pickle as pkl
from utils.writeNNet import writeNNet
from utils.readNNet import readNNet
from utils.normalizeNNet import normalizeNNet
import copy
import time
import os

# Path to Marabou folder if you did not export it
sys.path.append('/home/nfunk/Code_MA/nn_veri/Marabou')


def default_block(last_bias,final=False):
	# this function builds a default block such that we can "elongate" a certain output
	# the last bias block has to be passed to have an idea about the dimenstion
	dim = np.shape(last_bias)[0]
	out_dim = dim
	if (final):
		out_dim = int(dim/2)
	ws = np.zeros((out_dim,dim))
	bs = np.zeros((out_dim))
	curr = 1.0
	fac = 1.0
	if (final):
		fac = -1.0
	if not(final):
		for i in range(out_dim):
			ws[i,i] = curr
			curr = curr*fac
	else:
		j = 0
		for i in range(out_dim):
			ws[i,j] = curr
			curr = curr*fac
			j += 1
			ws[i,j] = curr
			curr = curr*fac
			j += 1

	return ws,bs

def special_concat(weights,add_w,bias,add_b):
	dim_orig = np.shape(weights)
	dim_add = np.shape(add_w)
	dim_complete = []
	for i in range(len(dim_orig)):
		dim_complete.append(dim_orig[i]+dim_add[i])
	fused_w = np.zeros((dim_complete))
	fused_w[:dim_orig[0],:dim_orig[1]] = weights
	fused_w[dim_orig[0]:,dim_orig[1]:] = add_w

	fused_b = np.concatenate((bias,add_b))

	return fused_w,fused_b

def identity_block(dim):
	# this one returns an identity block such that numbers can be fed through the network
	ws = np.zeros((2*dim,dim))
	bs = np.zeros((2*dim))
	j = 0
	num = 1
	fac = -1
	for i in range (dim):
		ws[j,i]=num
		num = num*fac
		j += 1
		ws[j,i]=num
		num = num*fac
		j += 1

	return ws,bs

def identity_block_denormalize(dim,norm_w,norm_b):
	# this one returns an identity block such that numbers can be fed through the network
	ws = np.zeros((2*dim,dim))
	bs = np.zeros((2*dim))
	j = 0
	num = 1
	fac = -1
	for i in range (dim):
		ws[j,i]=num*norm_w[i]
		bs[j] = num*norm_b[i]
		num = num*fac
		j += 1
		ws[j,i]=num*norm_w[i]
		bs[j] = num*norm_b[i]
		num = num*fac
		j += 1

	return ws,bs


def sys_dynamics_block(val,idx,prev_layer):
	#assumes that this block is placed in the last layer
	# (Reason: Then no ReLU is applied in the last layer)
	ws = np.zeros((1,np.shape(prev_layer)[0]))
	bs = np.zeros((1))
	for i in range(len(val)):
		ws[0,idx[i]] = val[i]
		ws[0,idx[i]+1] = -val[i]

	return ws,bs

from maraboupy import Marabou

# This function again connects the network with the system dynamics
def build_correct():
	# First step: load pkl file
	f = open("NN_retrain_analysis.pkl","rb")
	weigh_list = pkl.load(f)
	bias_list = pkl.load(f)
	inMins = pkl.load(f)
	inMaxs = pkl.load(f)
	inmeans = pkl.load(f)
	inranges =pkl.load(f)
	ob_mean = pkl.load(f)
	ob_std = pkl.load(f)
	ob_only_mean = pkl.load(f)
	ob_only_rms = pkl.load(f)
	min_action = pkl.load(f)
	max_action = pkl.load(f)

	prop_o0_w = pkl.load(f) 
	prop_o0_b = pkl.load(f) 
	prop_o1_w = pkl.load(f) 
	prop_o1_b = pkl.load(f) 

	print ("Pickle file loaded succesfully")



	# ------------------------------------- TEST STANDART FUNCTIONALITY ---------------------------------------------------------
	#REMARK: those files are not really needed in the end:
	if (False):
		# write and normalize NNet -> very important
		writeNNet(weigh_list,bias_list,inMins,inMaxs,inmeans,inranges,'standart.nnet')
		normalizeNNet('standart.nnet', 'standart.nnet')

		writeNNet(prop_o0_w,prop_o0_b,inMins,inMaxs,inmeans,inranges,'standart_o0.nnet')
		normalizeNNet('standart_o0.nnet', 'standart_o0.nnet')

		writeNNet(prop_o1_w,prop_o1_b,inMins,inMaxs,inmeans,inranges,'standart_o1.nnet')
		normalizeNNet('standart_o1.nnet', 'standart_o1.nnet')

		if (False):
			# Stuff here seems to behave as expected :)
			# Sanity check woth the new one
			print ("Check control network:")
			netcheck = Marabou.read_nnet('standart.nnet')
			cos_th = 1.0
			sin_th = -0.07625
			th_dot = -0.02785
			netcheck.setLowerBound(netcheck.inputVars[0][0], cos_th)
			netcheck.setUpperBound(netcheck.inputVars[0][0], cos_th)
			netcheck.setLowerBound(netcheck.inputVars[0][1], sin_th)
			netcheck.setUpperBound(netcheck.inputVars[0][1], sin_th)
			netcheck.setLowerBound(netcheck.inputVars[0][2], th_dot)
			netcheck.setUpperBound(netcheck.inputVars[0][2], th_dot)
			vals1, stats1 = netcheck.solve()
			print (inmeans)
			input ("TOP SIMPLEST CHECK CTRL POLICY,...")

			print ("Check option 0 network:")
			netcheck = Marabou.read_nnet('standart_o0.nnet')
			cos_th = 1.0
			sin_th = -0.07625
			th_dot = -0.02785
			u_k_prev = 0.0
			netcheck.setLowerBound(netcheck.inputVars[0][0], cos_th)
			netcheck.setUpperBound(netcheck.inputVars[0][0], cos_th)
			netcheck.setLowerBound(netcheck.inputVars[0][1], sin_th)
			netcheck.setUpperBound(netcheck.inputVars[0][1], sin_th)
			netcheck.setLowerBound(netcheck.inputVars[0][2], th_dot)
			netcheck.setUpperBound(netcheck.inputVars[0][2], th_dot)
			netcheck.setLowerBound(netcheck.inputVars[0][3], u_k_prev)
			netcheck.setUpperBound(netcheck.inputVars[0][3], u_k_prev)
			vals1, stats1 = netcheck.solve()
			print (inmeans)
			input ("TOP SIMPLEST CHECK o0 POLICY,...")

			print ("Check option 1 network:")
			netcheck = Marabou.read_nnet('standart_o1.nnet')
			cos_th = 1.0
			sin_th = -0.07625
			th_dot = -0.02785
			u_k_prev = 0.0
			netcheck.setLowerBound(netcheck.inputVars[0][0], cos_th)
			netcheck.setUpperBound(netcheck.inputVars[0][0], cos_th)
			netcheck.setLowerBound(netcheck.inputVars[0][1], sin_th)
			netcheck.setUpperBound(netcheck.inputVars[0][1], sin_th)
			netcheck.setLowerBound(netcheck.inputVars[0][2], th_dot)
			netcheck.setUpperBound(netcheck.inputVars[0][2], th_dot)
			netcheck.setLowerBound(netcheck.inputVars[0][3], u_k_prev)
			netcheck.setUpperBound(netcheck.inputVars[0][3], u_k_prev)
			vals1, stats1 = netcheck.solve()
			print (inmeans)
			input ("TOP SIMPLEST CHECK o1 POLICY,...")


	# ------------------------------------- END TEST STANDART FUNCTIONALITY ---------------------------------------------------------


	# ------------------------------------- STACK THE BLOCKS ON TOP OF EACH OTHER ---------------------------------------------------------
	# This corresponds to STAGE 1
	layers_ctrl_pol = len(weigh_list)
	layers_opt_0 = len(prop_o0_w)
	layers_opt_1 = len(prop_o1_w)


	complete_w = []
	complete_b = []
	len_final = (np.max([layers_ctrl_pol,layers_opt_0,layers_opt_1])) + 1 # add one final layer that everything is added
	for i in range (len_final):
		final_layer = False
		if (i==len_final-1):
			final_layer = True

		# add the control policy:
		if (i<layers_ctrl_pol):
			ws = weigh_list[i]
			bs = bias_list[i]

		else:
			ws,bs = default_block(bias_list[-1],final=final_layer)

		# add the layer for propability of opt0
		if (i<layers_opt_0):
			w_to_add = prop_o0_w[i]
			b_to_add = prop_o0_b[i]
		else:
			w_to_add,b_to_add = default_block(prop_o0_b[-1],final=final_layer)
		if (i==0):
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)
		else:
			ws,bs = special_concat(ws,w_to_add,bs,b_to_add)

		# add the layer for propability of opt1
		if (i<layers_opt_1):
			w_to_add = prop_o1_w[i]
			b_to_add = prop_o1_b[i]
		else:
			w_to_add,b_to_add = default_block(prop_o1_b[-1],final=final_layer)
		if (i==0):
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)
		else:
			ws,bs = special_concat(ws,w_to_add,bs,b_to_add)

		# add identity layer to push everything through:
		if (i==0):
			#w_to_add,b_to_add = identity_block(4)
			w_to_add,b_to_add = identity_block_denormalize(4,inranges,inmeans)
		else:
			# just to generate the block,...
			#w_to_add,b_to_add = identity_block(4)
			w_to_add,b_to_add = identity_block_denormalize(4,inranges,inmeans)
			w_to_add,b_to_add = default_block(b_to_add,final=final_layer)
		if (i==0):
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)
		else:
			ws,bs = special_concat(ws,w_to_add,bs,b_to_add)


		complete_w.append(ws)
		complete_b.append(bs)
		#print (np.shape(ws))
		#print (np.shape(bs))
		#input ("interm stop")

	if os.path.exists('standart_all_stage1.nnet'):
		os.remove('standart_all_stage1.nnet')
	writeNNet(complete_w,complete_b,inMins,inMaxs,inmeans,inranges,'standart_all_stage1.nnet')
	normalizeNNet('standart_all_stage1.nnet', 'standart_all_stage1.nnet')

	# TEST:
	if (False):
		print ("Check all network:")
		netcheck = Marabou.read_nnet('standart_all_stage1.nnet')
		cos_th = 1.0
		sin_th = -0.07625
		th_dot = -0.02785
		u_k_prev = 0.0
		netcheck.setLowerBound(netcheck.inputVars[0][0], cos_th)
		netcheck.setUpperBound(netcheck.inputVars[0][0], cos_th)
		netcheck.setLowerBound(netcheck.inputVars[0][1], sin_th)
		netcheck.setUpperBound(netcheck.inputVars[0][1], sin_th)
		netcheck.setLowerBound(netcheck.inputVars[0][2], th_dot)
		netcheck.setUpperBound(netcheck.inputVars[0][2], th_dot)
		netcheck.setLowerBound(netcheck.inputVars[0][3], u_k_prev)
		netcheck.setUpperBound(netcheck.inputVars[0][3], u_k_prev)
		vals1, stats1 = netcheck.solve()
		input ("TOP SIMPLEST CHECK all,...")

	# ------------------------------------- END STACK THE BLOCKS ON TOP OF EACH OTHER ---------------------------------------------------------



	# ------------------------------------- STACK THE BLOCKS ON TOP OF EACH OTHER ALSO ADD SYS DYN---------------------------------------------------------
	# This is now Stage 1 and 2 combined
	layers_ctrl_pol = len(weigh_list)
	layers_opt_0 = len(prop_o0_w)
	layers_opt_1 = len(prop_o1_w)


	complete_w = []
	complete_b = []
	len_final = (np.max([layers_ctrl_pol,layers_opt_0,layers_opt_1])) + 2 # add one final layer that everything is added
	for i in range (len_final):
		final_layer = False
		if (i==len_final-1):
			final_layer = True

		# add the control policy:
		if (i<layers_ctrl_pol):
			ws = weigh_list[i]
			bs = bias_list[i]

		else:
			ws,bs = default_block(bias_list[-1],final=final_layer)

		# add the layer for propability of opt0
		if (i<layers_opt_0):
			w_to_add = prop_o0_w[i]
			b_to_add = prop_o0_b[i]
		else:
			w_to_add,b_to_add = default_block(prop_o0_b[-1],final=final_layer)
		if (i==0):
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)
		else:
			ws,bs = special_concat(ws,w_to_add,bs,b_to_add)

		# add the layer for propability of opt1
		if (i<layers_opt_1):
			w_to_add = prop_o1_w[i]
			b_to_add = prop_o1_b[i]
		else:
			w_to_add,b_to_add = default_block(prop_o1_b[-1],final=final_layer)
		if (i==0):
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)
		else:
			ws,bs = special_concat(ws,w_to_add,bs,b_to_add)

		# add identity layer to push everything through:
		if (i==0):
			#w_to_add,b_to_add = identity_block(4)
			w_to_add,b_to_add = identity_block_denormalize(4,inranges,inmeans)
		else:
			# just to generate the block,...
			#w_to_add,b_to_add = identity_block(4)
			w_to_add,b_to_add = identity_block_denormalize(4,inranges,inmeans)
			w_to_add,b_to_add = default_block(b_to_add,final=final_layer)
		if (i==0):
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)
		else:
			ws,bs = special_concat(ws,w_to_add,bs,b_to_add)

		if (final_layer):
			# now add the evolution of the system dynamics:
			ang_dyn = []
			ang_dyn.append(1.01881)
			ang_dyn.append(0.0503131)
			ang_dyn.append(0.0503131*3*max_action*0.05)

			idx = []
			idx.append(8)
			idx.append(10)
			idx.append(12)

			ang_vel_dyn = []
			ang_vel_dyn.append(0.754696)
			ang_vel_dyn.append(1.01881)
			ang_vel_dyn.append(1.01881*3*max_action*0.05)

			# To add: next angle and next angvel when using the control policy:
			w_to_add, b_to_add = sys_dynamics_block(ang_dyn,idx,complete_w[-1])
			#print (np.shape(ws))
			#print (np.shape(w_to_add))
			#input ("WAIT")
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)
			w_to_add, b_to_add = sys_dynamics_block(ang_vel_dyn,idx,complete_w[-1])
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)

			idx = []
			idx.append(8)
			idx.append(10)
			idx.append(0)

			# To add: next angle and angvel when using the zero order hold:
			w_to_add, b_to_add = sys_dynamics_block(ang_dyn,idx,complete_w[-1])
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)
			w_to_add, b_to_add = sys_dynamics_block(ang_vel_dyn,idx,complete_w[-1])
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)


			idx = []
			idx.append(2)
			idx.append(4)
			subs = []
			subs.append(1)
			subs.append(-1)
			# P(comm=0) - P(comm=1)
			w_to_add, b_to_add = sys_dynamics_block(subs,idx,complete_w[-1])
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)

			idx = []
			idx.append(4)
			idx.append(2)
			# P(comm=1) - P(comm=0)
			w_to_add, b_to_add = sys_dynamics_block(subs,idx,complete_w[-1])
			ws = np.concatenate((ws,w_to_add),axis=0)
			bs = np.concatenate((bs,b_to_add),axis=0)


		complete_w.append(ws)
		complete_b.append(bs)

	if (os.path.exists('standart_all_comp.nnet')):
		os.remove('standart_all_comp.nnet')
	writeNNet(complete_w,complete_b,inMins,inMaxs,inmeans,inranges,'standart_all_comp.nnet')
	normalizeNNet('standart_all_comp.nnet', 'standart_all_comp.nnet')

	# TEST:
	if (False):
		print ("Check all network:")
		netcheck = Marabou.read_nnet('standart_all.nnet')
		cos_th = 1.0
		sin_th = 0.0#-0.07625
		th_dot = 0.0#-0.02785
		u_k_prev = 0.0
		netcheck.setLowerBound(netcheck.inputVars[0][0], cos_th)
		netcheck.setUpperBound(netcheck.inputVars[0][0], cos_th)
		netcheck.setLowerBound(netcheck.inputVars[0][1], sin_th)
		netcheck.setUpperBound(netcheck.inputVars[0][1], sin_th)
		netcheck.setLowerBound(netcheck.inputVars[0][2], th_dot)
		netcheck.setUpperBound(netcheck.inputVars[0][2], th_dot)
		netcheck.setLowerBound(netcheck.inputVars[0][3], u_k_prev)
		netcheck.setUpperBound(netcheck.inputVars[0][3], u_k_prev)
		vals1, stats1 = netcheck.solve()
		input ("TOP SIMPLEST CHECK COMPLETE...")

	print ("Completed rewriting")


def input_check(ang,angvel,file):
	# This file should simplify to accound all the input stuff already:
	netcheck = Marabou.read_nnet(file)
	# cos th: we assume this to be always 1.0
	netcheck.setLowerBound(netcheck.inputVars[0][0], 1.0)
	netcheck.setUpperBound(netcheck.inputVars[0][0], 1.0)
	# sin th: we assume that the small angle approximation holds
	netcheck.setLowerBound(netcheck.inputVars[0][1], -ang)
	netcheck.setUpperBound(netcheck.inputVars[0][1], ang)
	# angular velocity:
	netcheck.setLowerBound(netcheck.inputVars[0][2], -angvel)
	netcheck.setUpperBound(netcheck.inputVars[0][2], angvel)
	# input: is normalized in between 1 and -1:
	netcheck.setLowerBound(netcheck.inputVars[0][3], -1.0)
	netcheck.setUpperBound(netcheck.inputVars[0][3], 1.0)

	return netcheck

def try_to_find_input_check(ang,angvel,file):
	# This file should simplify to accound all the input stuff already:
	netcheck = Marabou.read_nnet(file)
	# cos th: we assume this to be always 1.0
	netcheck.setLowerBound(netcheck.inputVars[0][0], 1.0)
	netcheck.setUpperBound(netcheck.inputVars[0][0], 1.0)
	# sin th: we assume that the small angle approximation holds
	netcheck.setLowerBound(netcheck.inputVars[0][1], ang)
	netcheck.setUpperBound(netcheck.inputVars[0][1], ang)
	# angular velocity:
	netcheck.setLowerBound(netcheck.inputVars[0][2], angvel)
	netcheck.setUpperBound(netcheck.inputVars[0][2], angvel)
	# input: is normalized in between 1 and -1:
	netcheck.setLowerBound(netcheck.inputVars[0][3], -1.0)
	netcheck.setUpperBound(netcheck.inputVars[0][3], 1.0)

	return netcheck

def try_to_find_input_comp_check(ang,angvel,uk,file):
	# This file should simplify to accound all the input stuff already:
	netcheck = Marabou.read_nnet(file)
	# cos th: we assume this to be always 1.0
	netcheck.setLowerBound(netcheck.inputVars[0][0], 1.0)
	netcheck.setUpperBound(netcheck.inputVars[0][0], 1.0)
	# sin th: we assume that the small angle approximation holds
	netcheck.setLowerBound(netcheck.inputVars[0][1], ang)
	netcheck.setUpperBound(netcheck.inputVars[0][1], ang)
	# angular velocity:
	netcheck.setLowerBound(netcheck.inputVars[0][2], angvel)
	netcheck.setUpperBound(netcheck.inputVars[0][2], angvel)
	# input: is normalized in between 1 and -1:
	netcheck.setLowerBound(netcheck.inputVars[0][3], uk)
	netcheck.setUpperBound(netcheck.inputVars[0][3], uk)

	return netcheck

# this function checks the network for stability
def check_whole(nnet_file_name,ang,ang_save,angvel,angvel_save):
	unsuccessful_points = []


	if (True):
		# Before starting make the etreme check:
		netcheck = try_to_find_input_check(ang,angvel,nnet_file_name)
		#netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("BEGINNING: (POS) IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")

		# Before starting make the etreme check to the other direction
		netcheck = try_to_find_input_check(-ang,-angvel,nnet_file_name)
		#netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("BEGINNING: (NEG) IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")

	ang_lb_comm = False
	ang_lb_nocomm = False
	ang_rb_comm = False
	ang_rb_nocomm = False

	angvel_top_comm = False
	angvel_top_nocomm = False
	angvel_bot_comm = False
	angvel_bot_nocomm = False

	# first check angle to the LEFT: ------------------------------------------------------------------------
	netcheck = input_check(ang,angvel,nnet_file_name)
	# no communication:
	netcheck.setLowerBound(netcheck.outputVars[0][11],0.0)
	netcheck.setUpperBound(netcheck.outputVars[0][7],-ang) 
	vals1, stats1 = netcheck.solve()
	del netcheck
	if (vals1=={}):
		print ("Found left bound for no communication")
		ang_lb_nocomm = True
	else:
		print ("left bound unsafe for no communication")
		# try to find input that recovers system
		rem = vals1[3] #this is the old action
		netcheck = try_to_find_input_check(vals1[1],vals1[2],nnet_file_name)
		#netcheck.setLowerBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")
		else:
			vals1[4] = rem
			unsuccessful_points.append(vals1)

	netcheck = input_check(ang,angvel,nnet_file_name)
	# communication:
	netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
	netcheck.setUpperBound(netcheck.outputVars[0][9],-ang)
	vals1, stats1 = netcheck.solve()
	del netcheck
	if (vals1=={}):
		print ("Found left bound for communication")
		ang_lb_comm = True
	else:
		print ("left bound unsafe for communication")
		# try to find new control action recovering the system
		rem = vals1[3] #this is the old action
		netcheck = try_to_find_input_check(vals1[1],vals1[2],nnet_file_name)
		#netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		time.sleep(5.0)
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")
		else:
			vals1[4] = rem
			#print (vals1[3])
			#print (vals1[4])
			#input ("SCHDOOP")
			unsuccessful_points.append(vals1)


	# then check angle to the RIGHT: -----------------------------------------------------------------------
	netcheck = input_check(ang,angvel,nnet_file_name)
	# no communication:
	netcheck.setLowerBound(netcheck.outputVars[0][11],0.0)
	netcheck.setLowerBound(netcheck.outputVars[0][7],ang)
	vals1, stats1 = netcheck.solve()
	del netcheck
	if (vals1=={}):
		print ("Found right bound for no communication")
		ang_rb_nocomm = True
	else:
		print ("right bound unsafe for no communication")
		rem = vals1[3]
		netcheck = try_to_find_input_check(vals1[1],vals1[2],nnet_file_name)
		#netcheck.setLowerBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")
		else:
			vals1[4] = rem
			unsuccessful_points.append(vals1)


	netcheck = input_check(ang,angvel,nnet_file_name)
	# communication:
	netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
	netcheck.setLowerBound(netcheck.outputVars[0][9],ang)
	vals1, stats1 = netcheck.solve()
	del netcheck
	if (vals1=={}):
		print ("Found right bound for communication")
		ang_rb_comm = True
	else:
		print ("right bound unsafe for communication")
		rem = vals1[3]
		netcheck = try_to_find_input_check(vals1[1],vals1[2],nnet_file_name)
		#netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")
		else:
			vals1[4] = rem
			unsuccessful_points.append(vals1)


	# first check angvel to the BOTTOM: ----------------------------------------------------------------------
	netcheck = input_check(ang,angvel,nnet_file_name)
	# no communication:
	netcheck.setLowerBound(netcheck.outputVars[0][11],0.0)
	netcheck.setUpperBound(netcheck.outputVars[0][8],-angvel)
	vals1, stats1 = netcheck.solve()
	del netcheck
	if (vals1=={}):
		print ("Found bottom bound for no communication")
		angvel_bot_nocomm = True
	else:
		print ("bottom bound unsafe for no communication")
		rem = vals1[3]
		netcheck = try_to_find_input_check(vals1[1],vals1[2],nnet_file_name)
		#netcheck.setLowerBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")
		else:
			vals1[4] = rem
			unsuccessful_points.append(vals1)


	netcheck = input_check(ang,angvel,nnet_file_name)
	# communication:
	netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
	netcheck.setUpperBound(netcheck.outputVars[0][10],-angvel)
	vals1, stats1 = netcheck.solve()
	del netcheck
	if (vals1=={}):
		print ("Found bottom bound for communication")
		angvel_bot_comm = True
	else:
		print ("bottom bound unsafe for communication")
		rem = vals1[3]
		netcheck = try_to_find_input_check(vals1[1],vals1[2],nnet_file_name)
		#netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")
		else:
			vals1[4] = rem
			unsuccessful_points.append(vals1)


	# then check angvel to the TOP: --------------------------------------------------------------------------
	netcheck = input_check(ang,angvel,nnet_file_name)
	# no communication:
	netcheck.setLowerBound(netcheck.outputVars[0][11],0.0)
	netcheck.setLowerBound(netcheck.outputVars[0][8],angvel)
	vals1, stats1 = netcheck.solve()
	del netcheck
	if (vals1=={}):
		print ("Found top bound for no communication")
		angvel_top_nocomm = True
	else:
		print ("top bound unsafe for no communication")
		netcheck = try_to_find_input_check(vals1[1],vals1[2],nnet_file_name)
		rem = vals1[3]
		#netcheck.setLowerBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")
		else:
			vals1[4] = rem
			unsuccessful_points.append(vals1)


	netcheck = input_check(ang,angvel,nnet_file_name)
	# communication:
	netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
	netcheck.setLowerBound(netcheck.outputVars[0][10],angvel)
	vals1, stats1 = netcheck.solve()
	del netcheck
	if (vals1=={}):
		print ("Found top bound for communication")
		angvel_top_comm = True
	else:
		print ("top bound unsafe for communication")
		rem = vals1[3]
		netcheck = try_to_find_input_check(vals1[1],vals1[2],nnet_file_name)
		#netcheck.setUpperBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			input("IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")
		else:
			vals1[4] = rem
			unsuccessful_points.append(vals1)

	print("ang_lb_comm " + str(ang_lb_comm))
	print("ang_lb_nocomm " + str(ang_lb_nocomm))
	print("ang_rb_comm " + str(ang_rb_comm))
	print("ang_rb_nocomm " + str(ang_rb_nocomm))
	print("angvel_top_comm " + str(angvel_top_comm))
	print("angvel_top_nocomm " + str(angvel_top_nocomm))
	print("angvel_bot_comm " + str(angvel_bot_comm))
	print("angvel_bot_nocomm " + str(angvel_bot_nocomm))
	print (len(unsuccessful_points))

	time.sleep(5.0)

	for i in range(len(unsuccessful_points)):
		length_all = len(unsuccessful_points[i])
		print (unsuccessful_points[i][0])
		print (unsuccessful_points[i][1])
		print (unsuccessful_points[i][2])
		print (unsuccessful_points[i][3])
		print (unsuccessful_points[i][length_all-2])

	return unsuccessful_points



def check_whole_comp_comm_eff(nnet_file_name,ang,ang_save,angvel,angvel_save,uk,opt):
	# same as above but now also returns false if it is not resource efficient
	netcheck = try_to_find_input_comp_check(ang,angvel,uk,nnet_file_name)
	idx = 7
	if (opt==1):
		idx = 9
	netcheck.setLowerBound(netcheck.outputVars[0][idx],-ang_save)
	netcheck.setUpperBound(netcheck.outputVars[0][idx],ang_save)
	netcheck.setLowerBound(netcheck.outputVars[0][idx+1],-angvel_save)
	netcheck.setUpperBound(netcheck.outputVars[0][idx+1],angvel_save)
	vals1, stats1 = netcheck.solve()
	del netcheck
	if not(vals1=={}):
		# this means that it is fine
		if (opt==1):
			# if option was communicating check if no comm also possible:
			check, val = check_whole_comp_comm_eff(nnet_file_name,ang,ang_save,angvel,angvel_save,uk,0)
			if (check):
				# Then option 0 is also possible:
				return True, "no_comm_also_works"

		return True,[]
	else:
		#input ("WAIT")
		netcheck = try_to_find_input_check(ang,angvel,nnet_file_name)
		#netcheck.setLowerBound(netcheck.outputVars[0][11],0.0)
		netcheck.setLowerBound(netcheck.outputVars[0][7],-ang_save)
		netcheck.setUpperBound(netcheck.outputVars[0][7],ang_save)
		netcheck.setLowerBound(netcheck.outputVars[0][8],-angvel_save)
		netcheck.setUpperBound(netcheck.outputVars[0][8],angvel_save)
		vals1, stats1 = netcheck.solve()
		del netcheck
		if (vals1=={}):
			# if not capable of solving this problem it will never work,...
			print (ang)
			print (angvel)
			input ("IT IS IMPOSSIBLE TO SOLVE THIS PROBLEM")
		else:
			return False, vals1
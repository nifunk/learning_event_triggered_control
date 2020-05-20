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
from eval_code import add_render
from multiprocessing import Process, Queue

# Note: this file is used for visualization
# Further it exports the results from the rollout to a pickle file called, eval_model_param ... .pkl
# The pickle file is placed in exactly the same directory as from where the model is loaded.

def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        episode_len = 0,
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        num_options=1,
        app='',
        saves=False,
        wsaves=False,
        epoch=-1,
        seed=1,
        dc=0,
        path='',
        render=0,
        official=0,
        orig_ppo=0
        ):
    
    # if using PPOC implementation including the event-trigger, the variable below has to be set true
    event_trigger = False

    # if you want to use an LQR agent, set variable below true
    use_LQR_agent = False
    
    if (use_LQR_agent):
        # The lines below determine, if LQR is used, which LQR verion in loaded.
        from nfunk.LQR.LQR_standart_agent import LQR_standart_agent
        from nfunk.LQR.LQR_state_trigger_agent import LQR_state_trigger_agent
        from nfunk.LQR.LQR_input_trigger_agent import LQR_input_trigger_agent
        from nfunk.LQR.LQR_state_diff_trigger_agent import LQR_state_diff_trigger_agent

        #LQR_agent = LQR_standart_agent()
        #comm_pen = "LQR_standart"
        #LQR_agent = LQR_state_trigger_agent()
        #comm_pen = "LQR_state_trigger_agent_2_0_deg_2_norm"
        #LQR_agent = LQR_input_trigger_agent()
        #comm_pen = "LQR_input_trigger_agent_0_98_FIRST_MODE"
        LQR_agent = LQR_state_diff_trigger_agent()
        comm_pen = "LQR_state_diff_trigger_agent_0_75"
    
    max_episode = episode_len
    render = bool(render)
    if (render):
        # render the second image (communication decision) in seperate process
        q = Queue()
        p = Process(target=test, args=(q,))
        p.start()

    try:
        # just ensure here that nothin of importance is overwrtitten!!!
        saves = False
        wsaves = False
        official = bool(official) # denotes whether the official implementation is used or not,..
        orig_ppo = bool(orig_ppo) # not even options framework,...

        optim_batchsize_ideal = optim_batchsize 
        np.random.seed(seed)
        tf.set_random_seed(seed)
        env.seed(seed)

        ### Retrieve the name of the environment to later load the right model:
        gamename = env.spec.id[:-3].lower()
        env_name = env.spec.id[:-3].lower()
        gamename += 'seed' + str(seed)
        gamename += app

        # decide if you only want to use option 1 only
        opt1_only = False


        # Setup losses and stuff
        # ----------------------------------------
        ob_space = env.observation_space
        ac_space = env.action_space

        # if we use the hierarchical learning the dimensions have to be increased!
        if not(official) or event_trigger:
            ob_space.shape =((ob_space.shape[0] + ac_space.shape[0]),)

        if not(use_LQR_agent):
            pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
            

            U.initialize()


            saver = tf.train.Saver(max_to_keep=10000)
            print("Loading weights from iteration: " + str(epoch))

            comm_pen = "unknown"

            if (epoch==-5):
                filename = path + 'best.ckpt'
            else:    
                filename = path + '{}_epoch_{}.ckpt'.format(gamename,epoch)
            # This lines exploit the naming of the file to associate the comm_pen. It is not needed for running
            if (("ESCH-1-" in path) and ("-nI" in path)):
                comm_pen = path.split("ESCH-1-")[-1]
                comm_pen = comm_pen.split("-nI")[0]

            saver.restore(U.get_session(),filename)
            ###    

        if (render):
            input ("Press a button to actually start the visualization")


        # Prepare for rollouts
       
        # select whether a stochastic or deterministic policy should be used
        stochastic = True
        # select the number of rollouts
        num_runs = 10

        # Those arrays log the essential information during the rollout
        traj = -100*np.ones((num_runs,7,max_episode))   # 7 due to theta, theta_dot, input, comm, current reward, option, termination
        rew_arr = -100*np.ones((num_runs,2))                  # 2 due to reward and communication savings
        run_info = -100*np.ones((num_runs,3,max_episode))   # 2 due to option, run reward, acc run reward

        curr_run_num = 0

        t = 0
        acc_run_rew = 0
        ac = env.action_space.sample() # not used, just so we have the datatype

        max_action = env.action_space.high
        new = True # marks if we're on first timestep of an episode
        ob = env.reset()

        # store the shapes!
        ob_env_shape = np.shape(ob)
        ac_env_shape = np.shape(ac)

        if not (use_LQR_agent):
            if not(official) or event_trigger:
                ac = pi.reset_last_act().eval()
        else:
            LQR_agent.reset_last_act(ob[:ob_env_shape[0]])
        ob = np.concatenate((ob,ac))

        cur_ep_ret = 0 # return in current episode
        cur_ep_len = 0 # len of current episode

        if not(use_LQR_agent):
            # do the stuff for NN
            if not(orig_ppo):
                if (official):
                    if (event_trigger):
                        option = pi.get_option(ob)
                    else:
                        option = pi.get_option(ob[:ob_env_shape[0]])
                else:
                    option = pi.get_option(ob)
            else:
                # original ppo always assumes option 1
                option = 1
        else:
            # LQR let the option choose
            option = LQR_agent.get_option(ob[:ob_env_shape[0]])

        curr_opt_duration = 0.
        rew_total = 0.0
        comm_total = 0.0
        timesteps = 0.0
        factor = 1.0
        # Set discount factor to 0.99
        disc = 0.99

        while (curr_run_num<num_runs):
            prevac = ac
            ob[ob_env_shape[0]:] = ac
            if (opt1_only):
                option = 1

            if not(use_LQR_agent):
                if not(official):
                    # FROM here: this part is used for customly trained policies using our own algorithm

                    # Set this to true if you want to evaluate a model that has been retrained using our verification framework
                    if (False):

                        # Set this to true of you want to evaluate a model that has been retrained using communication only
                        # Set to false if evaluating a retraining file with both, communication and no communication
                        if (False):
                            # this explicitly uses the retrained (stability) version:
                            print ("CAREFUL NO RANDOMACTION -> VERIFICATION MODE")
                            ac = pi._act_mean([ob],[option])[0][0]
                        else:
                            print ("CAREFUL NO RANDOMACTION -> VERIFICATION MODE")
                            dec = pi._get_op_orig([ob])[0][0][0]

                            if (dec>0):
                                option = 0
                                ac = prevac
                            else:
                                option = 1
                                ac = pi._act_mean([ob],[option])[0][0]

                    else:
                        # normal operation:
                        ac, vpred, feats,logstd = pi.act(stochastic, ob, option)
                else:


                    if not(orig_ppo):
                        # FROM here: this code is for the case of PPOC implementation
                        if (event_trigger):

                            # If this is set true, communication will be skipped stochastically
                            if (False):
                                if (t==0):
                                    print ("Careful: stochastic capping of communication")
                                if (np.random.uniform()<0.2):
                                    option = 0
                                else:
                                    option = 1
                            ac, vpred, feats,logstd = pi.act(stochastic, ob, option)
                        else:
                            ac, vpred, feats,logstd = pi.act(stochastic, ob[:ob_env_shape[0]], option)
                    else:
                        # THIS is for the original ppo implementation
                        ac, vpred  = pi.act(stochastic, ob[:ob_env_shape[0]])
                        feats = 0.0
                        logstd = 0.0
            else:
                # This executes the LQR
                ac = LQR_agent.act(ob[:ob_env_shape[0]],option)
                


            # If set to true, this means that we deterministically save communication
            if (False):
                if (t==0):
                    print ("Careful: deterministic capping of communication")
                if ((t+1)%2==0):  # only comm every nth timeslot -> save(1/n)%
                    ac = prevac
                    option = 0

            # If set to true, this means that we stochastically save communication:
            if (False):
                if (t==0):
                    print ("!!!!!!-----Careful: stochastic capping of communication-----!!!!!")
                if (np.random.uniform()<0.4):
                    ac = prevac
                    option = 0

            # This applies the control action to the environment
            # Note: only for our algorithm the action to be applied is scaled to [-1,1]
            if not(use_LQR_agent):
                if not(official):
                    ob[:ob_env_shape[0]], rew, new, _ = env.step(max_action*np.copy(ac))
                else:
                    ob[:ob_env_shape[0]], rew, new, _ = env.step(np.copy(ac))
            else:
                ob[:ob_env_shape[0]], rew, new, _ = env.step(np.copy(ac))

            # These commands log the control reward for the pendulum
            # the distance covered by the cheetah or ant
            if (env_name=="halfcheetah"):
                rew_run = _['reward_run']
                acc_run_rew += rew
                run_info[curr_run_num,0,t] = option
                run_info[curr_run_num,1,t] = rew
                run_info[curr_run_num,2,t] = acc_run_rew

            if (env_name=="ant" or env_name=="antnff"):
                rew_run = _['reward_forward']
                acc_run_rew += rew_run
                run_info[curr_run_num,0,t] = option
                run_info[curr_run_num,1,t] = rew
                run_info[curr_run_num,2,t] = acc_run_rew

            if (env_name=="pendulumnf"):
                acc_run_rew += rew
                run_info[curr_run_num,0,t] = option
                run_info[curr_run_num,1,t] = rew
                run_info[curr_run_num,2,t] = acc_run_rew

            comm_total += option
            timesteps +=1
            rew_total += factor*rew
            factor = factor*disc


            if (render):
                # here print the communication decision and enforce 20Hz in case of rendering
                q.put(option)
                env.render()
                print (option)
                time.sleep(0.05)
            
            curr_opt_duration += 1

            # If the policy has a termination function, i.e. PPOC, determine here whether one wants to terminate or not
            # Otherwise set termination simply to True
            if (official) and not(orig_ppo):
                if (event_trigger):
                    term = pi.get_term([ob],[option])[0][0]
                else:
                    term = pi.get_term([ob[:ob_env_shape[0]]],[option])[0][0]
            else:
                term = True

            # here: log the normal reward, i.e. the overall reward and for the pendulum save the exact trajectory
            if (env_name=='pendulumnf'):
                traj[curr_run_num,:,t] = [np.arctan2(ob[1],ob[0]),ob[2], ac[0], option,rew_total, option, term]
            else:
                traj[curr_run_num,:,t] = [ob[0],0.0, ac[0], option,rew_total, option, term]

            # In case of termination -> determine which action should be executed next:
            if term:            
                curr_opt_duration = 0.
                if not(use_LQR_agent):
                    if not(orig_ppo):
                        if (official):
                            if (event_trigger):
                                option = pi.get_option(ob)
                            else:
                                option = pi.get_option(ob[:ob_env_shape[0]])
                        else:
                            option = pi.get_option(ob)
                    else:
                        option = 1  # indicate there is always communication
                else:
                    option = LQR_agent.get_option(ob[:ob_env_shape[0]])
          
            cur_ep_ret += rew
            cur_ep_len += 1


            # This condition is triggered when the rollout is finished, either on completion as time passed or due to the environment signalling
            # termination
            if new or t==max_episode-1:
                print ("NEW ROLLOUT STARTING")
                cur_ep_ret = 0
                cur_ep_len = 0
                if (t==max_episode-1):
                    # This means it finished regularily:
                    rew_arr[curr_run_num,0] = rew_total
                    rew_arr[curr_run_num,1] = 1-comm_total/timesteps
                else:
                    # The rollout finished through the environment -> the agent failed
                    traj[curr_run_num,4,t:] = [-9999]
                    rew_arr[curr_run_num,0] = rew_total - 10000
                    rew_arr[curr_run_num,1] = 1-comm_total/timesteps
                    print ("Terminated earlier!!!")                
                print("Episode finished after {} timesteps".format(t+1))
                print ("Avg comm savings: " + str(1-comm_total/timesteps))
                print ("Distance traveled: " + str(acc_run_rew))
                if (render):
                    input ("Press key before starting again")
                curr_run_num += 1
                t = 0
                acc_run_rew = 0
                comm_total = 0
                timesteps = 0

                # In case of restarting, reset the environments, decide which option to execute, etc.
                ob[:ob_env_shape[0]] = env.reset()
                if not(use_LQR_agent):
                    if not(official):
                        ob[ob_env_shape[0]:] = pi.reset_last_act().eval()
                        ac = pi.reset_last_act().eval()
                    if not(orig_ppo):
                        if (official):
                            if (event_trigger):
                                option = pi.get_option(ob)
                                ac = pi.reset_last_act().eval()
                            else:
                                option = pi.get_option(ob[:ob_env_shape[0]])
                        else:
                            option = pi.get_option(ob)
                    else:
                        option = 1  # indicate there is always communication
                else:
                    LQR_agent.reset_last_act(ob[:ob_env_shape[0]])
                    option = LQR_agent.get_option(ob[:ob_env_shape[0]])

            else:
                t += 1

        # export the logged data:
        filename = (path + "eval_model_param"+str(epoch)+("_commpen")+comm_pen+("_seed")+str(seed)+".pkl")
        fileObject = open(filename,'wb')
        pkl.dump(traj,fileObject)
        pkl.dump(rew_arr, fileObject)
        pkl.dump(run_info, fileObject)

        fileObject.close()

        print ("execution is now finished,...")

    except KeyboardInterrupt:
            p.terminate()
            print ("finished on ctrl c")



def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def test(q):
    # This function provides the rendering functionality
    add_render_obj = add_render.AddRender()

    while True:
        try:
            a = q.get()
            #print ("a" +str(a))
            add_render_obj.set_option(a)
            add_render_obj.render_new()
        except:
            print ("fail")
            r=0




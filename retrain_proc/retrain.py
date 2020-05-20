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

from exportpol import do_export
from checkpol import build_correct
from checkpol import check_whole
from checkpol import check_whole_comp_comm_eff

import sobol_seq

def traj_segment_generator(pi, env, horizon, stochastic, num_options,saves,results,rewbuffer,dc):
    max_action = env.action_space.high
    t = 0
    glob_count = 0
    glob_count_thresh = -1
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    ob_env_shape = np.shape(ob)
    ac_env_shape = np.shape(ac)

    ac = pi.reset_last_act().eval()

    ob = np.concatenate((ob,ac))

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    horizon = 500 #500
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    realrews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    wrong_checks = np.zeros(1, 'int32')
    news = np.zeros(horizon, 'int32')
    opts = np.zeros(horizon, 'int32')
    incorrect = np.zeros(horizon, 'int32')
    opts_val = np.zeros(horizon, 'float32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    # These are params that need to be predefined!
    # load the previously exported file:
    nnet_file_name = 'standart_all_comp.nnet'
    # define the angular range and the safe angular range
    ang = 1*2.5*0.0174533 #1.0*0.0174533 # was 3.5 # 7.5 and 1.5 works!!!
    ang_save = ang - 0.0001
    # define the angular velocity and the safe angular velocity range
    angvel = 1*5.0*0.0174533 #2.0*0.0174533 # was 3
    angvel_save = angvel - 0.0001

    while True:
        #if (t==horizon):
        if (t>0):
            yield {"ob" : obs, "rew" : rews, "realrew": realrews, "vpred" : vpreds, "new" : news,
                "ac" : acs, "opts" : opts, "opts_val" : opts_val, "incorrect" : incorrect, "wrong_checks" : wrong_checks}
        t = 0
        # first do the check:
        unsucess = check_whole(nnet_file_name,ang,ang_save,angvel,angvel_save)
        #unsucess = []
        if (True):
            if (len(unsucess)==0):
                print ("YES WE DID IT -> CONVERGENCE WAS REACHED!")
                wrong_checks[0] = 0
            else:
                wrong_checks[0] = len(unsucess)
                for i in range(len(unsucess)):
                    obs[t] = [unsucess[i][0],unsucess[i][1],unsucess[i][2],unsucess[i][4]]
                    acs[t] = unsucess[i][3]
                    incorrect[t] = 1
                    opts_val[t] = -2.0
                    opts[t] = 1
                    t += 1

        sample = None

        # idea was here to sample locally around unsuccesfull points:
        num_local_samples = 0#50
        if not(num_local_samples==0):
            for i in range(len(unsucess)):
                sample_new = np.random.uniform(low=-1.0, high=1.0, size=(num_local_samples,4))
                sample_new[:,0] = 1.0
                #sample_new[:,1] = np.clip(unsucess[i][1] + sample_new[:,1]*0.0174533,-ang,ang)
                sample_new[:,1] = np.clip(unsucess[i][1] + sample_new[:,1]*0.0,-ang,ang)
                #sample_new[:,2] = np.clip(unsucess[i][2] + sample_new[:,2]*2*0.0174533,-angvel,angvel)
                sample_new[:,2] = np.clip(unsucess[i][2] + sample_new[:,2]*0.0,-angvel,angvel)
                #sample_new[:,3] = np.clip(unsucess[i][3] + sample_new[:,2]*0.2,-1.0,1.0)
                sample_new[:,3] = np.clip(unsucess[i][3] + sample_new[:,2]*0.0,-1.0,1.0)
                if (sample is None):
                    sample = sample_new
                else:
                    sample = np.concatenate((sample,sample_new))
        
        # generate additional points globally using Sobol sequences
        size_to_gen = (horizon-(num_local_samples+1)*len(unsucess))
        if (True):
            # using sobol
            sample_new = np.zeros((size_to_gen,4))
            sample_new[:,1:4] = ((sobol_seq.i4_sobol_generate(3, size_to_gen))*2-1).reshape(size_to_gen,3)
            sample_new[:,0] = 1.0
            sample_new[:,1] = sample_new[:,1]*ang
            sample_new[:,2] = sample_new[:,2]*angvel
        if (sample is None):
            sample = sample_new
        else:
            sample = np.concatenate((sample,sample_new))

        overall_things = 0
        curr_correct = 0
        comm_savings = 0
        no_comm_samples = 0

        # now assign the correct values to the points:
        for i in range((np.shape(sample))[0]):
            overall_things += 1
            dec = pi._get_op_orig([sample[i]])[0][0][0]
            if (dec>0): #this means we select opt 0:
                curr_opt = 0
                comm_savings += 1
            else:
                curr_opt = 1
            
            # check the individual points:
            corr, point = check_whole_comp_comm_eff(nnet_file_name,sample[i,1],ang_save,sample[i,2],angvel_save,sample[i,3],curr_opt)

            if (corr):
                if (point=="no_comm_also_works"):
                    # special case: option 0 also works
                    obs[t] = [sample[i,0],sample[i,1],sample[i,2],sample[i,3]]
                    acs[t] = sample[i,3]
                    incorrect[t] = 1
                    opts_val[t] = 0.4#0.1 # make this less as not so important
                    opts[t] = 0
                    no_comm_samples += 1
                else:
                    curr_correct += 1
                    obs[t] = [sample[i,0],sample[i,1],sample[i,2],sample[i,3]]
                    acs[t] = sample[i,3]
                    incorrect[t] = 0
                    if (dec>0):
                        # this means no communication: assign value of 0.25
                        opts_val[t] = 0.4
                        no_comm_samples += 1
                        opts[t] = 0
                    else:
                        # for case of communication assign -2 -> important that this value is kept
                        opts_val[t] = -2.0
                        opts[t] = 1
                    
            else:
                # the point is incorrect -> also give -2 but additionally mark as incorrect
                obs[t] = [sample[i,0],sample[i,1],sample[i,2],sample[i,3]]
                acs[t] = point[3]
                incorrect[t] = 1
                opts_val[t] = -2.0
                opts[t] = 1
            t += 1

        print ("Rate correct samples: " + str(curr_correct/overall_things))
        print ("Comm savings: " + str(comm_savings/overall_things))
        print (no_comm_samples)
        time.sleep(5.0)






def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        num_options=1,
        app='',
        saves=True,
        wsaves=True,
        epoch=-1,
        seed=1,
        dc=0,
        path=None
        ):

    if (path is None):
        input ("Retraining is useless stop here!!!")

    optim_batchsize_ideal = optim_batchsize 
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)

    ### Book-keeping
    gamename = env.spec.id[:-3].lower()
    gamename += 'seed' + str(seed)
    gamename += app
    version_name = 'NORM-ACT-LOWER-LR-len-400-wNoise-update1-ppo-ESCH-1-1-0-nI' 

    # statically create folder called retrain
    dirname = path + "retrain/"
    
    # Specify the paths where results shall be written to:
    results_path = dirname + "retrain/"


    # if wsaves -> save the results
    if wsaves:
        first=True
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            first = False
        else:
            input ("ALREADY EXISTS,...")
    

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    max_action = env.action_space.high

    # add the dimension in the observation space!
    ob_space.shape =((ob_space.shape[0] + ac_space.shape[0]),)
    print (ob_space.shape)
    print (ac_space.shape)


    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy

    incorrect = tf.placeholder(dtype=tf.float32, shape=[None]) # Denotes whether sample is incorrect or not

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

    # pdb.set_trace()
    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    term_adv = U.get_placeholder(name='term_adv', dtype=tf.float32, shape=[None])

    # variable for supervised action
    ac_super = pi.pdtype.sample_placeholder([None]) # supervised label for action
    op_pi_super = tf.placeholder(name='op_pi_super', dtype=tf.float32, shape=[None]) # supervised label for policy over options


    ac = pi.pdtype.sample_placeholder([None])

    # fuer den op pi loss muss no was andres her
    ac_nn = pi.ac_mean
    #ac_nn =tf.Print(ac_nn,[ac_nn,ac_nn-ac_super])
    op_pi_nn = pi.op_pi_orig
    op_pi_nn = tf.reshape(op_pi_nn,[-1])
    #op_pi_super = tf.reshape(op_pi_super,[1,-1])
    op_pi_nn = tf.Print(op_pi_nn,[tf.math.sigmoid(5*(tf.math.abs(op_pi_nn-op_pi_super)-0.5)),(op_pi_nn-op_pi_super)])

    # only influence the policy over options
    # use a sigmoid weight to ensure, e.g. if the label is 60% but we use 100% this is also fine,..
    total_loss00 = tf.reduce_sum(tf.square(tf.math.sigmoid(5*(tf.math.abs(tf.stop_gradient(op_pi_nn)-op_pi_super)-0.5))*(op_pi_nn-op_pi_super)))
    # only affects the control action for the completely wrongly labelled point
    total_loss01 = tf.reduce_sum(tf.square(10*(ac_nn-ac_super)))
    losses00 = [total_loss00]
    losses01 = [total_loss01]


    var_list = pi.get_trainable_variables()
    term_list = var_list[6:8]
    
    lossandgrad00 = U.function([ob, option, ac_super, op_pi_super, incorrect], losses00 + [U.flatgrad(total_loss00, var_list)])
    lossandgrad01 = U.function([ob, option, ac_super, op_pi_super, incorrect], losses01 + [U.flatgrad(total_loss01, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])


    U.initialize()
    adam.sync()


    saver = tf.train.Saver(max_to_keep=10000)

    ### More book-kepping
    results=[]
    if saves:
        results = open(dirname + version_name + '_' + gamename +'_'+str(num_options)+'opts_'+'_results.csv','w')

        out = 'epoch,avg_reward'

        for opt in range(num_options): out += ',option {} dur'.format(opt)
        for opt in range(num_options): out += ',option {} std'.format(opt)
        for opt in range(num_options): out += ',option {} term'.format(opt)
        for opt in range(num_options): out += ',option {} adv'.format(opt)
        out+='\n'
        results.write(out)
        # results.write('epoch,avg_reward,option 1 dur, option 2 dur, option 1 term, option 2 term\n')
        results.flush()

    if epoch >= 0:
        
        print("Loading weights from iteration: " + str(epoch))

        filename = path + '{}_epoch_{}.ckpt'.format(gamename,epoch)
        saver.restore(U.get_session(),filename)
    ###    

    


    epoch = 0 #directly reset the epoch to 0!!!
    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    des_pol_op_ent = 0.1
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, num_options=num_options,saves=saves,results=results,rewbuffer=rewbuffer,dc=dc)

    datas = [0 for _ in range(num_options)]

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        assign_old_eq_new() # set old parameter values to new parameter values

        # export the NN parameters
        do_export(pi.get_trainable_variables(),env,pi)
        # use policy and system dynamics to build a file that Marabou can read:
        build_correct()

        # now get the samples:
        seg = seg_gen.__next__()

        
       

        ob, ac, opts, opts_val, incorrect, wrong_checks = seg["ob"], seg["ac"], seg["opts"], seg["opts_val"], seg["incorrect"], seg["wrong_checks"]

        # save the weights:
        if iters_so_far % 1 == 0 and wsaves:
            print("weights are saved...")
            filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,iters_so_far)
            save_path = saver.save(U.get_session(),filename)

            indices_incorrect = np.where(incorrect==1)[0]
            #if (np.shape(indices_incorrect)[0]==0):
            if (wrong_checks[0]==0):
                # there are no more incorrect results -> we are finished :)
                input ("----------FINISHED------------")
                break



        # perform the optimization:
        for opt in range(num_options):
            indices = np.where(opts==opt)[0]
            curr_size = indices.size


            if (opt==0):
                # or take all:
                # refine everything in here to not destabilize too greatly,...
                indices_incorrect = np.where((incorrect==1)|(incorrect==0))[0]
            else:
                # get incorrect and action
                indices_incorrect = np.where((incorrect==1)&(opts==opt))[0]
            indices_wanted = indices_incorrect
            if (True):
                curr_size = indices_wanted.size
                datas[opt] = d = Dataset(dict(ob=ob[indices_wanted], ac=ac[indices_wanted], opts_val=opts_val[indices_wanted], incorrect=incorrect[indices_wanted]), shuffle=not pi.recurrent)
            else:
                datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], opts_val=opts_val[indices], incorrect=incorrect[indices]), shuffle=not pi.recurrent)

            real_optim_batchsize = min(optim_batchsize,curr_size)#indices.size)

            if (real_optim_batchsize==0):
                optim_epochs = 0
            elif(opt==0):
                optim_epochs = 10
            else:
                optim_epochs = 100
            print("optim epochs:", optim_epochs)
            logger.log("Optimizing...")


            # Here we do a bunch of optimization epochs over the data
            print ("OPTION IS: " + str(opt))
            #for _ in range(optim_epochs):
            curr_iter = 0
            while (True):
                curr_iter += 1
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                loss_opt_0 = []
                temp_loss = []
                for batch in d.iterate_once(real_optim_batchsize):
                    # take different lossfuctions for option 0 and option 1
                    if (opt==0):
                        *newlosses, grads = lossandgrad00(batch["ob"], [opt], batch["ac"], batch["opts_val"], batch["incorrect"])
                        loss_opt_0.append(newlosses)
                        #input ("WAIT")
                    else:
                        *newlosses, grads = lossandgrad01(batch["ob"], [opt], batch["ac"], batch["opts_val"], batch["incorrect"])
                        temp_loss.append(newlosses)

                    adam.update(grads, optim_stepsize * cur_lrmult) 
                    losses.append(newlosses)
                
                # different terminating conditions of the optimization depending on the option
                if (opt==0):
                    print (np.mean(loss_opt_0))
                    if (curr_iter > 100):
                        break
                if (opt==1):
                    if ((np.mean(temp_loss)<0.01 and curr_iter>=100) or np.mean(temp_loss)<0.001 or curr_iter>500):
                        #time.sleep(1.0)
                        break
                    # more adaptive way of approaching:
                    if ((np.mean(temp_loss)<0.01 and curr_iter>=5000) or np.mean(temp_loss)<0.0001 or curr_iter>50000):
                        #time.sleep(1.0)
                        break


        iters_so_far += 1
        logger.record_tabular("iters_so_far", iters_so_far)
        logger.record_tabular("mean loss", np.mean(losses))
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

        #input ("SCHDOOOP")
        ### Book keeping
        if saves:
            out = "{},{}"
            for _ in range(num_options): out+=",{},{},{},{}"
            out+="\n"
            

            info = [iters_so_far, np.mean(losses)]
            for i in range(num_options): info.append(0)
            for i in range(num_options): info.append(0)
            for i in range(num_options): info.append(0)
            for i in range(num_options): 
                info.append(0)

            results.write(out.format(*info))
            results.flush()
        ###


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
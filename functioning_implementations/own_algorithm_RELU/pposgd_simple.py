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

def traj_segment_generator(pi, env, horizon, stochastic, num_options,saves,results,rewbuffer,dc):
    # sample state action pairs, i.e. sample rollouts on the real system
    
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
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    realrews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    opts = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    option = pi.get_option(ob)
    if (glob_count<glob_count_thresh):
        option = 1

    optpol_p=[]    
    term_p=[]
    value_val=[]
    opt_duration = [[] for _ in range(num_options)]
    logstds = [[] for _ in range(num_options)]
    curr_opt_duration = 0.


    while True:
        # in here collect the state action pairs:

        prevac = ac
        # remember u[k-1]
        ob[ob_env_shape[0]:] = ac
        # evaluate policy and recieve action
        ac, vpred, feats,logstd = pi.act(stochastic, ob, option)
        logstds[option].append(logstd)
        
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "realrew": realrews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "opts" : opts, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, 'term_p': term_p, 'value_val': value_val,
                     "opt_dur": opt_duration, "optpol_p":optpol_p, "logstds": logstds}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            term_p  = []
            value_val=[]
            opt_duration = [[] for _ in range(num_options)]
            logstds = [[] for _ in range(num_options)]
            curr_opt_duration = 0.
            glob_count += 1

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        opts[i] = option
        acs[i] = ac
        prevacs[i] = prevac



        # Careful: Without this "copy" operation the variable ac is actually modified...
        # Apply the action to the environment
        ob[:ob_env_shape[0]], rew, new, _ = env.step(max_action*np.copy(ac))
        # IMPORTANT: here the reward is modified in case of communication, i.e., executing option 1
        comm_penalty = 1.0
        if (option==1):
            rew = rew - comm_penalty
        # for pendulum: comm penalty 0.0 0.5 1.0 5.0 rew scaling 1.0 0.85 0.75 0.55
        rew = rew*1.0


        rew = rew/10 if num_options > 1 else rew # To stabilize learning.
        rews[i] = rew
        realrews[i] = rew

        curr_opt_duration += 1

        ### Book-keeping
        t_p = []
        v_val = []
        for oopt in range(num_options):
            v_val.append(pi.get_vpred([ob],[oopt])[0][0])
            t_p.append(pi.get_tpred([ob],[oopt])[0][0])
        term_p.append(t_p)
        optpol_p.append(pi._get_op([ob])[0][0])
        value_val.append(v_val)
        term = pi.get_term([ob],[option])[0][0]


        # in case of termination, decide which option to execute next:
        if term:            
            opt_duration[option].append(curr_opt_duration)
            curr_opt_duration = 0.
            option = pi.get_option(ob)
            if (glob_count<glob_count_thresh):
                option = 1

      
        cur_ep_ret += rew*10 if num_options > 1 else rew
        cur_ep_len += 1


        if new:
            # if new rollout starts -> reset last action and start anew
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob[:ob_env_shape[0]] = env.reset()
            ob[ob_env_shape[0]:] = pi.reset_last_act().eval()
            ac = pi.reset_last_act().eval()
            option = pi.get_option(ob)
            if (glob_count<glob_count_thresh):
                option = 1
        t += 1

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["tdlamret"] = seg["adv"] + seg["vpred"]

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
        saves=False,
        wsaves=False,
        epoch=-1,
        seed=1,
        dc=0
        ):


    optim_batchsize_ideal = optim_batchsize 
    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)

    ### Book-keeping
    gamename = env.spec.id[:-3].lower()
    gamename += 'seed' + str(seed)
    gamename += app
    # This variable: "version name, defines the name of the training"
    version_name = 'ALL-ReLU-1-1-0-nI-Start-TOP-NONSTOP' 

    dirname = '{}_{}_{}opts_saves/'.format(version_name,gamename,num_options)
    print (dirname)
    # retrieve everything using relative paths. Create a train_results folder where the repo has been cloned
    dirname_rel = os.path.dirname(__file__)
    splitted = dirname_rel.split("/")
    dirname_rel = ("/".join(dirname_rel.split("/")[:len(splitted)-3])+"/")
    dirname = dirname_rel + "train_results/" + dirname

    # if saving -> create the necessary directories
    if wsaves:
        first=True
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            first = False

        # copy also the original files into the folder where the training results are stored

        files = ['pposgd_simple.py','mlp_policy.py','run_mujoco.py']
        first = True
        for i in range(len(files)):
            src = os.path.join(dirname_rel,'baselines/baselines/ppo1/') + files[i]
            print (src)
            #dest = os.path.join('/home/nfunk/results_NEW/ppo1/') + dirname
            dest = dirname + "src_code/"
            if (first):
                os.makedirs(dest)
                first = False
            print (dest)
            shutil.copy2(src,dest)
        # brute force copy normal env file at end of copying process:
        src = os.path.join(dirname_rel,'nfunk/envs_nf/pendulum_nf.py')         
        shutil.copy2(src,dest)
        os.makedirs(dest+"assets/")
        src = os.path.join(dirname_rel,'nfunk/envs_nf/assets/clockwise.png')
        shutil.copy2(src,dest+"assets/")
    ###


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
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function 
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    pol_ov_op_ent = tf.placeholder(dtype=tf.float32, shape=None) # Entropy coefficient for policy over options


    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon for PPO


    # setup observation, option and terminal advantace
    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    term_adv = U.get_placeholder(name='term_adv', dtype=tf.float32, shape=[None])

    # create variable for action
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    # propability of choosing action under new policy vs old policy (PPO)
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) 
    # advantage of choosing the action
    atarg_clip = atarg
    # surrogate 1:
    surr1 = ratio * atarg_clip #atarg # surrogate from conservative policy iteration
    # surrogate 2:
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg_clip 
    # PPO's pessimistic surrogate (L^CLIP)
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) 

    # Loss on the Q-function
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    # calculate the total loss
    total_loss = pol_surr + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    # calculate logarithm of propability of policy over options
    log_pi = tf.log(tf.clip_by_value(pi.op_pi, 1e-5, 1.0))
    # calculate logarithm of propability of policy over options old parameter
    old_log_pi = tf.log(tf.clip_by_value(oldpi.op_pi, 1e-5, 1.0))
    # calculate entropy of policy over options
    entropy = -tf.reduce_sum(pi.op_pi * log_pi, reduction_indices=1)

    # calculate the ppo update for the policy over options:
    ratio_pol_ov_op = tf.exp(tf.transpose(log_pi)[option[0]] - tf.transpose(old_log_pi)[option[0]]) # pnew / pold
    term_adv_clip = term_adv 
    surr1_pol_ov_op = ratio_pol_ov_op * term_adv_clip # surrogate from conservative policy iteration
    surr2_pol_ov_op = U.clip(ratio_pol_ov_op, 1.0 - clip_param, 1.0 + clip_param) * term_adv_clip #
    pol_surr_pol_ov_op = - U.mean(tf.minimum(surr1_pol_ov_op, surr2_pol_ov_op)) # PPO's pessimistic surrogate (L^CLIP)
    
    op_loss = pol_surr_pol_ov_op - pol_ov_op_ent*tf.reduce_sum(entropy)

    # add loss of policy over options to total loss
    total_loss += op_loss
    
    var_list = pi.get_trainable_variables()
    term_list = var_list[6:8]

    # define function that we will later do gradien descent on
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult,option, term_adv,pol_ov_op_ent], losses + [U.flatgrad(total_loss, var_list)])
    
    # define adam optimizer
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    # define function that will assign the current parameters to the old policy
    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, option], losses)


    U.initialize()
    adam.sync()


    # NOW: all the stuff for training was defined, from here on we start with the execution:

    # initialize "savers" which will store the results
    saver = tf.train.Saver(max_to_keep=10000)
    saver_best = tf.train.Saver(max_to_keep=1)


    ### Define the names of the .csv files that are going to be stored
    results=[]
    if saves:
        results = open(dirname + version_name + '_' + gamename +'_'+str(num_options)+'opts_'+'_results.csv','w')
        results_best_model = open(dirname + version_name + '_' + gamename +'_'+str(num_options)+'opts_'+'_bestmodel.csv','w')


        out = 'epoch,avg_reward'

        for opt in range(num_options): out += ',option {} dur'.format(opt)
        for opt in range(num_options): out += ',option {} std'.format(opt)
        for opt in range(num_options): out += ',option {} term'.format(opt)
        for opt in range(num_options): out += ',option {} adv'.format(opt)
        out+='\n'
        results.write(out)
        # results.write('epoch,avg_reward,option 1 dur, option 2 dur, option 1 term, option 2 term\n')
        results.flush()

    # speciality: if running the training with epoch argument -> a model is loaded
    if epoch >= 0:
        
        dirname = '{}_{}opts_saves/'.format(gamename,num_options)
        print("Loading weights from iteration: " + str(epoch))

        filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,epoch)
        saver.restore(U.get_session(),filename)
    ###    


    # start training
    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    des_pol_op_ent = 0.1    # define policy over options entropy scheduling
    max_val = -100000       # define max_val, this will be updated to always store the best model
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

        # Sample (s,a)-Transitions
        seg = seg_gen.__next__()
        # Calculate A(s,a,o) using GAE
        add_vtarg_and_adv(seg, gamma, lam)


        # calculate information for logging
        opt_d = []
        for i in range(num_options):
            dur = np.mean(seg['opt_dur'][i]) if len(seg['opt_dur'][i]) > 0 else 0.
            opt_d.append(dur)

        std = []
        for i in range(num_options):
            logstd = np.mean(seg['logstds'][i]) if len(seg['logstds'][i]) > 0 else 0.
            std.append(np.exp(logstd))
        print("mean opt dur:", opt_d)             
        print("mean op pol:", np.mean(np.array(seg['optpol_p']),axis=0))         
        print("mean term p:", np.mean(np.array(seg['term_p']),axis=0))
        print("mean value val:", np.mean(np.array(seg['value_val']),axis=0))
       

        ob, ac, opts, atarg, tdlamret = seg["ob"], seg["ac"], seg["opts"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy
        if hasattr(pi, "ob_rms_only"): pi.ob_rms_only.update(ob[:,:-ac_space.shape[0]]) # update running mean/std for policy
        assign_old_eq_new() # set old parameter values to new parameter values

        # if iterations modulo 1000 -> adapt entropy scheduling coefficient
        if (iters_so_far+1)%1000 == 0:
            des_pol_op_ent = des_pol_op_ent/10

        # every 50 epochs save the best model
        if iters_so_far % 50 == 0 and wsaves:
            print("weights are saved...")
            filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,iters_so_far)
            save_path = saver.save(U.get_session(),filename)

        # adaptively save best model -> if current reward is highest, save the model
        if (np.mean(rewbuffer)>max_val) and wsaves:
            max_val = np.mean(rewbuffer)
            results_best_model.write('epoch: '+str(iters_so_far) + 'rew: ' + str(np.mean(rewbuffer)) + '\n')
            results_best_model.flush()
            filename = dirname + 'best.ckpt'.format(gamename,iters_so_far)
            save_path = saver_best.save(U.get_session(),filename)



        # minimum batch size:
        min_batch=160 
        t_advs = [[] for _ in range(num_options)]
        
        # select all the samples concering one of the options
        # Note: so far the update is that we first use all samples from option 0 to update, then we use all samples from option 1 to update
        for opt in range(num_options):
            indices = np.where(opts==opt)[0]
            print("batch size:",indices.size)
            opt_d[opt] = indices.size
            if not indices.size:
                t_advs[opt].append(0.)
                continue


            ### This part is only necessasry when we use options. We proceed to these verifications in order not to discard any collected trajectories.
            if datas[opt] != 0:
                if (indices.size < min_batch and datas[opt].n > min_batch):
                    datas[opt] = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)
                    t_advs[opt].append(0.)
                    continue

                elif indices.size + datas[opt].n < min_batch:
                    # pdb.set_trace()
                    oldmap = datas[opt].data_map

                    cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                    cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                    cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                    cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                    datas[opt] = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg), shuffle=not pi.recurrent)
                    t_advs[opt].append(0.)
                    continue

                elif (indices.size + datas[opt].n > min_batch and datas[opt].n < min_batch) or (indices.size > min_batch and datas[opt].n < min_batch):

                    oldmap = datas[opt].data_map
                    cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                    cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                    cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                    cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                    datas[opt] = d = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg), shuffle=not pi.recurrent)

                if (indices.size > min_batch and datas[opt].n > min_batch):
                    datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)

            elif datas[opt] == 0:
                datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)
            ###


            # define the batchsize of the optimizer:
            optim_batchsize = optim_batchsize or ob.shape[0]
            print("optim epochs:", optim_epochs)
            logger.log("Optimizing...")


            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):

                    # Calculate advantage for using specific option here
                    tadv,nodc_adv = pi.get_opt_adv(batch["ob"],[opt])
                    tadv = tadv if num_options > 1 else np.zeros_like(tadv)
                    t_advs[opt].append(nodc_adv)

                    # calculate the gradient
                    *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, [opt], tadv,des_pol_op_ent)

                    # perform gradient update
                    adam.update(grads, optim_stepsize * cur_lrmult) 
                    losses.append(newlosses)


        # do logging:
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

        ### Book keeping
        if saves:
            out = "{},{}"
            for _ in range(num_options): out+=",{},{},{},{}"
            out+="\n"
            

            info = [iters_so_far, np.mean(rewbuffer)]
            for i in range(num_options): info.append(opt_d[i])
            for i in range(num_options): info.append(std[i])
            for i in range(num_options): info.append(np.mean(np.array(seg['term_p']),axis=0)[i])
            for i in range(num_options): 
                info.append(np.mean(t_advs[i]))

            results.write(out.format(*info))
            results.flush()
        ###


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]



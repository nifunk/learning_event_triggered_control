from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

import os
import shutil
import glob

def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        #env.render()
        #time.sleep(0.05)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
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
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant',
        seed = 1 # annealing for stepsize parameters (epsilon and adam)
        ):

    # We want to log:
    num_options = 1 # Hacky solution -> enables to use same logging!
    epoch = -1
    saves = True
    wsaves = True

    ### Book-keeping
    gamename = env.spec.id[:-3].lower()
    gamename += 'seed' + str(seed)
    #gamename += app
    version_name = 'officialPPO' 

    dirname = '{}_{}_{}opts_saves/'.format(version_name,gamename,num_options)
    
    # retrieve everything using relative paths. Create a train_results folder where the repo has been cloned
    dirname_rel = os.path.dirname(__file__)
    splitted = dirname_rel.split("/")
    dirname_rel = ("/".join(dirname_rel.split("/")[:len(splitted)-3])+"/")
    dirname = dirname_rel + "train_results/" + dirname

    # Specify the paths where results shall be written to:
    src_code_path = dirname_rel + 'baselines/baselines/ppo1/'
    results_path = dirname_rel
    envs_path = dirname_rel + 'nfunk/envs_nf/'

    dirname = dirname_rel + "train_results/" + dirname

    print (dirname)
    #input ("wait here after dirname")

    if wsaves:
        first=True
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            first = False
        # while os.path.exists(dirname) and first:
        #     dirname += '0'

        files = ['pposgd_simple.py','mlp_policy.py','run_mujoco.py']
        first = True
        for i in range(len(files)):
            src = os.path.join(src_code_path) + files[i]
            print (src)
            #dest = os.path.join('/home/nfunk/results_NEW/ppo1/') + dirname
            dest = dirname + "src_code/"
            if (first):
                os.makedirs(dest)
                first = False
            print (dest)
            shutil.copy2(src,dest)
        # brute force copy normal env file at end of copying process:
        env_files = ['pendulum_nf.py','gym_scalar_nf.py','cartpole.py']
        for i in range(len(env_files)):
            src = os.path.join(envs_path+env_files[i])         
            shutil.copy2(src,dest)

    np.random.seed(seed)
    tf.set_random_seed(seed)
    env.seed(seed)


    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    saver = tf.train.Saver(max_to_keep=10000)
    saver_best = tf.train.Saver(max_to_keep=1)

    ### More book-kepping
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

    if epoch >= 0:
        
        print("Loading weights from iteration: " + str(epoch))

        filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,epoch)
        saver.restore(U.get_session(),filename)


    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    max_val = -100000
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

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

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values

        if iters_so_far % 50 == 0 and wsaves:
            print("weights are saved...")
            filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,iters_so_far)
            save_path = saver.save(U.get_session(),filename)


        if (np.mean(rewbuffer)>max_val) and wsaves:
            max_val = np.mean(rewbuffer)
            results_best_model.write('epoch: '+str(iters_so_far) + 'rew: ' + str(np.mean(rewbuffer)) + '\n')
            results_best_model.flush()
            filename = dirname + 'best.ckpt'.format(gamename,iters_so_far)
            save_path = saver_best.save(U.get_session(),filename)


        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult) 
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)            
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
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
            #for _ in range(num_options): #out+=",{},{},{},{}"
            out+="\n"
            

            info = [iters_so_far, np.mean(rewbuffer)]
            #for i in range(num_options): info.append(opt_d[i])
            #for i in range(num_options): info.append(std[i])
            #for i in range(num_options): info.append(np.mean(np.array(seg['term_p']),axis=0)[i])
            #for i in range(num_options): 
            #    info.append(np.mean(t_advs[i]))

            results.write(out.format(*info))
            results.flush()
        ###

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

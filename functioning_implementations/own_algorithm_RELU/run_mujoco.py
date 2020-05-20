# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import gym, logging
import pdb
import os
dirname_rel = os.path.dirname(__file__)
splitted = dirname_rel.split("/")
dirname_rel = ("/".join(dirname_rel.split("/")[:len(splitted)-3])+"/")
from baselines import logger
import sys
import sys
sys.path.append('../../../')
sys.path.append(dirname_rel)

def train(env_id, num_timesteps, seed, num_options,app, saves ,wsaves, epoch,dc):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    from gym.envs.registration import register
    # Potential Pendulum Env
    if (True):
        register(
            id='Pendulumnf-v0',
            entry_point='nfunk.envs_nf.pendulum_nf:PendulumEnv',
            max_episode_steps=400,  # was 200 maybe,...
            #kwargs = vars(args),
        )
        env = gym.make('Pendulumnf-v0')
    # Potential Scalar Env
    if (False):
        register(
            id='Scalarnf-v0',
            entry_point='nfunk.envs_nf.gym_scalar_nf:GymScalarEnv',
            max_episode_steps=400,
            #kwargs = vars(args),
        )
        env = gym.make('Scalarnf-v0')
    if (False):
        env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, dc=dc)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)

    if num_options ==1:
        optimsize=64
    elif num_options ==2:
        optimsize=32
    else:
        print("Only two options or primitive actions is currently supported.")
        sys.exit()

    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-5, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,dc=dc
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--opt', help='number of options', type=int, default=2) 
    parser.add_argument('--app', help='Append to folder name', type=str, default='')        
    parser.add_argument('--saves', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=False)    
    parser.add_argument('--epoch', help='Epoch', type=int, default=-1) 
    parser.add_argument('--dc', type=float, default=0.1)


    args = parser.parse_args()

    train(args.env, num_timesteps=10e6, seed=args.seed, num_options=args.opt, app=args.app, saves=args.saves, wsaves=args.wsaves, epoch=args.epoch,dc=args.dc)


if __name__ == '__main__':
    main()

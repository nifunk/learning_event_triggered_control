The files in here represent the original ppo1 implementation from baselines.

They have been adapted such that they can be used with the herein presented code.

Exemplary command for running the training:

python run_mujoco.py --env Ant-v2 --seed 100

Exemplary commands for evaluating the policy:
python run_visual.py --env=HalfCheetah-v2 --seed=500 --opt=1 --epoch=485 --path=PATH_TO_THE_RESULTS_FOLDER --render=1 --official=1 --orig_ppo=1

# PPOSGD

- Original paper: https://arxiv.org/abs/1707.06347
- Baselines blog post: https://blog.openai.com/openai-baselines-ppo/
- `mpirun -np 8 python -m baselines.ppo1.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
- `python -m baselines.ppo1.run_mujoco` runs the algorithm for 1M frames on a Mujoco environment.


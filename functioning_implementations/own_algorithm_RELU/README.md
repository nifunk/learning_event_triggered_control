The files in here represent the implementation of the algorithm presented in the paper, but only considering ReLU activation functions.

Before usage, we have to move the files into the baselines/baselines/ppo1 folder.

To train a model, execute: 

python run_mujoco.py --env Ant-v2 --seed 0 --app savename --opt 2 --saves --wsaves

To evaluate the model, execute:

python run_visual.py --env=HalfCheetah-v2 --seed=500 --opt=2 --app savename --epoch=485 --path=PATH_TO_THE_RESULTS_FOLDER --render=1




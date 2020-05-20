The files in here represent the implementation of the algorithm in the paper, without any event-triggering involved. Therefore, here we deal with the case where just the two policies are executed without any zero order hold. Thus, there is also no communication penalty involved.

Command to train a model:

python run_mujoco.py --env Ant-v2 --seed 0 --app savename --opt 2 --saves --wsaves

Command to execute a model:

python run_visual.py --env=HalfCheetah-v2 --seed=500 --opt=2 --epoch=485 --path=/home/niggi/results_ppoc_off_tryout/from_cluster/officialPPO_halfcheetahseed500_1opts_saves/ --render=1


